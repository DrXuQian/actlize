/***************************************************************************************************
 * Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>
#include <limits>

#include "cutlass/cutlass.h"

namespace cutlass::gemm::kernel::detail {

// Pure integer seam shared by production and the device-free L126 oracle.
// Keep device primitives, CuTe layouts and workspace policy out of this type:
// its only job is Marlin's scheduler-owned launch/decomposition/peer algebra.
struct MarlinStripeSchedulerCore {
  struct Params {
    uint64_t tiles_m_ = 0;
    uint64_t tiles_n_ = 0;
    uint64_t tiles_l_ = 0;
    uint64_t k_tiles_per_output_ = 0;
    uint64_t output_tiles_ = 0;
    uint64_t total_k_tiles_ = 0;
    uint64_t grid_blocks_ = 0;
    uint64_t active_blocks_ = 0;
    uint64_t iters_per_block_ = 0;
    bool valid_ = false;
  };

  struct WorkTileInfo {
    int32_t M_idx = -1;
    int32_t N_idx = -1;
    int32_t L_idx = -1;
    uint32_t K_idx = 0;
    uint32_t k_tile_count = 0;
    uint32_t slice_count = 0;
    uint32_t slice_idx = 0;
    uint64_t output_tile_idx = 0;
    uint64_t lock_idx = 0;
    uint64_t block_idx = 0;
    uint64_t linear_begin = 0;
    uint64_t linear_next = 0;
    uint64_t linear_end = 0;
    bool valid = false;

    CUTLASS_HOST_DEVICE bool is_valid() const { return valid; }
    CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() { return {}; }
  };

  CUTLASS_HOST_DEVICE static constexpr uint64_t ceil_div_u64(uint64_t x, uint64_t y) {
    return y == 0 ? 0 : x / y + uint64_t(x % y != 0);
  }

  CUTLASS_HOST_DEVICE static constexpr bool mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
      out = 0;
      return false;
    }
    out = a * b;
    return true;
  }

  CUTLASS_HOST_DEVICE static Params make_params_for_tiles(
      uint64_t tiles_m, uint64_t tiles_n, uint64_t tiles_l,
      uint64_t k_tiles, uint64_t cu_count) {
    Params p;
    p.tiles_m_ = tiles_m;
    p.tiles_n_ = tiles_n;
    p.tiles_l_ = tiles_l;
    p.k_tiles_per_output_ = k_tiles;
    bool ok = tiles_m > 0 && tiles_n > 0 && tiles_l > 0 &&
              k_tiles > 0 && cu_count > 0;
    ok = ok && tiles_m <= uint64_t(std::numeric_limits<int32_t>::max()) &&
         tiles_n <= uint64_t(std::numeric_limits<int32_t>::max()) &&
         tiles_l <= uint64_t(std::numeric_limits<int32_t>::max()) &&
         k_tiles <= uint64_t(std::numeric_limits<uint32_t>::max());
    ok = ok && mul_u64(tiles_m, tiles_n, p.output_tiles_);
    ok = ok && mul_u64(p.output_tiles_, tiles_l, p.output_tiles_);
    ok = ok && mul_u64(p.output_tiles_, k_tiles, p.total_k_tiles_);
    if (!ok) {
      return p;
    }

    // WorkTileInfo narrows M/N/L to int32_t and the cooperative passes q to
    // the barrier ABI as an int lock index.  Reject an unrepresentable
    // problem here; a zero launch returned later by get_grid_shape is not an
    // acceptable substitute for can_implement() failing closed.
    if (p.output_tiles_ > uint64_t(std::numeric_limits<int>::max())) {
      return p;
    }

    // Classic Marlin's measured safety rail: never multiply CU by occupancy.
    p.grid_blocks_ = p.output_tiles_ >= cu_count ? p.output_tiles_ : cu_count;
    if (p.grid_blocks_ > uint64_t(std::numeric_limits<unsigned>::max())) {
      return p;
    }
    p.iters_per_block_ = ceil_div_u64(p.total_k_tiles_, p.grid_blocks_);
    p.active_blocks_ = ceil_div_u64(p.total_k_tiles_, p.iters_per_block_);
    p.valid_ = p.grid_blocks_ > 0 && p.iters_per_block_ > 0 &&
               p.active_blocks_ <= p.grid_blocks_;
    return p;
  }

private:
  CUTLASS_HOST_DEVICE static WorkTileInfo make_work(
      Params const& p, uint64_t block_idx, uint64_t cursor, uint64_t stripe_end) {
    if (!p.valid_ || block_idx >= p.grid_blocks_ || cursor >= stripe_end ||
        cursor >= p.total_k_tiles_) {
      return WorkTileInfo::invalid_work_tile();
    }
    WorkTileInfo out;
    uint64_t const q = cursor / p.k_tiles_per_output_;
    uint64_t const k = cursor % p.k_tiles_per_output_;
    uint64_t const q_end = (q + 1) * p.k_tiles_per_output_;
    uint64_t const segment_end = stripe_end < q_end ? stripe_end : q_end;
    uint64_t const first_peer = (q * p.k_tiles_per_output_) / p.iters_per_block_;
    uint64_t const last_peer = (q_end - 1) / p.iters_per_block_;

    // Reverse peer order is part of Marlin's protocol, not presentation:
    // the highest-K/highest-block slice initializes the chain.
    out.slice_count = uint32_t(last_peer - first_peer + 1);
    out.slice_idx = uint32_t(last_peer - block_idx);
    out.output_tile_idx = q;
    out.lock_idx = q;
    out.block_idx = block_idx;
    out.K_idx = uint32_t(k);
    out.k_tile_count = uint32_t(segment_end - cursor);
    out.linear_begin = cursor;
    out.linear_next = segment_end;
    out.linear_end = stripe_end;

    uint64_t q_mn = q;
    out.N_idx = int32_t(q_mn % p.tiles_n_);
    q_mn /= p.tiles_n_;
    out.M_idx = int32_t(q_mn % p.tiles_m_);
    out.L_idx = int32_t(q_mn / p.tiles_m_);
    out.valid = q < p.output_tiles_ && uint64_t(out.L_idx) < p.tiles_l_ &&
                out.k_tile_count != 0 && block_idx >= first_peer && block_idx <= last_peer;
    return out.valid ? out : WorkTileInfo::invalid_work_tile();
  }

public:
  CUTLASS_HOST_DEVICE static WorkTileInfo get_work_for_block(
      Params const& p, uint64_t block_idx) {
    if (!p.valid_ || block_idx >= p.grid_blocks_) {
      return WorkTileInfo::invalid_work_tile();
    }
    uint64_t const begin = block_idx * p.iters_per_block_;
    uint64_t end = begin + p.iters_per_block_;
    if (end > p.total_k_tiles_ || end < begin) {
      end = p.total_k_tiles_;
    }
    return make_work(p, block_idx, begin, end);
  }

  CUTLASS_HOST_DEVICE static WorkTileInfo fetch_next_work(
      Params const& p, WorkTileInfo const& work) {
    return work.is_valid() && work.linear_next < work.linear_end
        ? make_work(p, work.block_idx, work.linear_next, work.linear_end)
        : WorkTileInfo::invalid_work_tile();
  }
};

} // namespace cutlass::gemm::kernel::detail
