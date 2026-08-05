/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cute/algorithm/tuple_algorithms.hpp"

namespace cutlass::gemm::collective::detail {

// Public type witness consumed by the launcher descriptor. It makes the cadence provider part of the policy seam,
// while the source-level local gate prevents a collective from retaining the witness but bypassing the driver.
struct MixedPipelineDriver {};

// The one shared stage-ring driver for ordinary, folded and two-plane mixed-input providers. Hooks deliberately
// describe lifetime boundaries rather than storage details:
//   bind_read   -- point provider views at a stage before its async completion is published;
//   publish     -- provider work after cp.async wait and before the CTA barrier (for example packed decode);
//   prepare     -- load/convert the next register delivery from the current read stage; its final Int<> argument
//                  distinguishes the one-time register prime from steady-state preparation;
//   prefetch    -- issue provider gmem->smem work for the current K tile into the write stage;
//   consume     -- issue MMA, with the immutable stage token captured before the ring advances.
//
// Keeping the consume-stage token here is correctness-critical. Both deferred B providers once read scale or plane-2
// data from the post-advance stage when K_BLOCK_MAX == 1.
template <int Stages, class KBlockMax, class KTileIterator,
          class BindRead, class Publish, class Prepare, class Prefetch, class Consume>
CUTLASS_DEVICE void run_mixed_pipeline(
    KBlockMax const&,
    KTileIterator k_tile_iter,
    int k_tile_count,
    BindRead&& bind_read,
    Publish&& publish,
    Prepare&& prepare,
    Prefetch&& prefetch,
    Consume&& consume) {
  static_assert(Stages >= 2, "mixed cp.async pipeline needs at least two stages");
  constexpr int KBlocks = KBlockMax::value;
  static_assert(KBlocks >= 1, "mixed pipeline needs at least one B copy block");

  int smem_pipe_read  = 0;
  int smem_pipe_write = Stages - 1;

  if constexpr (KBlocks > 1) {
    bind_read(smem_pipe_read);
    cp_async_wait<Stages - 2>();
    publish(smem_pipe_read);
    __syncthreads();
    prepare(cute::Int<0>{}, smem_pipe_read, cute::Int<1>{});
  }

  CUTLASS_PRAGMA_NO_UNROLL
  for (; k_tile_count > -(Stages - 1); --k_tile_count) {
    cute::for_each(cute::make_int_sequence<KBlocks>{}, [&] (auto k_block) {
      if constexpr (decltype(k_block)::value == KBlocks - 1) {
        bind_read(smem_pipe_read);
        cp_async_wait<Stages - 2>();
        publish(smem_pipe_read);
        __syncthreads();
      }

      int const consume_stage = smem_pipe_read;
      auto k_block_next = (k_block + cute::Int<1>{}) % cute::Int<KBlocks>{};
      prepare(k_block_next, smem_pipe_read, cute::Int<0>{});

      if constexpr (decltype(k_block)::value == 0) {
        prefetch(*k_tile_iter, smem_pipe_write);
        cp_async_fence();
        if (k_tile_count > 1) {
          ++k_tile_iter;
        }
      }

      if constexpr (decltype(k_block)::value == KBlocks - 2 || KBlocks == 1) {
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
      }

      consume(k_block, consume_stage);
    });
  }

  cp_async_wait<0>();
  __syncthreads();
}

}  // namespace cutlass::gemm::collective::detail
