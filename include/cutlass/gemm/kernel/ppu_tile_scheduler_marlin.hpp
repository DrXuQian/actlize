/***************************************************************************************************
 * Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/workspace.h"
#include "cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"

namespace cutlass::gemm::kernel::detail {

// Marlin's dispatcher is not Stream-K with a different partition heuristic.
// It owns one equal-length stripe per launched CTA in the flattened (q, k)
// space, with K as the fast dimension.  The launch count is part of that
// decomposition.  The default launch policy retains classic Marlin's one CTA
// per CU safety rail.  A host caller may explicitly sweep a larger
// blocks-per-CU cohort; production must validate that cohort against the
// instantiated kernel's occupancy before requesting it.
template <class TileShape_, class ClusterShape_, uint32_t FixupThreadCount_ = 0>
class PersistentTileSchedulerPPUMarlin {
public:
  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;
  static constexpr uint32_t FixupThreadCount = FixupThreadCount_;
  using BarrierType = typename NamedBarrierManager<1>::T;
  using StripeCore = MarlinStripeSchedulerCore;
  using StripeParams = typename StripeCore::Params;
  static constexpr uint64_t OutputTileElements =
      uint64_t(cute::size<0>(TileShape{})) *
      uint64_t(cute::size<1>(TileShape{}));

  // This is a structural capability, not a list of benchmark cohorts and not
  // a claim that every admitted kernel has positive occupancy on every PPU.
  // The PPU builder launches whole 32-thread warps, while the shared tactic
  // authority caps one CTA at 32 warps.  Exact accumulator coverage and the
  // named-barrier arrival count are asserted independently below.
  CUTLASS_HOST_DEVICE static constexpr bool fixup_thread_count_capable(
      uint32_t thread_count) {
    return thread_count >= uint32_t(cutlass::NumThreadsPerWarp) &&
           thread_count <= 32u * uint32_t(cutlass::NumThreadsPerWarp) &&
           thread_count % uint32_t(cutlass::NumThreadsPerWarp) == 0;
  }

  static_assert(cute::is_static<TileShape>::value);
  static_assert(cute::is_static<ClusterShape>::value);
  static_assert(cute::size(ClusterShape{}) == 1,
                "the first Marlin scheduler wiring does not support clusters");
  // Zero is the public-selector spelling: fixup derives the exact cohort from
  // tile-elements / per-thread-fragment-elements.  Named kernels may pass an
  // explicit value and assert it against their launch shape.  This prevents a
  // generic 64-thread caller from silently inheriting a default 128-thread
  // barrier cohort merely because TileSchedulerSelector cannot see TiledMma.
  static_assert(FixupThreadCount == 0 ||
                fixup_thread_count_capable(FixupThreadCount),
                "Marlin cooperative requires a warp-aligned 32..1024-thread CTA cohort");

  struct Arguments {
    uint32_t blocks_per_cu = 1;
  };

  struct Params : StripeParams {
    void* workspace_ = nullptr;
  };
  static_assert(std::is_trivially_copyable_v<Params>,
                "Marlin scheduler Params must cross the host/device ABI unchanged");
  using WorkTileInfo = typename StripeCore::WorkTileInfo;

private:
  Params scheduler_params_{};

  template <class ElementAccumulator>
  CUTLASS_HOST_DEVICE static uint64_t reduction_workspace_bytes(Params const& p) {
    uint64_t all_elements = 0, bytes = 0;
    bool ok = StripeCore::mul_u64(OutputTileElements, p.output_tiles_, all_elements) &&
              StripeCore::mul_u64(all_elements, sizeof(ElementAccumulator), bytes);
    return ok ? bytes : 0;
  }

  CUTLASS_HOST_DEVICE static uint64_t lock_workspace_bytes(Params const& p) {
    uint64_t bytes = 0;
    return StripeCore::mul_u64(p.output_tiles_, sizeof(BarrierType), bytes) ? bytes : 0;
  }

  template <class ElementAccumulator>
  CUTLASS_HOST_DEVICE static ElementAccumulator* reduction_workspace(Params const& p) {
    return reinterpret_cast<ElementAccumulator*>(p.workspace_);
  }

  template <class ElementAccumulator>
  CUTLASS_HOST_DEVICE static BarrierType* lock_workspace(Params const& p) {
    return reinterpret_cast<BarrierType*>(
        reinterpret_cast<uint8_t*>(p.workspace_) + reduction_workspace_bytes<ElementAccumulator>(p));
  }

public:
  CUTLASS_HOST_DEVICE constexpr PersistentTileSchedulerPPUMarlin() = default;
  CUTLASS_HOST_DEVICE constexpr explicit PersistentTileSchedulerPPUMarlin(Params const& p)
      : scheduler_params_(p) {}

  // One shared host/device constructor is the oracle seam.  Production
  // to_underlying_arguments and l126 both call this exact function.
  CUTLASS_HOST_DEVICE static constexpr Params make_params_for_tiles(
      uint64_t tiles_m, uint64_t tiles_n, uint64_t tiles_l,
      uint64_t k_tiles, uint64_t cu_count, void* workspace = nullptr,
      uint32_t blocks_per_cu = 1) {
    Params p;
    static_cast<StripeParams&>(p) = StripeCore::make_params_for_tiles(
        tiles_m, tiles_n, tiles_l, k_tiles, cu_count,
        uint64_t(blocks_per_cu));
    p.workspace_ = workspace;
    return p;
  }

  // Raw-problem seam shared by production lowering and the generated-code
  // oracle.  Keeping M/N/K -> tile ordinals here prevents a host proof from
  // silently validating hand-computed tile counts while the shipping path
  // lowers a different problem (the class of bug caught by expert-pitch).
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static constexpr Params make_params_for_problem_shape(
      ProblemShape problem_shape, uint64_t cu_count,
      void* workspace = nullptr, uint32_t blocks_per_cu = 1) {
    auto shape = cute::append<4>(problem_shape, cute::Int<1>{});
    uint64_t const tm = uint64_t(cute::size<0>(TileShape{}));
    uint64_t const tn = uint64_t(cute::size<1>(TileShape{}));
    uint64_t const tk = uint64_t(cute::size<2>(TileShape{}));
    uint64_t const m = uint64_t(cute::get<0>(shape));
    uint64_t const n = uint64_t(cute::get<1>(shape));
    uint64_t const k = uint64_t(cute::get<2>(shape));
    uint64_t const l = uint64_t(cute::get<3>(shape));
    return make_params_for_tiles(
        StripeCore::ceil_div_u64(m, tm), StripeCore::ceil_div_u64(n, tn), l,
        StripeCore::ceil_div_u64(k, tk), cu_count, workspace, blocks_per_cu);
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static Params to_underlying_arguments(
      ProblemShape problem_shape, TileShape, ClusterShape,
      KernelHardwareInfo const& hw_info, Arguments const& args, void* workspace,
      uint32_t = 1, uint32_t = 1) {
    return make_params_for_problem_shape(
        problem_shape, uint64_t(hw_info.cu_count), workspace,
        args.blocks_per_cu);
  }

  static bool can_implement(Arguments const& args) {
    return args.blocks_per_cu > 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t get_workspace_size(
      Arguments const& args, ProblemShape problem_shape,
      KernelHardwareInfo const& hw_info, uint32_t,
      uint32_t = 1, uint32_t = 1) {
    Params p = to_underlying_arguments(
        problem_shape, TileShape{}, ClusterShape{}, hw_info, args, nullptr);
    if (!p.valid_ || p.iters_per_block_ == p.k_tiles_per_output_) {
      return 0;
    }
    uint64_t const reduction = reduction_workspace_bytes<ElementAccumulator>(p);
    uint64_t const locks = lock_workspace_bytes(p);
    if (reduction == 0 || locks == 0 ||
        reduction > std::numeric_limits<size_t>::max() - locks) {
      return 0;
    }
    return size_t(reduction + locks);
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status initialize_workspace(
      Arguments const& args, void* workspace, hggcStream_t stream,
      ProblemShape const& problem_shape, KernelHardwareInfo const& hw_info,
      uint32_t mma_warp_groups, uint32_t epilogue_subtile = 1,
      uint32_t num_accumulator_mtxs = 1, HostAdapter* host_adapter = nullptr) {
    (void)mma_warp_groups;
    (void)epilogue_subtile;
    (void)num_accumulator_mtxs;
    Params p = to_underlying_arguments(
        problem_shape, TileShape{}, ClusterShape{}, hw_info, args, workspace);
    if (!p.valid_) {
      return Status::kErrorInvalidProblem;
    }
    if (p.iters_per_block_ == p.k_tiles_per_output_) {
      return Status::kSuccess;
    }
    uint64_t const reduction = reduction_workspace_bytes<ElementAccumulator>(p);
    uint64_t const locks = lock_workspace_bytes(p);
    if (reduction == 0 || locks == 0 ||
        reduction > std::numeric_limits<uint64_t>::max() - locks ||
        reduction + locks > std::numeric_limits<size_t>::max()) {
      return Status::kErrorInvalidProblem;
    }
    if (workspace == nullptr) {
      return Status::kErrorWorkspaceNull;
    }
    return zero_workspace(reinterpret_cast<uint8_t*>(workspace) + reduction,
                          size_t(locks), stream, host_adapter);
  }

  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(Params const& p) {
    return p.valid_ && p.grid_blocks_ <= std::numeric_limits<unsigned>::max()
        ? dim3(unsigned(p.grid_blocks_), 1, 1) : dim3(0, 0, 0);
  }

  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo get_work_for_block(
      Params const& p, uint64_t block_idx) {
    return StripeCore::get_work_for_block(
        static_cast<StripeParams const&>(p), block_idx);
  }

  CUTLASS_HOST_DEVICE constexpr WorkTileInfo get_work_for_block_index(
      uint64_t block_idx) const {
    return get_work_for_block(scheduler_params_, block_idx);
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return get_work_for_block_index(uint64_t(blockIdx.x));
  }

  CUTLASS_HOST_DEVICE static constexpr WorkTileInfo fetch_next_work_for_params(
      Params const& p, WorkTileInfo const& work) {
    return StripeCore::fetch_next_work(static_cast<StripeParams const&>(p), work);
  }

  CUTLASS_HOST_DEVICE constexpr WorkTileInfo get_next_work(
      WorkTileInfo const& work) const {
    return fetch_next_work_for_params(scheduler_params_, work);
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo const& work) const {
    return cute::make_tuple(get_next_work(work), true);
  }

  CUTLASS_HOST_DEVICE static constexpr bool valid_warpgroup_in_work_tile(WorkTileInfo const& work) {
    return work.is_valid();
  }
  CUTLASS_HOST_DEVICE static constexpr uint32_t get_work_k_tile_start(WorkTileInfo const& work) {
    return work.K_idx;
  }
  template <class KTileShape>
  CUTLASS_HOST_DEVICE static constexpr auto get_work_k_tile_coord(
      WorkTileInfo const& work, KTileShape const& shape) {
    // K_idx is a K-tile ordinal.  idx2crd consumes that ordinal directly;
    // multiplying by tactic TileK here would repeat the expert-pitch class of
    // logical-code/byte unit bugs at the scheduler/mainloop seam.
    return cute::idx2crd(get_work_k_tile_start(work), shape);
  }
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static constexpr uint32_t get_work_k_tile_count(
      WorkTileInfo const& work, ProblemShape, TileShape) {
    return work.k_tile_count;
  }
  CUTLASS_HOST_DEVICE static constexpr bool requires_fixup(Params const&, WorkTileInfo const& work) {
    return work.is_valid() && work.slice_count > 1;
  }
  CUTLASS_HOST_DEVICE static constexpr bool compute_epilogue(WorkTileInfo const& work, Params const&) {
    return work.is_valid() && work.slice_idx + 1 == work.slice_count;
  }
  CUTLASS_HOST_DEVICE static constexpr uint64_t output_tile_index(Params const&, WorkTileInfo const& work) {
    return work.output_tile_idx;
  }
  CUTLASS_HOST_DEVICE static constexpr uint64_t reduction_workspace_element_offset(
      WorkTileInfo const& work) {
    // Typed FP32 pointer arithmetic consumes elements, never bytes.
    return work.output_tile_idx * OutputTileElements;
  }
  CUTLASS_HOST_DEVICE static constexpr int barrier_lock_index(
      WorkTileInfo const& work) {
    return int(work.lock_idx);
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE static void fixup(
      Params const& p, WorkTileInfo const& work, FrgTensorC& accumulators,
      uint32_t num_barriers, uint32_t barrier_idx) {
    struct Always {
      CUTLASS_HOST_DEVICE bool operator()(int) const { return true; }
    };
    fixup_predicated(p, work, accumulators, num_barriers, barrier_idx, Always{});
  }

  template <class FrgTensorC, class Predicate>
  CUTLASS_DEVICE static void fixup(
      Params const& p, WorkTileInfo const& work, FrgTensorC& accumulators,
      uint32_t num_barriers, uint32_t barrier_idx, Predicate const& predicate) {
    fixup_predicated(p, work, accumulators, num_barriers, barrier_idx, predicate);
  }

private:
  template <class FrgTensorC, class Predicate>
  CUTLASS_DEVICE static void fixup_predicated(
      Params const& p, WorkTileInfo const& work, FrgTensorC& accumulators,
      uint32_t num_barriers, uint32_t barrier_idx, Predicate const& predicate) {
    if (!requires_fixup(p, work)) {
      return;
    }
    CUTLASS_ASSERT(num_barriers == 1 && barrier_idx == 0);
    static constexpr uint64_t TileElements = OutputTileElements;
    static constexpr uint64_t FragmentElements = uint64_t(cute::size(FrgTensorC{}));
    static_assert(FragmentElements > 0 && TileElements % FragmentElements == 0,
                  "Marlin cooperative fragment must divide one output tile exactly");
    static constexpr uint32_t DerivedThreadCount =
        uint32_t(TileElements / FragmentElements);
    static constexpr uint32_t Cohort =
        FixupThreadCount == 0 ? DerivedThreadCount : FixupThreadCount;
    static_assert(fixup_thread_count_capable(Cohort),
                  "Marlin cooperative derived a non-capable CTA cohort");
    static_assert(FixupThreadCount == 0 || FixupThreadCount == DerivedThreadCount,
                  "explicit Marlin cooperative cohort disagrees with the accumulator layout");
    static_assert(Cohort == DerivedThreadCount,
                  "Marlin cooperative cohort must equal the exact accumulator-derived CTA size");
    using BarrierManager = NamedBarrierManager<
        Cohort,
        static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0), 1>;
    using AccumulatorArray = Array<typename FrgTensorC::value_type, cute::size(FrgTensorC{})>;
    using Striped = BlockStripedReduce<Cohort, AccumulatorArray>;
    static_assert(
        uint64_t(Cohort) * FragmentElements == TileElements,
        "Marlin cooperative cohort must cover one exact FP32 output tile");
    static_assert(BarrierManager::ThreadCount == Cohort,
                  "Marlin named-barrier arrival count must equal the exact CTA cohort");
    static_assert(Striped::kStripes == cute::size(FrgTensorC{}),
                  "Marlin residue predicate must name each scalar accumulator");

    // The kernel launches exactly Cohort threads.  A modulo here would hide a
    // future launch/cooperative mismatch by aliasing surplus threads onto the
    // same FP32 workspace stripes.
    uint32_t const thread = uint32_t(threadIdx.x);
    auto* tile_workspace = reduction_workspace<typename FrgTensorC::value_type>(p) +
                           reduction_workspace_element_offset(work);
    auto* workspace_array = reinterpret_cast<AccumulatorArray*>(tile_workspace);
    auto* accumulator_array = reinterpret_cast<AccumulatorArray*>(accumulators.data());
    BarrierType* locks = lock_workspace<typename FrgTensorC::value_type>(p);
    int const lock = barrier_lock_index(work);

    if (work.slice_idx == 0) {
      Striped::store(workspace_array, *accumulator_array, thread, predicate);
      BarrierManager::arrive_inc(0, locks, thread, lock, 1);
    }
    else if (!compute_epilogue(work, p)) {
      BarrierManager::wait_eq(0, locks, thread, lock, work.slice_idx);
      Striped::load_add(*accumulator_array, workspace_array, thread, predicate);
      Striped::store(workspace_array, *accumulator_array, thread, predicate);
      BarrierManager::arrive_inc(0, locks, thread, lock, 1);
    }
    else {
      // The final peer resets q's lock after acquiring it.  The next launch on
      // the same stream therefore does not depend on an external barrier-tail
      // reset; its first peer cannot start until this kernel has completed.
      BarrierManager::wait_eq_reset(0, locks, thread, lock, work.slice_idx);
      Striped::load_add(*accumulator_array, workspace_array, thread, predicate);
    }
  }
};

} // namespace cutlass::gemm::kernel::detail
