/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

// Device harnesses use this as a stale-submodule gate. Version 3 is the materialized CuTe fragment construction;
// keep the marker beside the shared implementation so it cannot describe one collective while another drifts.
#define PPU_SCALE_FRAGMENT_API 3

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/numeric/numeric_types.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"

namespace cutlass::gemm::collective::detail {

// CuTe does not diagnose a tiled-copy thread layout larger than the CTA. The thread slice wraps modulo the copy's
// slots and silently leaves metadata behind, so every mixed-input collective publishes this same witness.
template <int CopyThreadSlots, int CtaThreads>
struct ScaleCopyCoverage {
  static constexpr bool value = CopyThreadSlots <= CtaThreads;
  static_assert(value,
                "scale copy asks for more thread slots than the CTA has -- the modulo slice would silently truncate it");
};

// One source of truth for the two values formerly spelled as `% Per` and `/ Per` at every COARSE/FINE call site.
template <int Per, int NGroups>
struct ScaleSplit {
  static_assert(Per > 0 && NGroups > 0, "metadata groups must have positive extents");
  using InGroupL = cute::Layout<cute::Shape <cute::Int<Per>, cute::Int<NGroups>>,
                                cute::Stride<cute::_1,       cute::_0>>;
  using GroupL   = cute::Layout<cute::Shape <cute::Int<Per>, cute::Int<NGroups>>,
                                cute::Stride<cute::_0,       cute::Int<1>>>;

  CUTE_HOST_DEVICE static constexpr int in_group(int i) { return int(InGroupL{}(i)); }
  CUTE_HOST_DEVICE static constexpr int group(int i)    { return int(GroupL{}(i)); }
};

// COARSE means one metadata group covers one or more complete B-copy steps. The policy owns both the predicate and
// the divisibility invariant so a collective cannot select COARSE with a different boundary/index calculation.
template <int ScaleGroups, int CopySteps>
struct CoarseScalePolicy {
  static_assert(ScaleGroups > 0 && CopySteps > 0, "metadata and B-copy extents must be positive");
  static constexpr bool active = ScaleGroups <= CopySteps;
  static_assert(!active || CopySteps % ScaleGroups == 0,
                "COARSE scale groups must evenly partition the B copy steps");
  static constexpr int steps_per_group = active ? CopySteps / ScaleGroups : 1;
  using Split = ScaleSplit<steps_per_group, ScaleGroups>;

  CUTE_HOST_DEVICE static constexpr bool starts_group(int copy_step) {
    return Split::in_group(copy_step) == 0;
  }
  CUTE_HOST_DEVICE static constexpr int group(int copy_step) {
    return Split::group(copy_step);
  }
};

// FINE means a B-copy step crosses metadata groups, so metadata is reloaded at MMA-atom boundaries. This is the
// exact complement of COARSE for a particular copy view; MmaAtoms supplies the group-to-atom mapping.
template <int ScaleGroups, int CopySteps, int MmaAtoms>
struct FineScalePolicy {
  using Coarse = CoarseScalePolicy<ScaleGroups, CopySteps>;
  static constexpr bool active = !Coarse::active;
  static_assert(!active || (MmaAtoms >= ScaleGroups && MmaAtoms % ScaleGroups == 0),
                "FINE scale groups must evenly partition the MMA K atoms");
  static constexpr int atoms_per_group = active ? MmaAtoms / ScaleGroups : 1;
  using Split = ScaleSplit<atoms_per_group, ScaleGroups>;

  CUTE_HOST_DEVICE static constexpr bool starts_group(int atom) {
    return Split::in_group(atom) == 0;
  }
  CUTE_HOST_DEVICE static constexpr int group(int atom) {
    return Split::group(atom);
  }
};

// Fragment construction is intentionally the plain CuTe idiom. Callers pass an already rank-2 scale tile; keeping
// the slice outside this helper lets flat and (group,stage) shared-memory views use the same implementation.
template <class ElementScale>
struct ScaleFragment {
  template <class TiledMma, class STensor>
  CUTLASS_DEVICE static auto make(TiledMma const& thr_mma, STensor const& scale_tile) {
    return cute::make_fragment_like<ElementScale>(thr_mma.partition_fragment_B(scale_tile));
  }

  template <class TiledMma, class STensor>
  CUTE_HOST_DEVICE static constexpr auto layout(TiledMma const& thr_mma, STensor const& scale_tile) {
    return cute::make_layout_like(thr_mma.partition_fragment_B(scale_tile).layout());
  }
};

// The ordinary and folded collectives flatten (group,stage); the two-plane collective keeps both coordinates. These
// are storage policies only. Group boundaries and the published register-fragment tuple contract remain common.
struct FlatMetadataAddress {
  template <int ScaleGroups, class Tensor>
  CUTLASS_DEVICE static auto source(Tensor const& tensor, int group, int stage) {
    return tensor(cute::_, cute::_, 0, stage * ScaleGroups + group);
  }
};

struct SplitMetadataAddress {
  template <int ScaleGroups, class Tensor>
  CUTLASS_DEVICE static auto source(Tensor const& tensor, int group, int stage) {
    return tensor(cute::_, cute::_, 0, group, stage);
  }
};

// All three collectives publish (scale source, scale fragment [, zero source, zero fragment]) and retile that as
// (copy atom, scale destination [, zero destination]). Centralizing those positional contracts makes reload one rule.
template <class Address, int ScaleGroups, bool HasZero, class Info, class Views>
CUTLASS_DEVICE void reload_metadata(Info const& info, Views const& views, int group, int stage) {
  auto smem_tiled_copy = cute::get<0>(views);
  auto scale_src       = cute::get<0>(info);
  auto scale_dst       = cute::get<1>(views);
  cute::copy(smem_tiled_copy,
             Address::template source<ScaleGroups>(scale_src, group, stage),
             scale_dst(cute::_, cute::_, 0));
  if constexpr (HasZero) {
    auto zero_src = cute::get<2>(info);
    auto zero_dst = cute::get<2>(views);
    cute::copy(smem_tiled_copy,
               Address::template source<ScaleGroups>(zero_src, group, stage),
               zero_dst(cute::_, cute::_, 0));
  }
}

template <bool HasZero, class BSlice, class Info>
CUTLASS_DEVICE void apply_metadata(BSlice&& b_slice, Info const& info) {
  cute::transform(b_slice, cute::get<1>(info)(cute::_, cute::_, 0), b_slice, cute::multiplies{});
  if constexpr (HasZero) {
    cute::transform(b_slice, cute::get<3>(info)(cute::_, cute::_, 0), b_slice, cute::plus{});
  }
}

// Public policy seam owned by every mixed-input CollectiveMma. The collective still owns shared-memory shape and
// packed-provider details, but it cannot independently redefine fragment construction, group boundaries, reload
// addressing or ordinary scale/zero application.
template <class ElementScale, int ScaleGroups, bool HasZero, class Address>
struct MixedMetadataPolicy {
  static_assert(ScaleGroups > 0, "a mixed metadata policy needs at least one K group");
  static constexpr int scale_groups = ScaleGroups;
  static constexpr bool has_zero = HasZero;
  using AddressPolicy = Address;

  template <int CopyThreadSlots, int CtaThreads>
  using Coverage = ScaleCopyCoverage<CopyThreadSlots, CtaThreads>;

  template <int CopySteps>
  using Coarse = CoarseScalePolicy<ScaleGroups, CopySteps>;

  template <int CopySteps, int MmaAtoms>
  using Fine = FineScalePolicy<ScaleGroups, CopySteps, MmaAtoms>;

  template <class TiledMma, class STensor>
  CUTLASS_DEVICE static auto make_fragment(TiledMma const& thr_mma, STensor const& scale_tile) {
    return ScaleFragment<ElementScale>::make(thr_mma, scale_tile);
  }

  template <class TiledMma, class STensor>
  CUTE_HOST_DEVICE static constexpr auto fragment_layout(TiledMma const& thr_mma, STensor const& scale_tile) {
    return ScaleFragment<ElementScale>::layout(thr_mma, scale_tile);
  }

  template <class Info, class Views>
  CUTLASS_DEVICE static void reload(Info const& info, Views const& views, int group, int stage) {
    reload_metadata<Address, ScaleGroups, HasZero>(info, views, group, stage);
  }

  template <class BSlice, class Info>
  CUTLASS_DEVICE static void apply(BSlice&& b_slice, Info const& info) {
    apply_metadata<HasZero>(static_cast<BSlice&&>(b_slice), info);
  }
};

} // namespace cutlass::gemm::collective::detail
