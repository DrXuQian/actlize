/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp" // cute::false_type
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

template <class T, template <int...> class U>
struct is_kernel_tag_of : cute::false_type {};

template <template <int...> class U, int... Args>
struct is_kernel_tag_of<U<Args...>, U> : cute::true_type {};

template <class T, template <int...> class U>
constexpr bool is_kernel_tag_of_v = is_kernel_tag_of<T, U>::value;

template <class T, template <int,bool> class U>
struct is_asymmetric_dma_kernel_tag_of : cute::false_type {};

template <template <int, bool> class U, int I0, bool B0>
struct is_asymmetric_dma_kernel_tag_of<U<I0, B0>, U> : cute::true_type {};

template <class T, template <int, bool> class U>
constexpr bool is_asymmetric_dma_kernel_tag_of_v = \
                              is_asymmetric_dma_kernel_tag_of<T, U>::value;

}

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

//////////////////////////////////////////////////////////////////////////////

namespace detail {

enum class KernelInputTransformType {
    FastF32,
    InterleavedComplexTF32
};

} // namespace detail

//////////////////////////////////////////////////////////////////////////////

namespace kernel::detail {

// Has_SwapAB<T>::value will be true only if:
//   class T has member SwapAB and T::SwapAB is true
template <typename T, typename = void>
struct Has_SwapAB { static constexpr bool value = false; };

template <typename T>
struct Has_SwapAB <T, CUTE_STL_NAMESPACE::void_t<decltype(T::SwapAB)>>
{ static constexpr bool value = T::SwapAB; };

template <typename T>
static constexpr bool Has_SwapAB_v = Has_SwapAB<T>::value;

} // namespace kernel::detail

//////////////////////////////////////////////////////////////////////////////

//
// Kernel schedule policies (the base class tags, one for each kernel layer file)
//
struct KernelMultistage { };
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { 
};
struct KernelTmaWarpSpecializedCooperative { 
};

struct KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpong { };

//////////////////////////////////////////////////////////////////////////////

//
// Builder dispatch policies (not a part of the main CUTLASS layers, simply used to opt into
// specific collective builder dispatches)
//

// FP8 related policies (including Fast Accumulation)
struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum : KernelPtrArrayTmaWarpSpecializedPingpong { };

// Policies to opt into mixed type GEMMs
// FIXME: These have been deleted in v3.6.0(change-79735). But they are used in 705_ppu_mixed_dtype_gemm.
struct KernelTmaWarpSpecializedMixedInput : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongMixedInput : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeMixedInput: KernelTmaWarpSpecializedCooperative { };

//////////////////////////////////////////////////////////////////////////////

// PPU specialized Kernel schedule policies
struct KernelAiuMultistage { };
struct KernelAiuMultistageMixedInput { };
struct KernelAiuMultistageMixedInputPerCol { };
struct KernelAiuMultistageMixedInputFinegrainedGs128 { };
struct KernelAiuMultistageMixedInputFinegrainedGs64 { };
struct KernelAiuMultistageMixedInputFinegrainedGs32 { };  // gs=32 (Q4_0/Q4_1/Q4_K-as-AWQ)

// Artifact-fold schedule wrapper. The folds describe the resident B planes, not the consumer TileShape.K: a tactic
// with a larger TileK may read the same bytes, but it must keep both physical (N/F, F*K) descriptors. The low fold
// selects the ordinary-vs-folded collective; the independent high fold sizes the second plane. Keep BaseSchedule_ in
// the middle so existing KernelAiuFold<F, Base> spellings remain source compatible.
template<int ArtifactLowFold_, class BaseSchedule_ = KernelAiuMultistageMixedInputFinegrainedGs32,
         int ArtifactHighFold_ = 0>
struct KernelAiuFold {
  static constexpr int FoldF = ArtifactLowFold_;  // compatibility for downstream users of the old name
  static constexpr int ArtifactLowFold = ArtifactLowFold_;
  static constexpr int ArtifactHighFold = ArtifactHighFold_;
  using BaseSchedule = BaseSchedule_;
};
// A zero fold means that no artifact contract was supplied. The builder retains its legacy derivation only for such
// direct CollectiveBuilder users; the shared quactlize policy always supplies both folds when either plane is folded.
template<class T> struct fold_schedule_traits {
  static constexpr int FoldF = 0;
  static constexpr int ArtifactLowFold = 0;
  static constexpr int ArtifactHighFold = 0;
  using Base = T;
};
template<int LowFold, class B, int HighFold>
struct fold_schedule_traits<KernelAiuFold<LowFold, B, HighFold>> {
  static constexpr int FoldF = LowFold;
  static constexpr int ArtifactLowFold = LowFold;
  static constexpr int ArtifactHighFold = HighFold;
  using Base = B;
};
struct KernelAiuMultistageBatchArray { };
struct KernelAiuMultistageBatchArrayOverlapPrologue { };
struct KernelAiuMultistageStreamK { };
struct KernelAiuMultistagePersistent {};
struct KernelAiuMultistagePersistentOverlapPrologue {};
struct KernelMultistagePersistent {};
struct KernelMultistageBatchArray {};

//////////////////////////////////////////////////////////////////////////////

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

// 2 stage pipeline through 1 stage in smem, 1 in rmem, WITHOUT predicated gmem loads
struct MainloopPPUTwoStageUnpredicated {
  constexpr static int Stages = 2;
  // PPU 1.0/1.5 both support
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// 2 stage pipeline through 1 stage in smem, 1 in rmem, with predicated gmem loads
struct MainloopPPUTwoStage {
  constexpr static int Stages = 2;
  // PPU 1.0/1.5 both support
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, WITHOUT predicated gmem loads
template<int Stages_>
struct MainloopPPUCpAsyncUnpredicated {
  constexpr static int Stages = Stages_;
  // PPU 1.0/1.5 both support
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads
// Parametrized by ClusterShape (Schedule is hardcoded to KernelMultistage).
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopPPUCpAsyncLegacy {
  constexpr static int Stages = Stages_;
  // PPU 1.0/1.5 both support
  using Schedule = KernelMultistage;
  using ClusterShape = ClusterShape_;
};

//////////////////////////////////////////////////////////////////////////////

//
// PPU specialized Collective Mainloop Policies
//

// n-buffer in smem (cp.async), pipelined with registers, WITHOUT predicated gmem loads
template<int Stages_, typename Schedule_ = KernelAiuMultistage>
struct MainloopPPUAiu {
  constexpr static int Stages = Stages_;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, typename Schedule_ = KernelAiuMultistagePersistentOverlapPrologue>
struct MainloopPPUAiuPersistentOverlapPrologue {
  constexpr static int Stages = Stages_;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, typename Schedule_ = KernelAiuMultistage>
struct MainloopPPUAiuFP8 {
  constexpr static int Stages = Stages_;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_>
struct MainloopPPUAiuBatchArray {
  constexpr static int Stages = Stages_;
  using Schedule = KernelAiuMultistageBatchArray;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, typename Schedule_ = KernelAiuMultistageBatchArrayOverlapPrologue>
struct MainloopPPUAiuBatchArrayPersistOverlapPrologue {
  constexpr static int Stages = Stages_;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_, typename Schedule_ = KernelAiuMultistageMixedInput>
struct MainloopPPUAiuMixedInput {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 0;  // default value
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputPerCol> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = -1;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs128> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 128;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs64> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 64;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs32> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 32;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

// ---- B BIT-PLANE CONCAT (Q3 = int2+int1, Q5 = int4+int1, Q6 = int4+int2) ------------------------------------
// A DISTINCT mainloop policy, so the two-B-plane collective (ppu_mma_aiu_mixed_input_2plane.hpp) is its OWN
// CollectiveMma specialization instead of competing with / being if-constexpr'd into the validated single-plane
// one. Mirrors MainloopPPUAiuMixedInput exactly -- including every per-Schedule StaticGroupSize specialization --
// because the 2-plane mainloop reuses the same scale/zero machinery (and the GGUF concats are gs=16, i.e. the
// FINE per-mma-atom scale path).
template<int Stages_, class kContinous_, typename Schedule_ = KernelAiuMultistageMixedInput>
struct MainloopPPUAiuMixedInput2Plane {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 0;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputPerCol> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = -1;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs128> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 128;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs64> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 64;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, class kContinous_>
struct MainloopPPUAiuMixedInput2Plane<Stages_, kContinous_, KernelAiuMultistageMixedInputFinegrainedGs32> {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 32;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

// ---- N-FOLD (TK-freeing) mainloop policy ---------------------------------------------------------------------
// A DISTINCT policy so the fold collective (ppu_mma_aiu_fold.hpp) is its OWN CollectiveMma specialization, zero
// regression to the validated single-plane / 2-plane ones. FoldFactor F = 32B-elems / TileShape.K: the B plane's
// AIU contiguous run folds F adjacent N-columns to reach 32B while TileShape.K (A / MMA) stays small (A-smem =
// TileM*TK*2 shrinks by F). B smem K-extent = F * TileShape.K; mainloop runs F gemm passes (B's K-atom blocks,
// reusing one A) into an F*N-wide accumulator. Offline data prepared by nfold_column_pairs_ppu (P1.1). Mirrors the
// per-Schedule StaticGroupSize set (GGUF concats are gs=16 = FINE per-atom scale).
template<int Stages_, class kContinous_, int FoldF_ = 2, typename Schedule_ = KernelAiuMultistageMixedInput>
struct MainloopPPUAiuFold {
  constexpr static int Stages = Stages_;
  constexpr static int StaticGroupSize = 0;
  constexpr static int FoldFactor = FoldF_;
  using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputPerCol> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = -1;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs128> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 128;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs64> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 64;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};
template<int Stages_, class kContinous_, int FoldF_>
struct MainloopPPUAiuFold<Stages_, kContinous_, FoldF_, KernelAiuMultistageMixedInputFinegrainedGs32> {
  constexpr static int Stages = Stages_; constexpr static int StaticGroupSize = 32;
  constexpr static int FoldFactor = FoldF_; using kContinous = kContinous_;
  using Schedule = KernelAiuMultistageMixedInput; using ClusterShape = Shape<_1,_1,_1>;
};


struct KernelMultistageWithScale { };
struct KernelAiuMultistageWithScale : public KernelAiuMultistage { };
struct KernelAiuMultistageWithBlockWiseScale : public KernelAiuMultistageWithScale {};

template<int Stages_, typename Schedule_ = KernelAiuMultistageWithScale>
struct MainloopWithScalePPUAiu {
  constexpr static int Stages = Stages_;
  constexpr static bool IsBlockWiseScale = false;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_>
struct MainloopWithScalePPUAiu<Stages_, KernelAiuMultistageWithBlockWiseScale> {
  constexpr static int Stages = Stages_;
  constexpr static bool IsBlockWiseScale = true;
  using Schedule = KernelAiuMultistageMixedInput;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_>
struct MainloopWithScalePPU0015Aiu {
  constexpr static int Stages = Stages_;
  using Schedule = KernelAiuMultistageWithScale;
  using ClusterShape = Shape<_1,_1,_1>;
};

struct MainloopPPUTwoStageUnpredicatedLdmatrix {
  constexpr static int Stages = 2;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

struct MainloopPPUTwoStageLdmatrix {
  constexpr static int Stages = 2;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

struct MainloopPPUTwoStageLdmatrixBatchArray {
  constexpr static int Stages = 2;
  using Schedule = KernelMultistageBatchArray;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_>
struct MainloopPPUCpAsyncWithScale {
  constexpr static int Stages = Stages_;
  using Schedule = KernelMultistageWithScale;
  using ClusterShape = Shape<_1,_1,_1>;
};

// main difference with MainloopPPUCpAsync is MainloopPPUCpAsyncLegacy can configure "Schedule" type, instead of hard coded to KernelMultistage.
template<
  int Stages_,
  typename Schedule_ = KernelMultistage
>
struct MainloopPPUCpAsync {
  constexpr static int Stages = Stages_;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads
template<
  int Stages_,
  typename Schedule_ = KernelMultistage
>
struct MainloopPPUCpAsyncBatchArray {
  constexpr static int Stages = Stages_;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm
