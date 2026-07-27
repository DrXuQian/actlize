// =============================================================================================================
// N-FOLD collective (P1.3). Copy of ppu_mma_aiu_multistage_mixed_input.hpp, specialized on MainloopPPUAiuFold.
// PURPOSE: free a sparse B plane's TileShape.K from the AIU 32B-contiguous-K floor, so A-smem (=TileM*TK*2) shrinks
// and occupancy rises to int4's level (int1: 12%->~50%). The offline data is folded by nfold_column_pairs_ppu (P1.1);
// this collective consumes it.
// KEY REALIZATION (scratchpad/cute_nfold3.cu, verified): the fold is a FOLD-IN-N smem layout, NOT fold-in-K. Present
// the physical folded run (FoldF cols x TK each) as a LOGICAL (FoldF*Ng output-N, TK) tile via strides -> partition_B
// puts the FoldF factor in MMA_N and MMA_K = TK/16. So the MMA accumulates over the REAL TK and emits FoldF*Ng output
// columns NATURALLY. This DISSOLVES the 2-pass mainloop and the interleaved epilogue I first thought were needed --
// mainloop, C accumulator, and epilogue are all STANDARD.
// DONE (structural, isolated -- zero regression; fold header not #included anywhere yet):
//   * matches MainloopPPUAiuFold<Stages,kContinous,FoldF,Schedule>;
//   * SmemLayoutB = FOLD-IN-N layout ((FoldF,Ng),TK):((TK,FoldF*TK),1)  <-- the defining change;
//   * lockstep-K assert UNCHANGED (fold-in-N keeps B's K-mode == TK == A's).
// TODO (box-iterated; PPU device asm, cannot compile locally):
//   * finalize the general FoldF/Ng SmemLayoutB form + verify the operand's swzl SmemCopyAtom (AiuContElemSize =
//     FoldF*TK, REUSED from validated int2@TK128/int1@TK256) delivers into this fold-in-N layout;
//   * gB / gmem-partition N-consistency (size<0>(gB)==size<0>(sB)=FoldF*Ng);
//   * BUILDER: fold path selecting MainloopPPUAiuFold + B operand Block_K = FoldF*blockK.
//   (mainloop + epilogue need NO fold-specific change -- standard, thanks to fold-in-N.)
// NOT gated on int4 ceiling -- int2/int1-fold reaching int4 geometry is guaranteed; biggest win is int1 (4x occ).
// =============================================================================================================
/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// VERSION GATE for the scale-fragment API. A harness static_asserts on this, so a STALE actlize submodule on the box
// fails to COMPILE instead of silently producing a binary identical to the previous one -- which is exactly the
// ambiguity that made an "every acu counter is identical" A/B uninterpretable. Bump it whenever the scale-fragment
// construction changes in a way a measurement is supposed to see.
//   1 = materialised fragment (make_fragment_like of partition_fragment_B)
//   2 = stride-0 broadcast -- REVERTED: a no-op in generated code, and it traded a cute idiom for a hand-built stride
//   3 = back to the cute idiom, as ONE shared entry point across the three collectives, plus scale_frag_cosize()
#define PPU_SCALE_FRAGMENT_API 3


#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Arch,
  int Stages,
  class kContinous,
  int FoldF,
  class KernelSchedule,
  class TileShapePair_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    Arch,
    MainloopPPUAiuFold<Stages, kContinous, FoldF, KernelSchedule>,
    TileShapePair_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
private:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;
  using TileShape_Scale = detail::deduce_mixed_width_dtype_t<1, TileShapePair_>;
public:
  //
  // Type Aliases
  //
  using DispatchPolicy =  MainloopPPUAiuFold<Stages, kContinous, FoldF, KernelSchedule>;
  // N-FOLD: the B plane loads FoldF adjacent N-columns folded into each 32B AIU contiguous run, so its smem K-extent
  // is FoldF x the tile's K, while A / MMA use the small TileShape.K (=> A-smem shrinks by FoldF). See dispatch
  // policy MainloopPPUAiuFold and the offline nfold_column_pairs_ppu (P1.1).
  static constexpr int FoldFactor = FoldF;
  using TileShape = detail::deduce_mixed_width_dtype_t<0, TileShapePair_>;
  using ScaleTileShape = cute::conditional_t<cute::is_void_v<TileShape_Scale>,
      decltype(make_shape(shape<1>(TileShape{}), Int<1>{})), TileShape_Scale>;
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using StrideB = StrideB_;
  // These are always MN major
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  // For cases where we can't have a void scale, we can use this to allow the code to compile when the scale is void.
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert((IsATransformed && cutlass::gemm::detail::is_k_major<StrideA>()) ||
                (!IsATransformed && cutlass::gemm::detail::is_k_major<StrideB>()),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                (cutlass::gemm::detail::is_k_major<StrideA>() &&
                 cutlass::gemm::detail::is_k_major<StrideB>()),
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
    "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");

  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;

  constexpr static int Scale_TileN = shape<0>(ScaleTileShape{});
  constexpr static int Scale_TileK = shape<1>(ScaleTileShape{});
  using Scale_GmemCopyThrLayoutH = Int<Scale_TileN / 8>;
  using Scale_GmemCopyThrLayoutW = Int<Scale_TileK>;
  using GmemTiledCopyScale = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementScale>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape < _8,_1>>{}));
  using GmemTiledCopyZero = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementZero>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape < _8,_1>>{}));

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale = Copy_Atom<cute::DefaultCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = IsATransformed;
  using InternalSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using InternalSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using InternalSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using InternalSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  // static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  // static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  // using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  // using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  // using InternalElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  // using InternalElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using RealInternalElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealInternalElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;

  // PPU_B_CHUNK is int1-ONLY, and the box found out the hard way: transform_B_atom calls MixGemmInt1Emit
  // unconditionally, so with the flag on it was instantiated for uint2b_t too and 576 errors followed. The gate is a
  // constexpr bool rather than #if so BOTH branches type-check for every width and only one is instantiated.
  //
  // int1 is also the only width that needs it: chunks = k-atoms per copy step = slots/delivery, which is 4 for int1,
  // 2 for int2 and 1 for int4 -- int4 has nothing to chunk, and its B fragment is already only 16 registers.
  static constexpr int kBChunkMode =
#if defined(PPU_B_CHUNK)
      (PPU_B_CHUNK);
#else
      0;
#endif
  static constexpr bool kBChunk = (kBChunkMode != 0)
                               && (cutlass::sizeof_bits<RealInternalElementB>::value == 1);
  using InternalStrideA  = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using InternalStrideB  = cute::conditional_t<!SwapAB, StrideB, StrideA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using InternalTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using InternalTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = Arch;

  using SmemLayoutAtomScale = Layout<Shape<_8, _1>>;

  static_assert(rank(InternalSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(InternalSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  // N-FOLD: B's atom N is the PHYSICAL row count (TileShape.N / FoldF), so relate them via FoldF.
  static_assert((size<1>(TileShape{}) % (size<0>(InternalSmemLayoutAtomB{}) * FoldF)) == 0,
                "fold: TileShape.N must be divisible by atomB.N * FoldF");
  // N-FOLD: B's atom K is the FOLDED run (FoldF * TileShape.K), so the original
  //   size<2>(TileShape) % size<1>(atomB) == 0  (i.e. 64 % 128) no longer holds by construction.
  // The folded relation is what must divide: atom K == FoldF * TileShape.K.
  static_assert((size<1>(InternalSmemLayoutAtomB{}) % size<2>(TileShape{})) == 0,
                "fold: B atom K must be a multiple of TileShape.K");
  static_assert(size<1>(InternalSmemLayoutAtomB{}) == FoldF * size<2>(TileShape{}),
                "fold: B atom K must equal FoldF * TileShape.K (operand Block_K must be the folded one)");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  // N-FOLD defining change = a FOLD-IN-N smem layout (verified in scratchpad/cute_nfold3.cu). The physical folded
  // run is [n_a's TK][n_b's TK]...(FoldF cols) = FoldF*TK contiguous. Instead of presenting this as (N, FoldF*TK)
  // [which would put the fold in MMA_K -> need a 2-pass mainloop + a 2xN interleaved epilogue], present it as a
  // LOGICAL (FoldF*Ng output-N, TK) tile via strides: logical n' = (f in [0,FoldF), g in [0,Ng)), phys offset =
  // f*TK + g*(FoldF*TK) + k. cute_nfold3 confirmed partition_B then puts the FoldF factor in MMA_N and MMA_K = TK/16
  // -> the MMA accumulates over the REAL TK and emits FoldF*Ng output columns NATURALLY. => STANDARD C accumulator +
  // STANDARD mainloop + STANDARD epilogue. No 2-pass, no dual accumulator, no interleaved epilogue.
  //   Ng = shape<1>(TileShape)/FoldF (physical N-groups); output N = shape<1>(TileShape).
  // NOTE: the exact generalized layout below is the FoldF=2-verified shape; finalize/verify the FoldF/Ng general
  // form (and its compatibility with the operand's swzl SmemCopyAtom, whose AiuContElemSize=FoldF*TK is reused from
  // the validated int2@TK128 / int1@TK256 config) on the box.
  static constexpr int TKe = shape<2>(TileShape{});          // real TK (A / MMA K-depth)
  static constexpr int TNe = shape<1>(TileShape{});          // output N per tile
  static constexpr int Ng  = TNe / FoldF;                    // physical N-groups
  // BUILD IT DIRECTLY -- do NOT use tile_to_shape here. Verified locally (scratchpad/cute_nfold4/5.cu):
  // tile_to_shape COALESCES the fold interleave away (atom size == tile size, so ((2,32),64):((64,128),1) collapses
  // to (64,64):(64,1) = plain row-major), which silently destroys the fold and then mismatches the operand's
  // AiuContElemSize=FoldF*TKe atom -> the collective fails to instantiate (surfacing as bogus StrideB / "no matching
  // make_cute_packed_stride" cascades at the launch site). Constructing all three modes explicitly keeps the
  // interleave: phys offset of (n'=(f,g), k, s) = s*TNe*TKe + g*FoldF*TKe + f*TKe + k  (mapping bad=0/12288 locally).
  // PHYSICAL shape (Ng, FoldF*TKe, PIPE) -- this is what the AIU writes and the swzl atom reads, and it is what the
  // gB/sB consistency asserts compare. The fold-in-N LOGICAL view (TNe output-N x TKe real-K) is recovered where the
  // MMA consumes it (see the fold_logical_B() helper), NOT here: keeping smem physical avoids a mismatch between the
  // gmem tiler, the AIU descriptor and the swzl read (that mismatch is what produced "TSM out of range").
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(Int<Ng>{}, Int<FoldF * TKe>{}, Int<DispatchPolicy::Stages>{})));
  // The fold-in-N logical view of one stage: (n'=(f,g), k) -> phys (g, f*TKe + k). Verified in cute_nfold5.cu.
  using SmemLayoutB_Logical = decltype(make_layout(
      make_shape (make_shape(Int<FoldF>{}, Int<Ng>{}), Int<TKe>{}),
      make_stride(make_stride(Int<TKe>{}, Int<FoldF * TKe>{}), _1{})));

  // It is assumed that the scales and zero-points share the same smem layout
  using SmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

private:
  static constexpr ConversionMode
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    }
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;

  static constexpr auto
  elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    }
    else if constexpr (ModeHasScales) {
      return cute::cosize_v<SmemLayoutScale>;
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
      assert(false);
    }
  }

  static constexpr auto
  elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale ) {
      return 0;
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute::cosize_v<SmemLayoutScale>;
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
      assert(false);
    }
  }

public:
  struct SharedStorage
  {
    static constexpr int scale_elements = elements_per_smem_scale();
    static constexpr int zero_elements = elements_per_smem_zero();
    cute::ArrayEngine<RealInternalElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::ArrayEngine<RealInternalElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
    cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
  };
  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    ElementScale const* ptr_S = nullptr;
    NonVoidStrideScale dS{};
    int group_size = 0;
    ElementZero const* ptr_Z = nullptr;
    int const* group_row_offsets = nullptr;   // ragged grouped: per-expert cumulative A row start; null=uniform
  };

  // Device side kernel params
  struct Params {
    GmemTiledCopyScale gmem_tiled_copy_scale;
    GmemTiledCopyZero gmem_tiled_copy_zero;

    RealInternalElementA const* ptr_A = nullptr;
    InternalStrideA dA{};
    RealInternalElementB const* ptr_B = nullptr;
    InternalStrideB dB{};

    NonVoidElementScale const* ptr_S = nullptr;
    NonVoidElementZero const* ptr_Z = nullptr;

    int group_size = 0;
    int64_t scale_k = 0;
    int reload_factor = 0;
    int const* group_row_offsets = nullptr;
  };

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  int64_t scale_residue_n = 0;
  int64_t scale_residue_k = 0;
  bool scale_valid = true;
  //
  // Methods
  //


  template <class ProblemShape>
  static Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    Params p;
    if constexpr (!SwapAB) {
      p.ptr_A = reinterpret_cast<RealInternalElementA const*>(args.ptr_A);
      p.ptr_B = reinterpret_cast<RealInternalElementB const*>(args.ptr_B);
      p.dA = args.dA;
      p.dB = args.dB;
    }
    else {
      p.ptr_A = reinterpret_cast<RealInternalElementA const*>(args.ptr_B);
      p.ptr_B = reinterpret_cast<RealInternalElementB const*>(args.ptr_A);
      p.dA = args.dB;
      p.dB = args.dA;
    }
    p.group_row_offsets = args.group_row_offsets;

    if constexpr (ModeHasScales) {
      p.gmem_tiled_copy_scale = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementScale>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape < _8,_1>>{});
      p.ptr_S = reinterpret_cast<NonVoidElementScale const*>(args.ptr_S);
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        p.gmem_tiled_copy_zero = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementZero>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape < _8,_1>>{});
        p.ptr_Z = reinterpret_cast<NonVoidElementZero const*>(args.ptr_Z);
      }
      p.group_size = args.group_size;
      p.scale_k = (get<2>(problem_shape) + args.group_size - 1) / args.group_size;
      p.reload_factor = (args.group_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    }
    return p;
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract.
  /// Returned tuple must contain at least two elements, with the first two elements being gA & gB.
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& mainloop_params) {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;

    // A init
    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    gmem_tiled_copy_A.desc_.template init<RealInternalElementA, false, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, mainloop_params.dA);
    // RAGGED grouped: expert l_coord's A starts group_row_offsets[l_coord] rows in (M is that expert's M_e).
    // When null (batched/uniform) fall back to the uniform l_coord*M offset -> identical to the original.
    int64_t a_row_off = mainloop_params.group_row_offsets ? int64_t(mainloop_params.group_row_offsets[l_coord])
                                                          : int64_t(l_coord) * int64_t(M);
    Tensor mA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_A + a_row_off * K),
                                make_shape(M,K,cute::Int<1>{}), mainloop_params.dA);                            // (m,k,1)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,0));                                                         // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});                    // (BLK_M,BLK_K,k)

    // B init (include init aiu desc)
    auto mB_nk = load_init_B(mainloop_params, N, K, L, l_coord);                                                // (n,k)
    // N-FOLD: B's PHYSICAL tile is (TileN/FoldF) x (FoldF*TileK) -- the same bytes as TileShape's (TileN, TileK) but
    // reshaped, because FoldF adjacent N-columns share one contiguous FoldF*TileK run. The gmem tile must be cut with
    // that folded tiler, otherwise the AIU descriptor is built for (TileN, TileK) while the swzl atom reads
    // (TileN/FoldF, FoldF*TileK) -> address stride mismatch -> "TSM out of range" at runtime.
    using FoldTilerB = Shape<Int<size<0>(TileShape{})>, Int<TNe / FoldF>, Int<FoldF * TKe>>;   // (M unused, N_phys, K_phys)
    Tensor gB = local_tile(mB_nk, FoldTilerB{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});                   // (BLK_N_phys,BLK_K_phys,k)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA, gB);
    }
    else if constexpr (ModeHasScales) {
      auto scale_k = mainloop_params.scale_k;
      Tensor mS_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_S), make_shape(N,scale_k,L));      // (n,scale_k,l)
      Tensor mS_nk = mS_nkl(_,_,l_coord);                                                              // (n,scale_k)
      Tensor gS = local_tile(mS_nk, ScaleTileShape{}, make_coord(n_coord, _));                         // (BLK_N, 1, scale_k)

      // init scale_residue_n
      scale_residue_n = N - size<0>(gB) * n_coord;

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA, gB, gS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_Z), make_shape(N,scale_k,L));    // (n,scale_k,l)
        Tensor mZ_nk = mZ_nkl(_,_,l_coord);
        Tensor gZ = local_tile(mZ_nk, ScaleTileShape{}, make_coord(n_coord, _));
        return cute::make_tuple(gA, gB, gS, gZ);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
      assert(false);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class... Ts,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC &accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char *smem_buf)
  {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int aiu_warp_group_thread_idx = warp_idx * 32;

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    auto k_iter_shape = cute::shape<2>(gB);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // get extra inputs
    auto extra_input_partitions = partition_extra_inputs(
        mainloop_params, load_inputs, storage, thread_idx % (Scale_GmemCopyThrLayoutH{} * Scale_GmemCopyThrLayoutW{}));

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    // N-FOLD: sB is PHYSICAL (Ng, FoldF*TKe), so its K-extent is FoldF x A's TK, and its N-extent is TNe/FoldF.
    // Both relations are checked here (gB is cut with the same folded tiler above, so gB/sB agree by construction).
    CUTE_STATIC_ASSERT_V(size<1>(sA) * Int<FoldF>{} == size<1>(sB));           // BLK_K (folded, physical)
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      auto k_iter_crd = cute::idx2crd(*k_tile_iter, k_iter_shape);
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,k_pipe),
        warp_idx
      );
      copy_async_extra_info(mainloop_params, extra_input_partitions, *k_tile_iter, k_pipe);
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA     = thr_mma.partition_fragment_A(sA(_,_,0));                // (MMA,MMA_M,MMA_K)
    // N-FOLD: the MMA must see B through the fold-in-N LOGICAL view -- (TNe output-N, TKe real-K) -- not the physical
    // (Ng, FoldF*TKe) smem shape. Partitioning the physical shape would give MMA_N=Ng and MMA_K=FoldF*TKe/16, which
    // mismatches both the accumulator's N and A's MMA_K (the two asserts below). The logical view re-labels the same
    // bytes: (n'=(f,g), k) -> phys (g, f*TKe + k)  [verified in scratchpad/cute_nfold3/5.cu, where partition_B on this
    // view puts the fold factor in MMA_N with MMA_K = TKe/16].
    // N-FOLD, division of views (this is the crux):
    //   * the swzl READ (tCrB_load / tCsB below) uses the PHYSICAL (Ng, FoldF*TKe) shape -- its atom IS physical, and
    //     handing it a logical view re-triggers "TSM out of range" (tried, reverted);
    //   * the MMA fragment uses a LOGICAL (TileShape.N, TileShape.K) VIEW of the same bytes, which is the only shape
    //     that yields ordinary N x K semantics (verified: partition on logical gives MMA_N=2/MMA_K=4 matching A, while
    //     the physical shape gives MMA_N=1/MMA_K=8 = still-folded semantics).
    // tCrB_mma is only a register destination (the converter writes into it), so using a logical view here does NOT
    // touch smem addressing and cannot overflow the TSM window.
    using SmemLayoutB_MmaView = decltype(make_layout(
        make_shape (Int<TNe>{}, Int<TKe>{}),
        make_stride(Int<TKe>{}, _1{})));       // logical (N,K) over the folded bytes; P must match L (see notes)
    Tensor tCrB_mma = thr_mma.partition_fragment_B(
        make_tensor(sB(_,_,0).data(), SmemLayoutB_MmaView{}));                 // (MMA,MMA_N,MMA_K) ordinary N x K

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                    // MMA_M
    // N-FOLD: tCrB_mma now comes from the LOGICAL (N,K) view, so it has ORDINARY semantics (MMA_N=2, MMA_K=4 matching
    // A) and the UNFOLDED equalities hold -- the fold is invisible past the ldmatrix, by design. (These two asserts
    // were left over from the abandoned Plan A, where B's fragment was folded; that mismatch is what failed here.)
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_mma) == size<2>(accum));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB_mma));                 // MMA_K

    //
    // Copy Atom retiling
    //

    using warpOnM = decltype(get<1>(tiled_mma.get_thr_layout_vmnk().shape()));
    using warpOnN = decltype(get<2>(tiled_mma.get_thr_layout_vmnk().shape()));
    using PermutationM = decltype(tiled_mma.template permutation_mnk<0>());
    using PermutationN = decltype(tiled_mma.template permutation_mnk<1>());

    using TiledMma_S8 = TiledMMA<
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
        MMA_Atom<PPU0010_16x16x32_S32S8S8S32_TN>,
#else
        MMA_Atom<PPU0015_16x16x32_S32S8S8S32_TN>,
#endif
        Layout<Shape<warpOnM, warpOnN,_1>>,
        Tile<PermutationM, PermutationN, _32>>;

    TiledMma_S8 tiled_mma_s8;
    auto thr_mma_s8 = tiled_mma_s8.get_thread_slice(thread_idx);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                       // (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto sB_s8 = recast<int8_t>(sB);
    Tensor tCrB_load =thr_mma_s8.partition_fragment_B(sB_s8(_,_,0));

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma_s8);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB_s8));             // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB_load);                                  // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // extra inputs partition and retile
    auto partitioned_extra_info = partition_extra_mma_info(tiled_mma, storage, thread_idx);
    auto copy_partitions_extra_info = retile_extra_mma_info(tiled_mma, partitioned_extra_info, thread_idx);

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrB_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrB_mma) / size<2>(tCrB_copy_view);
    // PPU_MMA_PROBE=1: print the two numbers any per-atom / chunked B scheme depends on, from the DEFAULT build.
    // Added because writing them as static_asserts and then building the dependent code on top is backwards -- the
    // assert makes a wrong assumption loud but not cheap, and it would only fire after the whole chunked path had
    // been written. One printf on the default path settles it first, for free.
#if defined(PPU_MMA_PROBE) && (PPU_MMA_PROBE != 0)
    if (thread0()) {
      cute::print("[mma probe] K_BLOCK_MAX(CPY_K)="); cute::print(K_BLOCK_MAX);
      cute::print("  K_ATOM_PER_COPY="); cute::print(K_ATOM_PER_COPY);
      cute::print("  MMA_K(tCrB_mma)="); cute::print(size<2>(tCrB_mma));
      cute::print("  MMA_N="); cute::print(size<1>(tCrB_mma));
      cute::print("  Scale_TileK="); cute::print(int(Scale_TileK));
      cute::print("\n");
    }
#endif
    // One k-atom of converted B, reused across atoms, instead of tCrB_mma's MMA_K atoms held simultaneously:
    // 4*MMA_N*MMA_K registers -> 4*MMA_N. Declared unconditionally and dead-code-eliminated when kBChunk is false,
    // because the for_each lambda below captures it by reference.
    static_assert(!kBChunk || (128 % decltype(K_ATOM_PER_COPY)::value == 0),
        "PPU_B_CHUNK: the delivery's 128 outputs must split evenly into K_ATOM_PER_COPY chunks");
    Tensor tCrB_one = make_fragment_like(tCrB_mma(_,_,Int<0>{}));

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();
      // Prefetch the first rmem from the first k-tile
      copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read);
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
if constexpr (!kBChunk) {
      transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB_mma, partitioned_extra_info, 0, K_ATOM_PER_COPY,
          copy_partitions_extra_info, smem_pipe_read);
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // THE STAGE THE CONSUMED DATA BELONGS TO, captured BEFORE smem_pipe_read advances below. The scale reload
        // must use this, not the post-advance value: with K_BLOCK_MAX == 1 the
        //   if (k_block == K_BLOCK_MAX-2 || K_BLOCK_MAX == 1) { ++smem_pipe_read; }
        // block fires EVERY iteration and sits BEFORE the mma loop, so a per-atom transform placed in that loop reads
        // an already-advanced stage. Measured, not guessed: decoding the mismatching outputs against
        // scale(g,n) = 1 + (1/16)*((5n+3g) mod 13) gave g = 2 for EVERY printed line where g = 0 was correct, and one
        // stage is Scale_TileK = 2 groups. The unchunked path never had this because its transform runs at step 2,
        // before the advance.
        int const b_consume_stage = smem_pipe_read;

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, k_block_next, smem_pipe_read);
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
if constexpr (!kBChunk) {
        transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB_mma, partitioned_extra_info, k_block_next, K_ATOM_PER_COPY,
          copy_partitions_extra_info, smem_pipe_read);
      }

        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          auto k_iter_crd = cute::idx2crd(*k_tile_iter, k_iter_shape);
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
          copy_async_extra_info(mainloop_params, extra_input_partitions, *k_tile_iter, smem_pipe_write);
          cp_async_fence();
          if (k_tile_count > 1) { ++k_tile_iter; }
        }
        if (k_block == K_BLOCK_MAX - 2 || K_BLOCK_MAX == 1) {
          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // N-FOLD: IDENTICAL to the unfolded mainloop. The fold lives only in the load layer -- gmem/smem are
        // (N/FoldF) x (FoldF*K) to satisfy the AIU 32B contiguous minimum, and the swzl ldmatrix already delivers the
        // fragment with ordinary N x K register semantics (same as int4). So no fold-specific compute code at all.
        // PPU_B_CHUNK=2 is a BISECTION, not a candidate: chunked EMISSION written at full-fragment indices, so it
        // keeps tCrB_mma (no register saving) and isolates the gating from the small-buffer plumbing. bad=0 here and
        // nonzero at PPU_B_CHUNK=1 points at tCrB_one / the pointers / the scale; nonzero here points at keep()/at().
        // A DEBUG bisection, on its OWN macro rather than sharing values with the production flag -- PPU_B_CHUNK=2 meant a
        // debug mode shipped inside the switch that turns the feature on.
#if defined(PPU_B_CHUNK_BISECT) && (PPU_B_CHUNK_BISECT != 0)
        constexpr bool kChunkFull = kBChunk;
#else
        constexpr bool kChunkFull = false;
#endif
        if constexpr (kChunkFull) {
          for_each(make_int_sequence<decltype(K_ATOM_PER_COPY)::value>{}, [&] (auto k_loop) {
            constexpr int kC = decltype(k_loop)::value;
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            Tensor dst = tCrB_mma(_,_,k_block * K_ATOM_PER_COPY);            // base of this k_block's atoms
            transform_B_atom<RealInternalElementB, kC, decltype(K_ATOM_PER_COPY)::value, false,
                             decltype(tCrB_mma.layout())>(
                tCrB_copy_view, dst, partitioned_extra_info, k_block, atom_idx,
                copy_partitions_extra_info, b_consume_stage);
          });
          CUTLASS_PRAGMA_UNROLL
          for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB_mma(_,_,atom_idx), TransformB{});
            cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB_mma(_,_,atom_idx), accum);
          }
        } else if constexpr (kBChunk) {
          // for_each, not a for loop: the chunk must be a TEMPLATE argument so the emission gate stays compile-time.
          for_each(make_int_sequence<decltype(K_ATOM_PER_COPY)::value>{}, [&] (auto k_loop) {
            constexpr int kChunk = decltype(k_loop)::value;
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            transform_B_atom<RealInternalElementB, kChunk, decltype(K_ATOM_PER_COPY)::value, true,
                             decltype(tCrB_mma.layout())>(
                tCrB_copy_view, tCrB_one, partitioned_extra_info, k_block, atom_idx,
                copy_partitions_extra_info, b_consume_stage);
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB_one, TransformB{});
            cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB_one, accum);
          });
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB_mma(_,_,atom_idx), TransformB{});
            cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB_mma(_,_,atom_idx), accum);
          }
        }
      });

    }

    cp_async_wait<0>();
    __syncthreads();
  }

private:
  CUTLASS_DEVICE
  auto load_init_B(Params const& mainloop_params, int N, int K, int L, int l_coord) {
    auto kCon = kContinous{};
    using TilerB = typename GmemTiledCopyB::Tiler_MN;
    if constexpr (kCon != 1) {
      // N-FOLD (interleaved-256 path -- THIS is the branch a %256-aligned shape takes, and it builds its own
      // shape/stride from N,K,kCon WITHOUT consulting mainloop_params.dB. A folded buffer is (N/FoldF) physical rows
      // of (FoldF*K) codes, so both must be folded here; patching only the non-interleaved branch (or only launch's
      // stride) is dead code for this path -- which is why four different offline placements all measured the same
      // ~72% random result: the gmem walk was wrong independently of the placement.
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B),
        make_shape(N / FoldF, make_shape(kCon, (K * FoldF) / kCon), L),
        // GROUPED FIX: interleaved desc base = (uint8_t*)raw_pointer_cast(mB_nk.data()) treats the packed
        // ELEMENT L-stride as a BYTE count, so the stride must be in bytes: one expert weight is
        // N*K * sizeof_bits<B>/8 bytes (int4 -> N*K/2, int2 -> N*K/4, int8 -> N*K). (Was Int<0> -> every expert
        // read plane 0; wrong scale -> e>=2 read OOB garbage. This lands each expert exactly. No effect on L=1.)
        make_stride(kCon, make_stride(cute::Int<1>{}, kCon * (N / FoldF)), int64_t(N) * int64_t(K) * sizeof_bits<RealInternalElementB>::value / 8)
      );
      Tensor mB_nk = mB_nkl(_,_,l_coord);
      auto layout_counting = make_layout(
        mB_nk.shape(),
        make_stride(ScaledBasis<_1, 1>{}, make_stride(ScaledBasis<_1, 0>{}, ScaledBasis<int, 1>{N / FoldF}))
      );
      Tensor mB_nk_counting = make_counting_tensor(layout_counting);
      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            (uint8_t*)(raw_pointer_cast(mB_nk.data())), N * K / kCon, kCon, mB_nk.stride());
      return mB_nk_counting;
    } else {
      // N-FOLD: the folded buffer is (N/FoldF) physical rows x (FoldF*K) codes; describing it as (N,K) would walk it
      // with the unfolded row pitch. dB already carries the folded pitch (set in launch), so the SHAPE must match it.
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B),
                                  make_shape(N / FoldF, K * FoldF, L), mainloop_params.dB);
      Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));

      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, N, K, mainloop_params.dB);
      return mB_nk;
    }
  }

  template <class... Ts>
  CUTLASS_DEVICE
  auto copy_async_extra_info(
        Params const& mainloop_params,
        cute::tuple<Ts...>& extra_input_partitions,
        int k_idx,
        int write_stage) {
    if constexpr (ModeHasScales) {
      auto tSgS = get<0>(extra_input_partitions);
      auto tSsS = get<1>(extra_input_partitions);
      auto tScS = get<2>(extra_input_partitions);
      // per-column path
      if constexpr(DispatchPolicy::StaticGroupSize == -1) {
        copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,0), tSsS(_,_,_,write_stage));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tZgZ = get<3>(extra_input_partitions);
          auto tZsZ = get<4>(extra_input_partitions);
          copy(mainloop_params.gmem_tiled_copy_zero, tZgZ(_,_,_,0), tZsZ(_,_,_,write_stage));
        }
      }
      else {
        int scale_load_k;
        // specific group-wise path
        if constexpr (DispatchPolicy::StaticGroupSize > 0) {
          constexpr int reload_factor = (DispatchPolicy::StaticGroupSize + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
          scale_load_k = k_idx / reload_factor;
        }
        // default path
        else {
          scale_load_k = k_idx / mainloop_params.reload_factor; // This will always be 0 when group_size == K.
        }
        if (scale_valid && (scale_load_k * Scale_TileK < scale_residue_k)) {
          copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,scale_load_k), tSsS(_,_,_,write_stage));
          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
            auto tZgZ = get<3>(extra_input_partitions);
            auto tZsZ = get<4>(extra_input_partitions);
            copy(mainloop_params.gmem_tiled_copy_zero, tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,write_stage));
          }
        }
      }
    }
  }

  template <class... Ts>
  CUTLASS_DEVICE
  auto partition_extra_inputs(
        Params const& mainloop_params,
        cute::tuple<Ts...> const& load_inputs,
        SharedStorage& shared_tensors,
        int const thread_idx) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{});
      Tensor gS = get<2>(load_inputs);
      // Construct identity layout for sS
      constexpr static Tensor cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));

      auto gmem_thr_copy_scale = mainloop_params.gmem_tiled_copy_scale.get_slice(thread_idx);

      Tensor tSgS = gmem_thr_copy_scale.partition_S(gS);
      Tensor tSsS = gmem_thr_copy_scale.partition_D(sS);
      Tensor tScS = gmem_thr_copy_scale.partition_S(cS);
      clear(tSsS);
      // init scale_residue_k
      scale_residue_k = mainloop_params.scale_k - get<1>(tScS(0,0,0));
      scale_valid = get<0>(tScS(0,0,0)) < scale_residue_n;

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS, tScS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ  = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{});
        Tensor gZ = get<3>(load_inputs);

        auto gmem_thr_copy_zero = mainloop_params.gmem_tiled_copy_zero.get_slice(thread_idx);

        Tensor tZgZ = gmem_thr_copy_zero.partition_S(gZ);
        Tensor tZsZ = gmem_thr_copy_zero.partition_D(sZ);
        clear(tZsZ);

        return cute::make_tuple(tSgS, tSsS, tScS, tZgZ, tZsZ);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");
      assert(false);
    }
  }

  // The scale/zero register fragment. The scale is k-INVARIANT, so every mma slot that differs only in k wants the
  // same value -- but the default construction materialises that replication in registers anyway:
  //
  //     partition_B(sS(_,_,0))                    ((2,2,2), MMA_N, MMA_K) : ((0,0,8), 32, 0)
  //     make_fragment_like of it (i.e. today)     ((2,2,2), MMA_N, MMA_K) : ((0,0,1),  8, 2)   <- MMA_K materialised
  //     this function                             ((2,2,2), MMA_N, MMA_K) : ((0,0,1),  2, 0)
  //
  // cute's make_fragment_like preserves stride-0s in MODE 0 ONLY (its own comment says so), which is why the two k
  // val-bits keep their zeros and MMA_K does not. That one materialised mode is a 4x in scale registers, and it also
  // enlarges what the copy has to issue: copy(Copy_Atom<...>) auto-filters on nullspace(layout<1>(dst_v)), so a
  // destination that distinguishes fewer registers is a destination the copy visits fewer times. Nothing else has to
  // change -- retile_D, the copy call and the elementwise transform are all untouched.
  //
  // The naive patch (zero MMA_K in make_fragment_like's output) does NOT work: compact_order follows the REFERENCE
  // strides and MMA_K's reference stride is 0, so it sorts MMA_K ahead of MMA_N and leaves MMA_N a non-compact 8
  // (cosize 26 instead of 8). filter_zeros first, so MMA_K is extent 1 during compaction, THEN restore its zero.
  //
  // Derived and verified in fold_derivation/l28_scale_bcast.cu: cosize equals the copy's distinct-element count on
  // every config in the tree, and -- the check that matters -- the equivalence classes this layout induces on mma
  // slots coincide EXACTLY with the classes induced by n, so no slot can receive another n's scale. Stated as classes
  // rather than as an index formula on purpose: two earlier attempts failed by guessing filter_zeros' ordering.
  // ONE entry point for the scale/zero register fragment, shared by all three collectives so the construction cannot
  // drift between them. The body is the plain cute idiom, deliberately.
  //
  // AN EARLIER VERSION OF THIS FUNCTION HAND-ASSEMBLED A STRIDE-0 "BROADCAST" LAYOUT here, on the theory that the
  // fragment materialised a 4x replication of each scale (32 half slots holding 8 distinct values). acu says that is
  // a NO-OP: the slots are register-resident, the loop is fully unrolled and the equalities are provable, so the
  // compiler had already coalesced them -- and it had CSE'd the copy's filtered iterations down to the 8 distinct
  // addresses too. Measured registers were IDENTICAL either way (and 186 against estimates of 176/164, i.e. above
  // both, so the fragment is buried under address arithmetic anyway).
  //
  // It was reverted for a second reason that outlives the measurement: that version replaced a one-line cute idiom
  // with a hand-built stride tuple that hardcoded "mode 2 is MMA_K" by position. Less cute-native, rank-fragile, and
  // it bought nothing. A cute layout describes what the PROGRAM asks for; whether the hardware does it is a codegen
  // question, and for register-resident provably-equal values the compiler wins first. Check that a cute-level
  // redundancy survives to the ISA before trading idiom for it.
  template <class TiledMma, class STensor>
  CUTLASS_DEVICE
  static auto make_scale_fragment(TiledMma const& thr_mma, STensor const& sS) {
    return make_fragment_like<ElementScale>(thr_mma.partition_fragment_B(sS(_,_,Int<0>{})));
  }

  // The same layout, HOST-callable, for the compile-time witness below (make_scale_fragment is CUTLASS_DEVICE and
  // cannot appear even unevaluated in a host constexpr context). make_fragment_like<T>(t) is
  // make_tensor<T>(make_layout_like(t.layout())), so these two stay in step by construction.
  template <class TiledMma, class STensor>
  CUTE_HOST_DEVICE static constexpr auto scale_fragment_layout(TiledMma const& thr_mma, STensor const& sS) {
    return make_layout_like(thr_mma.partition_fragment_B(sS(_,_,Int<0>{})).layout());
  }

  // Kept from the reverted work because it has independent value: it is what let a STALE submodule be distinguished
  // from an inert change. cosize is the scale fragment's real register footprint in halves.
  static constexpr int scale_frag_cosize() {
    if constexpr (ModeHasScales) {
      return cute::cosize_v<decltype(scale_fragment_layout(
          TiledMma{}.get_thread_slice(0),
          make_tensor(make_smem_ptr((NonVoidElementScale*)nullptr), SmemLayoutScale{})))>;
    } else {
      return 0;
    }
  }

  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.
  template <class TiledMma>
  CUTLASS_DEVICE
  auto partition_extra_mma_info(
    TiledMma const& tiled_mma,
    SharedStorage& storage,
    int thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      auto smem_tiled_copy_S   = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S     = smem_tiled_copy_S.get_thread_slice(thread_idx);

      static constexpr int smem_scale_k = Scale_TileK * DispatchPolicy::Stages;
      using SmemCopyLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{},
          make_shape(shape<0>(ScaleTileShape{}), Int<1>{}, Int<smem_scale_k>{})));
      Tensor sS   = make_tensor(make_smem_ptr(storage.smem_scale.begin()), SmemCopyLayoutScale{});
      Tensor tCsS = smem_thr_copy_S.partition_S(sS);
      Tensor tCrS = make_scale_fragment(thr_mma, sS);

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ   = make_tensor(make_smem_ptr(storage.smem_zero.begin()), SmemCopyLayoutScale{});
        Tensor tCsZ = smem_thr_copy_S.partition_S(sZ);
        Tensor tCrZ = make_scale_fragment(thr_mma, sZ);
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      assert(false);
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE
  auto retile_extra_mma_info(
    TiledMma const& tiled_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    int const thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S   = smem_tiled_copy_S.get_thread_slice(thread_idx);
      Tensor tCrS_copy_view  = smem_thr_copy_S.retile_D(cute::get<1>(partitioned_extra_info));        // (CPY,CPY_N,CPY_K)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view  = smem_thr_copy_S.retile_D(cute::get<3>(partitioned_extra_info));      // (CPY,CPY_N,CPY_K)
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view, tCrZ_copy_view);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      assert(false);
    }
  }

  /// Utilities to copy B and extra inputs from smem to RF
  template <class SmemTiledCopy,
            class TensorSmemView,
            class TensorCopyView,
            class... Ts,
            class... Us
            >
  CUTLASS_DEVICE
  void copy_B_and_extra_info(
    SmemTiledCopy const& smem_tiled_copy_B,
    TensorSmemView const& tCsB,
    TensorCopyView& tCrB_copy_view,
    cute::tuple<Ts...> const& partitioned_mma_extra_info,
    cute::tuple<Us...> const& tiled_copy_and_views,
    int k_block,
    int read_stage) {

    copy(smem_tiled_copy_B, tCsB(_,_,k_block,read_stage), tCrB_copy_view(_,_,k_block));

    // COARSE scale path: gs spans >= one full B copy step (Scale_TileK <= K_BLOCK_MAX), so one scale group
    // covers >=1 whole copy steps -> load it here, once per GroupK steps. The FINE case (gs < copy-step K, e.g.
    // gs=32 with a 64-K single-step copy -> Scale_TileK > K_BLOCK_MAX) makes GroupK=0 (a div-by-zero) and one
    // group can't cover the whole step; there the scale is loaded PER mma-atom in transform_B_kblock instead.
    constexpr int KBM_ = decltype(cute::size<2>(tCrB_copy_view))::value;
    if constexpr (int(Scale_TileK) <= KBM_) {
     auto GroupK= size<2>(tCrB_copy_view) / Scale_TileK;
     if (k_block % GroupK == 0) {
      // We are starting a new group k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      }
      else if constexpr (ModeHasScales) {
        const int scale_k_idx = read_stage * Scale_TileK + k_block / GroupK;
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCsS              = cute::get<0>(partitioned_mma_extra_info);
        auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
        copy(smem_tiled_copy_S, tCsS(_,_,0,scale_k_idx), tCrS_copy_view(_,_,0));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ              = cute::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view    = cute::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_,_,0,scale_k_idx), tCrZ_copy_view(_,_,0));
        } else {
          // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
          assert(false);
        }
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        assert(false);
      }
     }  // if (k_block % GroupK == 0)
    }   // if constexpr (Scale_TileK <= K_BLOCK_MAX)  [COARSE]
  }
  /// Utilities to transform B.
  // FINE-grained scale (gs < B-copy-step K, i.e. Scale_TileK > K_BLOCK_MAX): a single copy step's K_ATOM_PER_COPY
  // mma atoms straddle MORE than one scale group, so one pre-loaded scale reg can't cover the step (and the coarse
  // GroupK = K_BLOCK_MAX/Scale_TileK is 0). Here each mma atom reloads ITS group's scale straight from smem:
  // atom `a` (0..mma_K_atoms) belongs to group a/(mma_K_atoms/Scale_TileK), at smem slot read_stage*Scale_TileK+g.
  // This needs tiled_copy_and_views (smem_tiled_copy_S + reg-copy dst views) + read_stage, so both are passed in.
  // ONE place for the per-atom scale rule. transform_B_kblock loops its atoms and calls this; transform_B_atom calls
  // it once. They each had their own copy of the FINE / APG_ / reload-at-group-boundary logic -- one rule with two
  // implementations, which is where the next divergence was going to come from (the chunked path already had a
  // "> 1" where the other had "> KBM_", right by accident only because chunking is int1-only).
  //
  // It also removes a hazard the two copies handled differently: the non-FINE branches used a LOCAL `auto tCrS`,
  // which make_fragment_like makes owning, so it snapshots stale rmem after a reload. The FINE branches read
  // get<1>(info) directly for exactly that reason. This always reads get<1>(info), which is correct in both cases --
  // in the non-FINE case no reload happens, so the two are the same object.
  template <bool FINE, int APG, class BSlice, class... Ts, class CopyViews>
  CUTLASS_DEVICE
  void apply_scale_atom(BSlice&& b_slice,
                        cute::tuple<Ts...> const& info,
                        CopyViews const& views,
                        int const atom_idx,
                        int const read_stage) {
    if constexpr (ModeHasScales) {
      if constexpr (FINE) {
        auto smem_tiled_copy_S = cute::get<0>(views);
        auto tCsS              = cute::get<0>(info);
        auto tCrS_copy_view    = cute::get<1>(views);
        if (atom_idx % APG == 0) {                        // reload only at a scale group's first atom
          const int sk = read_stage * int(Scale_TileK) + atom_idx / APG;
          copy(smem_tiled_copy_S, tCsS(_,_,0,sk), tCrS_copy_view(_,_,0));
          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
            auto tCsZ           = cute::get<2>(info);
            auto tCrZ_copy_view = cute::get<2>(views);
            copy(smem_tiled_copy_S, tCsZ(_,_,0,sk), tCrZ_copy_view(_,_,0));
          }
        }
      }
      cute::transform(b_slice, cute::get<1>(info)(_,_,0), b_slice, cute::multiplies{});
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        cute::transform(b_slice, cute::get<3>(info)(_,_,0), b_slice, cute::plus{});
      }
    }
  }

  template <typename RealInternalElementB,
            class TCrB_load,
            class TCrB_mma,
            int K_ATOM_PER_COPY,
            class... Ts,
            class CopyViews>
  CUTLASS_DEVICE
  void transform_B_kblock(
    TCrB_load const& tCrB_load,
    TCrB_mma& tCrB_mma,
    cute::tuple<Ts...> const& partitioned_extra_info,
    int const k_block,
    cute::Int<K_ATOM_PER_COPY> k_atom,
    CopyViews const& tiled_copy_and_views,
    int const read_stage) {

    Tensor cvt_in  = recast<RealInternalElementB>(tCrB_load(_, _, k_block));
    Tensor cvt_out = make_tensor(tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(), cvt_in.layout());

    using CPY_VEC = Int<4 * 32 / sizeof_bits<RealInternalElementB>::value>;
    convert_tensor(cvt_in, cvt_out, CPY_VEC{});

    constexpr int MMA_KA_ = decltype(cute::size<2>(tCrB_mma))::value;   // total mma-K atoms in the tile
    constexpr int KBM_    = MMA_KA_ / K_ATOM_PER_COPY;                  // K_BLOCK_MAX (copy steps)
    constexpr bool FINE   = (int(Scale_TileK) > KBM_);                  // gs < copy-step K -> per-atom scale
    constexpr int APG_    = FINE ? (MMA_KA_ / int(Scale_TileK)) : 1;    // mma atoms per scale group (FINE only)

    // The four hand-rolled branches (FINE x zero) collapsed into one call: the per-atom rule now lives in
    // apply_scale_atom and both this function and transform_B_atom go through it.
    if constexpr (ModeHasScales) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < K_ATOM_PER_COPY; ++i) {
        const int atom_idx = k_block * K_ATOM_PER_COPY + i;
        apply_scale_atom<FINE, APG_>(tCrB_mma(_, _, atom_idx), partitioned_extra_info,
                                     tiled_copy_and_views, atom_idx, read_stage);
      }
    }
  }

  // PPU_B_CHUNK: convert and scale ONE k-atom into a one-atom fragment, instead of converting a whole delivery into
  // a fragment that holds all MMA_K atoms. See fold_derivation/HANDOFF_TASK9.md for the why; the short version is that
  // a 16 B delivery carries 16/Bits mma atom-slots, so the fp16 fragment is FORCED to >= 4*16/Bits = 64 registers for
  // int1, and those 64 push the best config past the power-of-two register billing boundary, halving warps/CU from 32
  // to 16. Converting one atom at a time drops B to 64/MMA_K with every delivered code still converted and used --
  // only the emission order changes, not the mma count or the total converter work.
  //
  // Chunk must be a TEMPLATE parameter: the emission gate is `if constexpr (MixGemmEmit index / kPer == Chunk)`, a
  // compile-time predicate (verified static for all widths in l32_chunk_predicate.cu), and a runtime `if` would make
  // the register saving depend on the compiler folding branches -- the assumption the scale-broadcast episode
  // punished.
  template <class RealB, int Chunk, int NChunk, bool Rebase, class FragL,
            class TCrB_load, class TCrB_one, class... Ts, class CopyViews>
  CUTLASS_DEVICE
  void transform_B_atom(
    TCrB_load const& tCrB_load,
    TCrB_one& tCrB_one,
    cute::tuple<Ts...> const& partitioned_extra_info,
    int const k_block,
    int const atom_idx,
    CopyViews const& tiled_copy_and_views,
    int const read_stage) {

    // NO CROSS-ITERATION HAZARD, and no dependence on CPY_K. tCrB_load has ONE SLOT PER COPY STEP -- the copy writes
    // tCrB_copy_view(_,_,k_block_next) while this reads (_,_,k_block) -- so deferring the conversion into the mma loop
    // cannot see overwritten codes. An earlier version hardcoded Int<0> here and then asserted CPY_K == 1 to protect
    // its own shortcut, which made a limitation of the implementation look like a property of the design. It is not:
    // "convert inside the loop instead of all at once" works for any K_BLOCK_MAX.
    //
    // For the record, CPY_K is DERIVABLE rather than assumption: it is slots/delivery in fp16 units, so at
    // WN=64/TK=64 it is 1 for int1 (delivery 128 == slots 128), 2 for int2 and 4 for int4. NChunk is therefore the
    // atoms per copy step, passed in rather than hardcoded.
    // raw_pointer_cast FIRST. tCrB_load is a SUBBYTE tensor, so .data() is a cute::subbyte_iterator and
    // reinterpret_cast from that class type to uint32_t const* is ill-formed -- which is exactly how the box build
    // failed. convert_tensor already had the right idiom a few lines below: raw_pointer_cast(t.data()) then cast.
    Tensor cvt_in = recast<RealB>(tCrB_load(_, _, k_block));
    // FragL is tCrB_mma's OWN layout, passed in rather than restated: at()/keep() are compositions over it, so a
    // change to the mma atom or the warp tile propagates instead of silently invalidating hand-typed strides.
    cutlass::MixGemmInt1Emit<Chunk, NChunk, Rebase, FragL>::emit(
        reinterpret_cast<uint32_t const*>(raw_pointer_cast(cvt_in.data())),
        reinterpret_cast<uint32_t*>(raw_pointer_cast(tCrB_one.data())));

    // Same rule, same code: apply_scale_atom. This block used to be a second copy of it -- and carried a
    // "Scale_TileK > 1" where the other had "> KBM_", right only by accident because chunking is int1-only and
    // int1's KBM_ is 1.
    constexpr int KBM_    = decltype(cute::size<2>(tCrB_load))::value;   // copy steps per k-tile
    constexpr int MMA_KA_ = NChunk * KBM_;                               // total mma-K atoms in the tile
    constexpr bool FINE   = (int(Scale_TileK) > KBM_);
    constexpr int APG_    = FINE ? (MMA_KA_ / int(Scale_TileK)) : 1;
    apply_scale_atom<FINE, APG_>(tCrB_one, partitioned_extra_info, tiled_copy_and_views, atom_idx, read_stage);
  }

  /// Utilities for transforming the A operand prior to issuing tensor cell math.
  template <class EngineIn,
            class EngineOut,
            class TensorLayout,
            int ConversionVectorWidth = cosize_v<TensorLayout>>
  CUTLASS_DEVICE void
  convert_tensor(
    Tensor<EngineIn,TensorLayout> const& in,
    Tensor<EngineOut,TensorLayout>& out,
    cute::Int<ConversionVectorWidth> width = {}) {

    /// This is an element-wise conversion where we expect both tensors to have the same layout.
    /// As a result, we can cast as a cutlass array to use the fast numeric converters without
    /// worrying about indexing into the layout.
    constexpr int N = size(TensorLayout{});
    // constexpr int N = cosize_v<TensorLayout>;

    /// The inputs must be backed by registers & be statically sized.
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_static_v<TensorLayout>, "Tensor layout for the conversion must be static");
    // static_assert(cosize_v<TensorLayout> == size(TensorLayout{}), "Cosize and size of the layout must be equal.");
    static_assert(N % ConversionVectorWidth == 0, "Conversion vector width must divide cosize of the tensor layout.");

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    using SrcArray = cutlass::Array<SrcType, ConversionVectorWidth>;
    using DstArray = cutlass::Array<DstType, ConversionVectorWidth>;

    // constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    // using Converter = cutlass::NumericArrayConverter<DstType, SrcType, ConversionVectorWidth, RoundStyle>;

    // SrcType int8_t consider as uint8_t
    using Converter = cutlass::MixGemmNumericArrayConverter<DstType, SrcType, ConversionVectorWidth>;

    constexpr int NumIterations = N / ConversionVectorWidth;

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < NumIterations; ++ii) {
      SrcArray const* src_array_ptr = reinterpret_cast<SrcArray const*>(raw_pointer_cast(in(_, ii).data()));
      DstArray* dst_array_ptr = reinterpret_cast<DstArray*>(raw_pointer_cast(out(_, ii).data()));
      *dst_array_ptr = Converter::convert(*src_array_ptr);
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
