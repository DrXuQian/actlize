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
// =============================================================================================================
// TWO-B-PLANE mixed-input mainloop: B arrives as TWO bit planes that are combined in the converter so a GGUF
// 3/5/6-bit weight runs as ONE GEMM (Q3 = int2 low + int1 high, Q5 = int4+int1, Q6 = int4+int2). This keeps the
// low-bit MEMORY win, which the alternative (pad the 3 bits into a nibble and reuse the int4 path) throws away.
//
// This is a SEPARATE collective on purpose -- see memory ppu-bconcat-bitplane / ppu-moe-format-collective-arch
// ("one collective per format"). Retrofitting the single-plane mainloop with if-constexpr guards was tried and
// reverted: with plane 2 absent its types are void, which forced conditional_t fallbacks, alias-to-plane-1 dummy
// tensors and static helper fns -- none of which exist here, where BOTH planes are unconditional. It also keeps
// all regression risk away from the validated single-plane paths (int4 59.7% / int2 50.8% / int1 48.1% MFU).
//
// KEEP IN SYNC with ppu_mma_aiu_multistage_mixed_input.hpp: everything except the plane-2 additions is a copy of
// it, so pipeline/scale/zero fixes must be mirrored in both files.
//
// Sizing: both planes share ONE Block_K, so the tile is bounded below by the SPARSEST plane's AIU 32-byte minimum
// (int1 -> Block_K >= 256, int2 -> >= 128). Cross-plane element correspondence is VERIFIED (scratchpad/xplane.py,
// bad=0/4096): int1's uint32 vector j1 == int2's vector j2>>1, and int1's lop3 pair == int2's pair + 8*(j2&1) --
// so BOTH planes keep their existing, already-validated offline relayouts and no new bit derivation is needed.
// =============================================================================================================

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
// Transport wrapper: hands this mainloop the SECOND plane's three B-side atoms through the EXISTING single
// template parameters, because CollectiveMma's parameter list is fixed by its primary template.
// A dedicated wrapper is required -- do NOT use cute::is_tuple: a cute Layout (which SmemLayoutAtomB_ is) is
// itself tuple-like, so is_tuple would be a false positive.
template <class T0, class T1>
struct BPlanes {};

namespace bplane_detail {
  template <class T> struct first  { using type = T; };
  template <class T0, class T1> struct first<BPlanes<T0, T1>>  { using type = T0; };
  template <class T> struct second { using type = void; };
  template <class T0, class T1> struct second<BPlanes<T0, T1>> { using type = T1; };
}
template <class T> using bplane_first_t  = typename bplane_detail::first<T>::type;
template <class T> using bplane_second_t = typename bplane_detail::second<T>::type;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Arch,
  int Stages,
  class kContinous,
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
    MainloopPPUAiuMixedInput2Plane<Stages, kContinous, KernelSchedule>,
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
  using DispatchPolicy =  MainloopPPUAiuMixedInput2Plane<Stages, kContinous, KernelSchedule>;
  using TileShape = detail::deduce_mixed_width_dtype_t<0, TileShapePair_>;
  using ScaleTileShape = cute::conditional_t<cute::is_void_v<TileShape_Scale>,
      decltype(make_shape(shape<1>(TileShape{}), Int<1>{})), TileShape_Scale>;
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  // The SECOND B plane rides the 4th member of the B element tuple:
  //     cute::tuple<ElementB, ElementScale, ElementZero, PlaneB2>
  // Unlike the single-plane mainloop this is MANDATORY -- this collective exists only for the 2-plane case.
  using PlaneB2 = detail::deduce_mixed_width_dtype_t<3, ElementBOptionalTuple>;
  static_assert(!cute::is_void_v<PlaneB2>,
    "2-plane mainloop: pass the high plane as the 4th member of the B element tuple");
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
  using GmemTiledCopyB  = bplane_first_t<GmemTiledCopyB_>;
  using GmemTiledCopyB2 = bplane_second_t<GmemTiledCopyB_>;
  static_assert(!cute::is_void_v<GmemTiledCopyB2>,
    "2-plane mainloop: GmemTiledCopyB must be BPlanes<plane0,plane1>");

  constexpr static int Scale_TileN = shape<0>(ScaleTileShape{});
  constexpr static int Scale_TileK = shape<1>(ScaleTileShape{});
  // THE SCALE COPY MUST NOT ASK FOR MORE THREADS THAN THE CTA HAS. H used to be Scale_TileN/8 unconditionally, so the
  // thread layout demanded (Scale_TileN/8) * Scale_TileK slots; the copy is then sliced by
  // `thread_idx % (H*W)`, which silently truncates when the CTA is smaller. At (64,128,128) w64x64 gs=16 that is 16*8 =
  // 128 slots against 64 threads (WOM = TM/WM = 1 halves the CTA), and the layout (16,8) maps t -> (t%16, t/16), so
  // threads 0..63 cover only w in [0,4): FOUR OF EIGHT SCALE GROUPS ARE NEVER LOADED. Measured as bad=13358/32768,
  // localised by the one-variable-per-rung ladder in test_q3_bconcat_real (rungs 1-3 MATCH, rung 4 = WM 32->64 fails).
  //
  // Fix: cap H at NumThreads/W and give each thread the remaining elements. ElemsPerThr stays a multiple of the atom's
  // 8 half_t, so make_tiled_copy just issues several cp.async per thread. Every previously working configuration keeps
  // its old H (there H*W was already <= NumThreads), so this is a no-op for them.
  constexpr static int Scale_NumThreads = size(TiledMma{});
  constexpr static int Scale_ThrH = cute::min(Scale_TileN / 8, Scale_NumThreads / Scale_TileK);
  static_assert(Scale_ThrH >= 1 && (Scale_TileN % (Scale_ThrH * 8)) == 0,
      "scale copy: Scale_TileN must split into ThrH threads of a multiple of 8 elements");
  // The durable guard. partition_extra_inputs slices this copy with `thread_idx % (ThrH*ThrW)`, so a layout that
  // asks for more slots than the CTA has does not fail -- it silently drops every slot above the thread count.
  // That is how the old (Scale_TileN/8, Scale_TileK) form survived until a config with WOM == 1 halved the CTA:
  // (64,128,128) w64x64 gs=16 wanted 16*8 = 128 slots against 64 threads and loaded 4 of 8 scale groups.
  static_assert(Scale_ThrH * Scale_TileK <= Scale_NumThreads,
      "scale copy asks for more thread slots than the CTA has -- the modulo slice would silently truncate it");
  constexpr static int Scale_ElemsPerThr = Scale_TileN / Scale_ThrH;
  using Scale_GmemCopyThrLayoutH = Int<Scale_ThrH>;
  using Scale_GmemCopyThrLayoutW = Int<Scale_TileK>;
  using GmemTiledCopyScale = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementScale>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape <Int<Scale_ElemsPerThr>,_1>>{}));
  using GmemTiledCopyZero = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementZero>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape <Int<Scale_ElemsPerThr>,_1>>{}));

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB  = bplane_first_t<SmemLayoutAtomB_>;
  using SmemLayoutAtomB2 = bplane_second_t<SmemLayoutAtomB_>;
  static_assert(!cute::is_void_v<SmemLayoutAtomB2>,
    "2-plane mainloop: SmemLayoutAtomB must be BPlanes<plane0,plane1>");

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB  = bplane_first_t<SmemCopyAtomB_>;
  using SmemCopyAtomB2 = bplane_second_t<SmemCopyAtomB_>;
  static_assert(!cute::is_void_v<SmemCopyAtomB2>,
    "2-plane mainloop: SmemCopyAtomB must be BPlanes<plane0,plane1>");
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
  // PPU_B_CHUNK on the 2-plane path (task #12). Same flag as the fold collective, and the same reason it is a
  // constexpr bool rather than an #if: an #if leaves the other branch un-type-checked, which is how an int1-only
  // emitter got instantiated for uint2b_t and produced 576 errors. Gated on the plane WIDTHS, because
  // MixGemm2Plane_uint2_uint1 is exactly (low int2, high int1) -- any other pair must not reach it.
  static constexpr int kBChunkMode =
#if defined(PPU_B_CHUNK)
      (PPU_B_CHUNK);
#else
      0;
#endif
  static constexpr bool kBChunk = (kBChunkMode != 0)
                               && (cutlass::sizeof_bits<RealInternalElementB>::value == 2)
                               && (cutlass::sizeof_bits<PlaneB2>::value == 1);
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
  // PER-PLANE N-FOLD: B's atom N is the PHYSICAL row count (TileShape.N / Fold) and its atom K is the FOLDED run
  // (Fold * TileShape.K), so the unfolded "atom must divide the tile" relations do not hold -- the FOLDED ones do. The
  // single-plane fold collective carries the same pair for the same reason. Both degenerate to the originals at
  // Fold == 1. (P1Fold/P2Fold are defined below, next to the smem layouts they size.)
  static_assert((size<1>(TileShape{}) % (size<0>(InternalSmemLayoutAtomB{})
                * (size<1>(InternalSmemLayoutAtomB{}) / size<2>(TileShape{})))) == 0,
                "fold: TileShape.N must be divisible by atomB.N * FoldF");
  static_assert((size<1>(InternalSmemLayoutAtomB{}) % size<2>(TileShape{})) == 0,
                "fold: B atom K must be an integer multiple of TileShape.K (that multiple IS the fold factor)");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  // PER-PLANE N-FOLD, LOW plane. Its fold factor is read off its OWN atom, which the builder already sizes folded
  // (BFoldBlockK = FoldF * blockK) -- the same trick P2Fold uses, so no fold factor has to travel through the dispatch
  // policy. P1Fold == 1 reproduces the previous layout exactly.
  static constexpr int P1Fold = size<1>(InternalSmemLayoutAtomB{}) / size<2>(TileShape{});
  static_assert(P1Fold >= 1 && (size<1>(InternalSmemLayoutAtomB{}) % size<2>(TileShape{})) == 0,
      "plane 1's atom K must be an integer multiple of TileShape.K (that multiple IS its fold factor)");
  static_assert((size<1>(TileShape{}) % P1Fold) == 0, "plane 1's fold must divide TileShape.N");
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(Int<size<1>(TileShape{}) / P1Fold>{}, Int<P1Fold * size<2>(TileShape{})>{},
                 Int<DispatchPolicy::Stages>{})));
  // The fold-in-N LOGICAL view the MMA fragment must be partitioned from when P1Fold > 1: (n'=(f,g), k) -> phys
  // (g, f*TKe + k). Partitioning the PHYSICAL shape would give MMA_N = Ng and MMA_K = P1Fold*TKe/16, mismatching the
  // accumulator's N and A's MMA_K. Verified in the single-plane fold collective, whose comment block this mirrors.
  using SmemLayoutB_MmaView = decltype(make_layout(
      make_shape (make_shape(Int<P1Fold>{}, Int<size<1>(TileShape{}) / P1Fold>{}), Int<size<2>(TileShape{})>{}),
      make_stride(make_stride(Int<size<2>(TileShape{})>{}, Int<P1Fold * size<2>(TileShape{})>{}), _1{})));

  // PER-PLANE N-FOLD. Plane 2 covers the same LOGICAL (N,K) extent, but PHYSICALLY it may be folded harder than plane
  // 1: the builder gave it (Block_N/P2Fold, P2Fold*Block_K) so its contiguous run reaches the AIU's 32 B minimum. The
  // fold factor is read straight OFF THE ATOM rather than passed in again -- the builder already encoded it, and a
  // second copy of the rule is where the next divergence comes from.
  //
  // Physical, not logical, exactly as the single-plane fold collective does: the AIU writes and the swzl atom reads
  // the physical shape, and presenting a logical view to them is what produced "TSM out of range" there. Plane 2 never
  // feeds the mma fragment (only plane 1 does, through tCrB_mma), so no logical view is needed for it at all.
  static constexpr int P2Fold = size<1>(SmemLayoutAtomB2{}) / size<2>(TileShape{});
  static_assert(P2Fold >= 1 && (size<1>(SmemLayoutAtomB2{}) % size<2>(TileShape{})) == 0,
      "plane 2's atom K must be an integer multiple of TileShape.K (that multiple IS its fold factor)");
  static_assert((size<1>(TileShape{}) % P2Fold) == 0, "plane 2's fold must divide TileShape.N");
  using SmemLayoutB2 = decltype(tile_to_shape(
      SmemLayoutAtomB2{},
      make_shape(Int<size<1>(TileShape{}) / P2Fold>{}, Int<P2Fold * size<2>(TileShape{})>{},
                 Int<DispatchPolicy::Stages>{})));

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
    cute::ArrayEngine<PlaneB2, cute::cosize_v<SmemLayoutB2>> smem_b2;   // 2nd bit plane
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
    // 2nd bit plane. Deliberately LAST so callers' positional brace-init of the fields above is unchanged; set it
    // separately (args.mainloop.ptr_B2 = ...).
    PlaneB2 const* ptr_B2 = nullptr;
    // PER-PLANE N-FOLD: plane 2 may be folded harder than plane 1, in which case its PHYSICAL buffer is
    // (N/P2Fold) rows x (K*P2Fold) codes and its row pitch differs from dB. Leave dB2_valid false to reuse dB, which
    // is correct exactly when the two planes share a fold factor (P2Fold == 1) -- the pre-existing behaviour.
    StrideB dB2{};
    bool dB2_valid = false;
  };

  // Device side kernel params
  struct Params {
    GmemTiledCopyScale gmem_tiled_copy_scale;
    GmemTiledCopyZero gmem_tiled_copy_zero;

    RealInternalElementA const* ptr_A = nullptr;
    InternalStrideA dA{};
    RealInternalElementB const* ptr_B = nullptr;
    InternalStrideB dB{};
    PlaneB2 const* ptr_B2 = nullptr;
    StrideB dB2{};
    bool dB2_valid = false;

    NonVoidElementScale const* ptr_S = nullptr;
    NonVoidElementZero const* ptr_Z = nullptr;

    int group_size = 0;
    int64_t scale_k = 0;
    int reload_factor = 0;
    int const* group_row_offsets = nullptr;
  };

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  GmemTiledCopyB2 gmem_tiled_copy_B2;   // 2nd plane (carries its own AIU desc_)
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
    // Bit-plane concat only makes sense with the narrow operand as B; the 2nd plane rides the same dB.
    static_assert(!SwapAB, "2-plane mainloop requires the narrow operand to be B (SwapAB=false)");
    p.ptr_B2 = args.ptr_B2;
    p.dB2 = args.dB2;
    p.dB2_valid = args.dB2_valid;
    p.group_row_offsets = args.group_row_offsets;

    if constexpr (ModeHasScales) {
      // Assign the TYPE, do not rebuild the layout. This site had the thread/value layouts written out a second time,
      // so capping ThrH at the CTA size above made the rebuilt type diverge from GmemTiledCopyScale and the assignment
      // stopped compiling. One definition, one use.
      p.gmem_tiled_copy_scale = GmemTiledCopyScale{};
      p.ptr_S = reinterpret_cast<NonVoidElementScale const*>(args.ptr_S);
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        p.gmem_tiled_copy_zero = GmemTiledCopyZero{};
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
    // PER-PLANE N-FOLD: same as plane 2 -- a folded gmem tensor is (N/P1Fold, K*P1Fold) physically, so the per-CTA
    // tile must be cut with the folded tiler or copy.hpp's size<1>(src) == size<1>(dst) fires.
    using FoldTilerB1 = Shape<Int<size<0>(TileShape{})>,
                              Int<size<1>(TileShape{}) / P1Fold>,
                              Int<P1Fold * size<2>(TileShape{})>>;
    Tensor gB = local_tile(mB_nk, FoldTilerB1{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});                    // (BLK_N_phys,BLK_K_phys,k)

    // Plane 2: same tiling as gB, appended LAST to the returned tuple. Safe because both kernels only index
    // get<0>/get<1>, static_assert size>=2, and forward the tail opaquely.
    auto mB2_nk = load_init_B2(mainloop_params, N, K, L, l_coord);
    // PER-PLANE N-FOLD: plane 2's gmem tensor is PHYSICALLY (N/P2Fold, K*P2Fold), so it must be cut with the folded
    // tiler. Cutting it with TileShape gives a (TileN, TileK) gmem tile against a (TileN/P2Fold, P2Fold*TileK) smem
    // tile -- the copy's `size<1>(src) == size<1>(dst)` static_assert then fires, which is exactly how the box build
    // failed. The single-plane fold collective carries the same FoldTilerB for the same reason (there the symptom was
    // an AIU descriptor built for the unfolded shape -> "TSM out of range" at runtime, which is worse than a compile
    // error). P2Fold == 1 reproduces TileShape exactly.
    using FoldTilerB2 = Shape<Int<size<0>(TileShape{})>,
                              Int<size<1>(TileShape{}) / P2Fold>,
                              Int<P2Fold * size<2>(TileShape{})>>;
    Tensor gB2 = local_tile(mB2_nk, FoldTilerB2{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA, gB, gB2);
    }
    else if constexpr (ModeHasScales) {
      auto scale_k = mainloop_params.scale_k;
      Tensor mS_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_S), make_shape(N,scale_k,L));      // (n,scale_k,l)
      Tensor mS_nk = mS_nkl(_,_,l_coord);                                                              // (n,scale_k)
      Tensor gS = local_tile(mS_nk, ScaleTileShape{}, make_coord(n_coord, _));                         // (BLK_N, 1, scale_k)

      // init scale_residue_n
      scale_residue_n = N - size<0>(gB) * n_coord;

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA, gB, gS, gB2);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_Z), make_shape(N,scale_k,L));    // (n,scale_k,l)
        Tensor mZ_nk = mZ_nkl(_,_,l_coord);
        Tensor gZ = local_tile(mZ_nk, ScaleTileShape{}, make_coord(n_coord, _));
        return cute::make_tuple(gA, gB, gS, gZ, gB2);
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
    Tensor gB2 = get<sizeof...(Ts) - 1>(load_inputs);      // plane 2 is tuple_cat'd last by load_init
    auto k_iter_shape = cute::shape<2>(gB);
    // PER-PLANE N-FOLD: plane 2's k-iteration shape is NOT plane 1's. local_tile divides the interleaved counting
    // tensor's NESTED K mode (kCon, K/kCon) by the tiler's K, and the two planes now use different tiler Ks -- so at
    // Block_K=128 plane 1's k mode comes out NESTED (2,1) while plane 2's folded tiler (F2*TK == kCon) divides it
    // exactly and leaves a SCALAR. Feeding plane 1's tuple coordinate to plane 2 is the box failure at 2plane.hpp:808
    // ("no matching function", Coords = <Underscore, packed_tuple<int,int>>): it is a COORDINATE SHAPE mismatch, not
    // the rank mismatch it first looked like -- both partition_S results are rank 4 (derived in
    // fold_derivation/l43_partition_rank.cu, with the real AIU copy atom). Identical shapes when P2Fold == 1.
    auto k_iter_shape_b2 = cute::shape<2>(gB2);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    Tensor sB2 = make_tensor(make_smem_ptr(storage.smem_b2.begin()), SmemLayoutB2{}); // 2nd plane

    // get extra inputs
    auto extra_input_partitions = partition_extra_inputs(
        mainloop_params, load_inputs, storage, thread_idx % (Scale_GmemCopyThrLayoutH{} * Scale_GmemCopyThrLayoutW{}));

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    // PER-PLANE N-FOLD: sB's K extent is the PHYSICAL folded run (P1Fold * BLK_K) while sA's is the real BLK_K, so the
    // equality only holds unfolded. Assert the folded relation instead of dropping the check -- it still catches a
    // mismatched B operand Block_K, which is what it was there for.
    static_assert(size<1>(SmemLayoutB{}) == P1Fold * size<2>(TileShape{}),
                  "fold: sB's physical BLK_K must be P1Fold * TileShape.K");
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    auto gmem_thr_copy_B2 = gmem_tiled_copy_B2.get_slice(thread_idx);
    Tensor tB2gB2 = gmem_thr_copy_B2.partition_S(gB2);
    Tensor tB2sB2 = gmem_thr_copy_B2.partition_D(sB2);

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      auto k_iter_crd    = cute::idx2crd(*k_tile_iter, k_iter_shape);
      auto k_iter_crd_b2 = cute::idx2crd(*k_tile_iter, k_iter_shape_b2);
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,k_pipe),
        warp_idx
      );
      copy_aiu(gmem_tiled_copy_B2, tB2gB2(_,_,_,k_iter_crd_b2), tB2sB2(_,_,_,k_pipe), warp_idx);
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
    // PER-PLANE N-FOLD: the MMA must see B through the fold-in-N LOGICAL view, not the physical smem shape. At
    // P1Fold == 1 SmemLayoutB_MmaView is the plain row-major (TN, TK) and this is the previous expression.
    Tensor tCrB_mma = thr_mma.partition_fragment_B(
        make_tensor(sB(_,_,0).data(), SmemLayoutB_MmaView{}));                 // (MMA,MMA_N,MMA_K) ordinary N x K
    // PPU_B_CHUNK: one k-atom of converted B, reused across atoms, instead of tCrB_mma's MMA_K atoms held at once:
    // 4*MMA_N*MMA_K fp16 registers -> 4*MMA_N. Declared unconditionally and dead-code-eliminated when kBChunk is
    // false, because the mainloop lambda captures it by reference.
    Tensor tCrB_one = make_fragment_like(tCrB_mma(_,_,Int<0>{}));

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                    // MMA_M
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

    // Plane 2: its own swzl atom (different CUBE_W, since its byte extent is 2x/4x smaller for the same Block_K).
    auto sB2_s8 = recast<int8_t>(sB2);
    Tensor tCrB2_load = thr_mma_s8.partition_fragment_B(sB2_s8(_,_,0));
    auto smem_tiled_copy_B2 = make_tiled_copy_B(SmemCopyAtomB2{}, tiled_mma_s8);
    auto smem_thr_copy_B2   = smem_tiled_copy_B2.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB2            = smem_thr_copy_B2.partition_S(make_mix_tensor_like(sB2_s8));
    Tensor tCrB2_copy_view  = smem_thr_copy_B2.retile_D(tCrB2_load);
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
    // ONE plane-2 k_block feeds P2_DIV plane-1 k_blocks: plane 2 is the DENSER plane (same BYTES per copy, 2x the
    // codes at half the bit width), so its CPY_K is P2_DIV times smaller. transform_B_kblock derives the same
    // ratio to pick which half of the plane-2 registers a given plane-1 k_block owns.
    constexpr int P2_DIV = decltype(cute::size<2>(tCrB_copy_view))::value
                         / decltype(cute::size<2>(tCrB2_copy_view))::value;
    static_assert(P2_DIV >= 1, "plane 2's CPY_K must not exceed plane 1's");
    // The chunk gate splits ONE int2 delivery (16 B = 64 codes per n = 8 mma B atoms) into 8 chunks, so the collective
    // must agree that a copy step is 8 atoms. It is structurally so for int2 -- the delivery width fixes it, and l40
    // confirms 8 at both (64,64,256) and (32,128,256) -- but assert it rather than trust it: if K_ATOM_PER_COPY were
    // SMALLER the loop would emit only part of the delivery and silently drop half2 slots, which is the failure mode
    // that reads as a plausible-but-wrong result rather than a crash.
    // A delivery is kAtoms mma atoms, split between n and k by tCrB_mma: K_ATOM_PER_COPY * NAPC == kAtoms. Testing
    // K_ATOM_PER_COPY alone was equivalent only while NAPC == 1, i.e. Block_K >= 128; it fired at Block_K=64, which is
    // now the fastest configuration on the box. Measured for all three Block_K in fold_derivation/l62.
    static_assert(!kBChunk || decltype(K_ATOM_PER_COPY)::value
                              * (decltype(cute::size<1>(tCrB_mma))::value / decltype(cute::size<1>(tCrB_copy_view))::value)
                              == MixGemm2Plane_uint2_uint1<>::kAtoms,
        "PPU_B_CHUNK (2-plane): K_ATOM_PER_COPY * (MMA_N / CPY_N) must be the kAtoms one int2 delivery carries");

    // PPU_MMA_PROBE=1: the same "let the kernel report its own indices" probe the fold collective has, plus the two
    // LAYOUTS -- because chunking B here needs to know what the emission's index space actually is, and a local cute
    // stub cannot answer it. This collective's tiled_mma carries PermutationM/N from the builder and its B fragment is
    // loaded through an int8 m16n16k32 atom (NOT the fp16 k16 one), so neither the fragment's mode order nor CPY_K can
    // be reproduced offline. What a stub CAN establish, and did (fold_derivation/l39_2plane_frag.cu): every fold-path
    // config has MMA_N == MMA_K == 4, so 8*MMA_K and 8*MMA_N coincide there and no fold measurement distinguishes
    // k-inner from k-outer. At TK=256, MMA_N=2 and MMA_K=16 -- they differ by 8x. Print it instead of assuming it.
#if defined(PPU_MMA_PROBE) && (PPU_MMA_PROBE != 0)
    if (thread0()) {
      cute::print("[2plane probe] K_BLOCK_MAX(CPY_K)="); cute::print(K_BLOCK_MAX);
      cute::print("  K_ATOM_PER_COPY="); cute::print(K_ATOM_PER_COPY);
      cute::print("  P2_DIV="); cute::print(P2_DIV);
      cute::print("  Scale_TileK="); cute::print(int(Scale_TileK));
      cute::print("\n  tCrB_mma      : "); cute::print(tCrB_mma.layout());
      cute::print("\n  tCrB_load     : "); cute::print(tCrB_load.layout());
      cute::print("\n  tCrB_copy_view: "); cute::print(tCrB_copy_view.layout());
      cute::print("\n  tCrB2_load    : "); cute::print(tCrB2_load.layout());
      // The emission index space: cvt_in's layout is what the converter's flat output pointer walks, and cvt_out
      // aliases tCrB_mma's registers THROUGH it. If cvt_in's mode-1 stride != tCrB_mma's MMA_N stride, a chunk gate
      // built on tCrB_mma's layout is built on the wrong space.
      cute::print("\n  cvt_in(=recast<B>(tCrB_load(_,_,0))): ");
      cute::print(recast<RealInternalElementB>(tCrB_load(_, _, 0)).layout());
      cute::print("\n  MMA_N stride of tCrB_mma="); cute::print(stride<1>(tCrB_mma.layout()));
      cute::print("  8*MMA_K="); cute::print(8 * int(size<2>(tCrB_mma)));
      cute::print("  8*MMA_N="); cute::print(8 * int(size<1>(tCrB_mma)));
      cute::print("\n");
    }
#endif

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();
      // Prefetch the first rmem from the first k-tile
      copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read);
      // Plane 2's smem->rmem swzl read. WITHOUT this the plane-2 registers stay untouched (read as zero), the
      // converter's high-bit OR contributes nothing, and D degenerates to the low plane exactly -- which is what
      // the controlled-input probe measured (rung1 got == exp-4 for high=ALL ONES, decoded index identically 0).
      copy(smem_tiled_copy_B2, tCsB2(_,_,Int<0>{},smem_pipe_read), tCrB2_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      // Chunked: the mma loop converts each atom just before its gemm, so there is nothing to do for the whole step.
      if constexpr (!kBChunk) {
        transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB2_copy_view, tCrB_mma, partitioned_extra_info, 0,
          K_ATOM_PER_COPY, copy_partitions_extra_info, smem_pipe_read);
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

        // THE STAGE THIS ITERATION'S B CODES CAME FROM, captured BEFORE the advance below. With K_BLOCK_MAX == 1 the
        //   if (k_block == K_BLOCK_MAX-2 || K_BLOCK_MAX == 1) { ++smem_pipe_read; }
        // block fires EVERY iteration and sits BEFORE the mma loop, so the chunked per-atom transform -- which runs
        // inside that loop -- would otherwise read the ALREADY-ADVANCED stage for its scale. On the fold path that
        // showed up as bad=85545 with every wrong line decoding to scale group g=2 where g=0 was right.
        int const b_consume_stage = smem_pipe_read;

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, k_block_next, smem_pipe_read);
        // Plane 2 (see the prologue site). Re-reading the same plane-2 k_block on each of the P2_DIV plane-1
        // k_blocks it serves is idempotent; gate it on (k_block_next % P2_DIV == 0) once this MATCHes.
        // CHUNKED: this prefetch is a READ-AFTER-CLOBBER hazard and is skipped. Plane 2 has P2_DIV times FEWER copy
        // slots than plane 1 -- at the shipping shape it has exactly ONE, shared by both k_blocks -- so on the last
        // k_block, where smem_pipe_read has already advanced, it overwrites that slot with the NEXT tile's codes. In
        // the unchunked flow that is harmless (the conversion already ran, the registers hold fp16); with the
        // conversion deferred into the mma loop the loop would read the next tile's high bits for half its k-atoms.
        // Wrong but plausible-looking output -- so plane 2 is instead re-read per k_block from b_consume_stage below.
        if constexpr (!kBChunk) {
          copy(smem_tiled_copy_B2, tCsB2(_,_,Int<decltype(k_block_next)::value / P2_DIV>{},smem_pipe_read),
               tCrB2_copy_view(_,_,Int<decltype(k_block_next)::value / P2_DIV>{}));
        }
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        if constexpr (!kBChunk) {
          transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB2_copy_view, tCrB_mma, partitioned_extra_info,
            k_block_next, K_ATOM_PER_COPY, copy_partitions_extra_info, smem_pipe_read);
        }

        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          auto k_iter_crd    = cute::idx2crd(*k_tile_iter, k_iter_shape);
          auto k_iter_crd_b2 = cute::idx2crd(*k_tile_iter, k_iter_shape_b2);
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
          copy_aiu(gmem_tiled_copy_B2, tB2gB2(_,_,_,k_iter_crd_b2), tB2sB2(_,_,_,smem_pipe_write), warp_idx);
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

        if constexpr (kBChunk) {
          // Plane 2, read HERE from the stage this iteration actually consumes -- see the skipped prefetch above.
          // Idempotent and cheap (one swzl read of a slot that serves P2_DIV k_blocks), and it removes the hazard
          // rather than reasoning about when the clobber is survivable.
          constexpr int kP2Slot = decltype(k_block)::value / P2_DIV;
          copy(smem_tiled_copy_B2, tCsB2(_,_,Int<kP2Slot>{}, b_consume_stage), tCrB2_copy_view(_,_,Int<kP2Slot>{}));
          // for_each, not a for loop: the chunk must be a TEMPLATE argument so the emission gate stays compile-time.
          for_each(make_int_sequence<decltype(K_ATOM_PER_COPY)::value>{}, [&] (auto k_loop) {
            constexpr int kChunk = decltype(k_loop)::value;
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            transform_B_atom<RealInternalElementB, kChunk, decltype(K_ATOM_PER_COPY)::value>(
                tCrB_copy_view, tCrB2_copy_view, tCrB_one, partitioned_extra_info, k_block, atom_idx,
                copy_partitions_extra_info, b_consume_stage);
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB_one, TransformB{});
            cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB_one, accum);
          });
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            // Transform before compute
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB_mma(_,_,atom_idx), TransformB{});
            // gemm for one tiled_mma atom on K
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
    // PER-PLANE N-FOLD: a folded plane-1 buffer is (N/P1Fold) physical rows of (K*P1Fold) codes, so BOTH shape and
    // stride must be folded. dB already carries the folded pitch (moe_grouped_ppu builds it from n/MOEG_FOLD), so the
    // SHAPE has to match it. P1Fold == 1 is the previous code exactly.
    const int N1_ = N / P1Fold, K1_ = K * P1Fold;
    auto kCon = kContinous{};
    using TilerB = typename GmemTiledCopyB::Tiler_MN;
    if constexpr (kCon != 1) {
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B),
        make_shape(N1_, make_shape(kCon, K1_ / kCon), L),
        // GROUPED FIX: interleaved desc base = (uint8_t*)raw_pointer_cast(mB_nk.data()) treats the packed
        // ELEMENT L-stride as a BYTE count, so the stride must be in bytes: one expert weight is
        // N*K * sizeof_bits<B>/8 bytes (int4 -> N*K/2, int2 -> N*K/4, int8 -> N*K). (Was Int<0> -> every expert
        // read plane 0; wrong scale -> e>=2 read OOB garbage. This lands each expert exactly. No effect on L=1.)
        make_stride(kCon, make_stride(cute::Int<1>{}, kCon * N1_), int64_t(N) * int64_t(K) * sizeof_bits<RealInternalElementB>::value / 8)
      );
      Tensor mB_nk = mB_nkl(_,_,l_coord);
      auto layout_counting = make_layout(
        mB_nk.shape(),
        make_stride(ScaledBasis<_1, 1>{}, make_stride(ScaledBasis<_1, 0>{}, ScaledBasis<int, 1>{N1_}))
      );
      Tensor mB_nk_counting = make_counting_tensor(layout_counting);
      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            (uint8_t*)(raw_pointer_cast(mB_nk.data())), N1_ * K1_ / kCon, kCon, mB_nk.stride());
      return mB_nk_counting;
    } else {
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B), make_shape(N1_, K1_, L), mainloop_params.dB);
      Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));

      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, N1_, K1_, mainloop_params.dB);
      return mB_nk;
    }
  }

  // Plane 2's gmem tensor + AIU descriptor. Exact mirror of load_init_B; note the per-expert L-stride uses
  // PlaneB2's OWN bit width, since the interleaved desc treats that byte count as the plane stride.
  CUTLASS_DEVICE
  auto load_init_B2(Params const& mainloop_params, int N, int K, int L, int l_coord) {
    auto kCon = kContinous{};
    using TilerB2 = typename GmemTiledCopyB2::Tiler_MN;
    // PER-PLANE N-FOLD: a folded plane-2 buffer is (N/P2Fold) physical rows of (K*P2Fold) codes, so BOTH the shape and
    // the stride must be folded here. The single-plane fold collective learned this the expensive way: the
    // interleaved branch builds its own shape/stride from N,K,kCon WITHOUT consulting dB, so patching only the
    // non-interleaved branch is dead code for a %256-aligned shape -- four different offline placements then all
    // measured the same ~72% random result because the gmem WALK was wrong independently of the placement.
    // P2Fold == 1 reproduces the previous code exactly.
    const int N2 = N / P2Fold, K2 = K * P2Fold;
    if constexpr (kCon != 1) {
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B2),
        make_shape(N2, make_shape(kCon, K2 / kCon), L),
        make_stride(kCon, make_stride(cute::Int<1>{}, kCon * N2),
                    int64_t(N) * int64_t(K) * sizeof_bits<PlaneB2>::value / 8)   // L-stride is fold-invariant (bytes)
      );
      Tensor mB_nk = mB_nkl(_,_,l_coord);
      auto layout_counting = make_layout(
        mB_nk.shape(),
        make_stride(ScaledBasis<_1, 1>{}, make_stride(ScaledBasis<_1, 0>{}, ScaledBasis<int, 1>{N2}))
      );
      Tensor mB_nk_counting = make_counting_tensor(layout_counting);
      gmem_tiled_copy_B2.desc_.template init<PlaneB2, false, get<0>(TilerB2{}), get<1>(TilerB2{})>(
            (uint8_t*)(raw_pointer_cast(mB_nk.data())), N2 * K2 / kCon, kCon, mB_nk.stride());
      return mB_nk_counting;
    } else {
      // dB2 carries plane 2's OWN folded row pitch (set in launch); the SHAPE must match it. Falls back to dB when the
      // caller has not supplied one, which is the unfolded case where the two coincide.
      auto d2 = mainloop_params.dB2_valid ? mainloop_params.dB2 : mainloop_params.dB;
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B2), make_shape(N2, K2, L), d2);
      Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));
      gmem_tiled_copy_B2.desc_.template init<PlaneB2, false, get<0>(TilerB2{}), get<1>(TilerB2{})>(
            nullptr, N2, K2, d2);
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
  template <typename RealInternalElementB,
            class TCrB_load,
            class TCrB2_load,
            class TCrB_mma,
            int K_ATOM_PER_COPY,
            class... Ts,
            class CopyViews>
  CUTLASS_DEVICE
  void transform_B_kblock(
    TCrB_load const& tCrB_load,
    TCrB2_load const& tCrB2_load,
    TCrB_mma& tCrB_mma,
    cute::tuple<Ts...> const& partitioned_extra_info,
    int const k_block,
    cute::Int<K_ATOM_PER_COPY> k_atom,
    CopyViews const& tiled_copy_and_views,
    int const read_stage) {

    constexpr int P2_DIV_ = decltype(cute::size<2>(tCrB_load))::value
                          / decltype(cute::size<2>(tCrB2_load))::value;
    static_assert(P2_DIV_ >= 1, "plane 2 must not have MORE K steps than plane 1 (it is the denser plane)");

    // ---- TWO-PLANE convert -------------------------------------------------------------------------------
    // Plane 2 is DENSER (more elements per uint32), so its copy view has FEWER K steps: one plane-2 step serves
    // P2_DIV plane-1 k_blocks. (int2 low / int1 high at the same Block_K: sB_s8 is [N][64] int8 while sB2_s8 is
    // [N][32], hence P2_DIV == 2.) Within plane 2's step, plane-1 k_block kb uses the vreg pair starting at
    // 2*(kb % P2_DIV) -- the tensor-level twin of the verified per-register rule (hi reg = lo reg v >> 1).
    Tensor cvt_in  = recast<RealInternalElementB>(tCrB_load(_, _, k_block));
    Tensor cvt_hi  = recast<PlaneB2>(tCrB2_load(_, _, k_block / P2_DIV_));

    // Write through the TENSOR, exactly like the single-plane convert_tensor does (`out(_, ii)`), NOT through a
    // linear pointer: tCrB_mma's fragment layout is not contiguous with the stride a linear walk would assume, so
    // a linear write scatters the fp16 into the wrong register slots (symptom: |D| 30-100x too large with random
    // signs, which cannot come from wrong BITS since q = low + 4*high is bounded by 7).
    // Mode-0 of cvt_in is one CPY_VEC chunk (64 low codes = 4 vregs); mode-1 is the chunk count.
    Tensor cvt_out = make_tensor(tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(), cvt_in.layout());
    constexpr int NumIter = decltype(cute::size<1>(cvt_in))::value;
    static_assert(decltype(cute::size<0>(cvt_in))::value == 64,
      "2-plane convert expects a 64-code low chunk (CPY_VEC for uint2); generalize before changing that width");
    static_assert(decltype(cute::size<0>(cvt_hi))::value == 128,
      "expected plane 2's chunk to hold 128 codes (2x the low chunk) -- it is the denser plane");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < NumIter; ++ii) {
      uint32_t const* lo_p = reinterpret_cast<uint32_t const*>(raw_pointer_cast(cvt_in(_, ii).data()));
      uint32_t*       o_p  = reinterpret_cast<uint32_t*>(raw_pointer_cast(cvt_out(_, ii).data()));
      // same N-chunk of plane 2, offset to the half of its 128 codes that THIS k_block owns
      // STRIDE 1, not 2: the high fragment's delivered vreg index decomposes as (k_block parity) + 2*(N-half), so
      // the two vregs a given low k_block owns are 2 APART (kb and kb+2), not adjacent. Offset by kb here; the
      // converter then indexes hi[2*(v>>1)]. DERIVED from the two validated single-plane converters -- see the
      // derivation block above MixGemm2Plane_uint2_uint1 in fast_numeric_conversion_for_mix_gemm.h.
      // PER-PLANE N-FOLD. With the shipped offset (k_block % P2_DIV) a folded plane 2 has P2_DIV == 1, so every ii
      // reads vregs {0, 2} and vregs {1, 3} are NEVER touched -- HALF the tile's high bits cannot arrive, whatever the
      // placement. The surviving N index must move into the vreg offset.
      //
      // This is the same expression that was reverted once. It is back because the derivation behind it is now GATED
      // END TO END, which it was not then: fold_derivation/l49 composes offline o delivery o converter-pairing and
      // requires the identity, and BOTH sides are validated byte-for-byte against the shipped offline -- plane 1 in
      // l52 (int2, DL=1 and DL=2), plane 2 in l51/l53 (int1 folded, at K=512 and at the box's K=256). At the unfolded
      // config, which measures bad=0 on the box, the composition reports 0 mispaired out of 32768.
      //
      // The run that measured 29666 was NOT a clean test of this: that pull also carried the whole batch of actlize
      // portability fixes (template-argument trailing commas, the `template` disambiguator, dim3, cast_smem_ptr_to_uint
      // and mma_ppu.h qualifiers). Two changes, one measurement.
      constexpr int N2_ = decltype(cute::size<1>(cvt_hi))::value;
      uint32_t const* hi_p = reinterpret_cast<uint32_t const*>(raw_pointer_cast(cvt_hi(_, ii % N2_).data()))
                           + (k_block % P2_DIV_) + P2_DIV_ * (ii / N2_);
      MixGemm2Plane_uint2_uint1<>::convert(lo_p, hi_p, o_p);   // <> == the full 32-half2 delivery
    }

    constexpr int MMA_KA_ = decltype(cute::size<2>(tCrB_mma))::value;   // total mma-K atoms in the tile
    constexpr int KBM_    = MMA_KA_ / K_ATOM_PER_COPY;                  // K_BLOCK_MAX (copy steps)
    constexpr bool FINE   = (int(Scale_TileK) > KBM_);                  // gs < copy-step K -> per-atom scale
    constexpr int APG_    = FINE ? (MMA_KA_ / int(Scale_TileK)) : 1;    // mma atoms per scale group (FINE only)

    // ONE per-atom scale rule (apply_scale_atom below), looped over this copy step's atoms. This block used to be
    // four hand-rolled copies of the FINE / APG_ / reload-at-group-boundary logic -- and two of them used a LOCAL
    // `auto tCrS`, which make_fragment_like makes OWNING, so it snapshots stale rmem after a reload. The chunked path
    // added below needs the same rule for a single atom, and one rule with two implementations is where the next
    // divergence comes from; the fold collective was de-duplicated the same way for the same reason.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < K_ATOM_PER_COPY; ++i) {
      int const atom_idx = k_block * K_ATOM_PER_COPY + i;
      apply_scale_atom<FINE, APG_>(tCrB_mma(_, _, atom_idx), partitioned_extra_info,
                                   tiled_copy_and_views, atom_idx, read_stage);
    }
  }

  // ONE place for the per-atom scale rule -- transform_B_kblock loops its atoms and calls this, transform_B_atom calls
  // it once. Always reads get<1>/get<3>(info) directly rather than a local snapshot, which is correct with and without
  // a reload (without one, they are the same object).
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

  // PPU_B_CHUNK (task #12): convert ONE k-atom of BOTH planes into a one-atom fp16 buffer, so B costs 4*MMA_N
  // registers instead of 4*MMA_N*MMA_K. Chunk must be a TEMPLATE parameter -- the emission gate is
  // `if constexpr (keep(T, V))`, and a runtime `if` would make the saving depend on the compiler folding branches.
  //
  // ONE GATE COVERS BOTH PLANES, and that is what makes this small rather than a fourth hand-derived index map: each
  // _E2 line reads the low crumb AND the high bit and writes ONE half2, so gating the line gates both
  // (fold_derivation/l38, verified for NChunk 2 and 4). The gate itself is at_plain/4 == Chunk, at_plain%4 -- a plain
  // division rather than the fold path's right_inverse composition, because l40 showed cvt_in's mode-1 stride equals
  // tCrB_mma's MMA_N stride here, so each `ii` fills 64 CONTIGUOUS fp16 = 8 k-atoms of one n with k inner.
  //
  // NO CROSS-ITERATION HAZARD: tCrB_load has one slot per copy step and the prefetch writes k_block_next while this
  // reads k_block, so deferring the conversion into the mma loop cannot see overwritten codes. Note the plane-2 read
  // is idempotent across the P2_DIV plane-1 k_blocks it serves.
  template <class RealB, int Chunk, int NChunk,
            class TCrB_load, class TCrB2_load, class TCrB_one, class... Ts, class CopyViews>
  CUTLASS_DEVICE
  void transform_B_atom(
    TCrB_load const& tCrB_load,
    TCrB2_load const& tCrB2_load,
    TCrB_one& tCrB_one,
    cute::tuple<Ts...> const& partitioned_extra_info,
    int const k_block,
    int const atom_idx,
    CopyViews const& tiled_copy_and_views,
    int const read_stage) {

    constexpr int P2_DIV_ = decltype(cute::size<2>(tCrB_load))::value
                          / decltype(cute::size<2>(tCrB2_load))::value;
    // raw_pointer_cast FIRST -- these are SUBBYTE tensors, so .data() is a cute::subbyte_iterator and
    // reinterpret_cast from that class type is ill-formed (exactly how the fold build failed on the box).
    Tensor cvt_in = recast<RealB>(tCrB_load(_, _, k_block));
    Tensor cvt_hi = recast<PlaneB2>(tCrB2_load(_, _, k_block / P2_DIV_));
    constexpr int NumIter = decltype(cute::size<1>(cvt_in))::value;          // CPY_N -- NOT MMA_N in general
    // ONE DELIVERY IS 8 MMA ATOMS, and tCrB_mma decides how they split between n and k -- 1 n x 8 k while Block_K >= 128,
    // but 2 n x 4 k at Block_K = 64, where MMA_N is 4 while CPY_N is only 2. The previous form looped Chunk over
    // K_ATOM_PER_COPY alone and wrote at `out + 4*ii`, which is correct only when CPY_N == MMA_N: at Block_K=64 it
    // emitted 4 of the delivery's 8 atoms and filled 2 of tCrB_one's 4 n-atoms, leaving the other two never written --
    // the same class of defect as hi_vreg0 (an index that assumed the copy and mma N extents agree).
    //
    // Derived from tCrB_mma's own layout and cvt_in's real mode-1 stride, then GATED at NAPC 1 and 2 across seven
    // configurations including the shipped Block_K=256 (fold_derivation/l63):
    //     Chunk_actual = Chunk + NChunk * n_local          destination atom = ii * NAPC + n_local
    // Both collapse to the previous expression when NAPC == 1, so this is a strict generalisation. NOTE l63's first
    // version hardcoded cvt_in's stride as 32 half2 and declared Block_K=256 broken; the stride is 64 there. The
    // harness was wrong, not the code -- ask cute for a stride, never write it down.
    constexpr int MMA_N_ = decltype(cute::size(tCrB_one))::value / 8;        // 8 fp16 per mma B atom
    constexpr int NAPC_  = MMA_N_ / NumIter;                                 // n-atoms carried by one delivery
    static_assert(NAPC_ >= 1 && MMA_N_ == NAPC_ * NumIter, "tCrB_one's n extent must be a multiple of CPY_N");
    static_assert(NChunk * NAPC_ == MixGemm2Plane_uint2_uint1<>::kAtoms,
        "a delivery is kAtoms mma atoms; K_ATOM_PER_COPY * NAPC must account for all of them");
    uint32_t* out = reinterpret_cast<uint32_t*>(raw_pointer_cast(tCrB_one.data()));
    // for_each, not a loop: Chunk_actual must be a TEMPLATE argument so the emission gate stays compile-time.
    cute::for_each(cute::make_int_sequence<NumIter>{}, [&] (auto ii_) {
      constexpr int ii = decltype(ii_)::value;
      uint32_t const* lo_p = reinterpret_cast<uint32_t const*>(raw_pointer_cast(cvt_in(_, cute::Int<ii>{}).data()));
      // THE HIGH-PLANE SOURCE INDEX MUST CARRY THE N INDEX, and this copy of the expression did not. The unchunked
      // path (~line 1340) was fixed for the per-plane fold; this one kept the shipped `k_block % P2_DIV` form, and a
      // folded plane 2 has P2_DIV == 1, so every ii read vregs {0, 2} while {1, 3} were NEVER touched -- half the
      // tile's high bits could not arrive, whatever the placement. Measured with PPU_B_CHUNK=1: control (Block_K=256,
      // where P2_DIV is 2 and the old form happens to be right) MATCH, every folded rung bad ~= 15000/32768, i.e.
      // ~46% -- the signature of "half the int1 contributions missing", not of a wrong placement.
      //
      // ii indexes the DELIVERY, so this depends on ii and k_block only, never on n_local: the low plane's 64 codes
      // and the high plane's 128 pair up per (t, v) INSIDE a delivery, and the chunk merely selects which of those
      // slots are emitted. Both n_local values therefore share this base pointer.
      //
      // Must stay identical to the unchunked expression. That is now gated rather than trusted -- l63 checks the
      // high-plane source index of BOTH paths, so a future edit to one of them fails locally instead of on the box.
      constexpr int N2_ = decltype(cute::size<1>(cvt_hi))::value;
      uint32_t const* hi_p = reinterpret_cast<uint32_t const*>(
                                 raw_pointer_cast(cvt_hi(_, cute::Int<ii % N2_>{}).data()))
                           + (k_block % P2_DIV_) + P2_DIV_ * (ii / N2_);
      cute::for_each(cute::make_int_sequence<NAPC_>{}, [&] (auto nl_) {
        constexpr int nl = decltype(nl_)::value;
        MixGemm2Plane_uint2_uint1<Chunk + NChunk * nl>::convert(lo_p, hi_p, out + 4 * (ii * NAPC_ + nl));
      });
    });

    constexpr int KBM_    = decltype(cute::size<2>(tCrB_load))::value;       // copy steps per k-tile
    constexpr int MMA_KA_ = NChunk * KBM_;                                   // total mma-K atoms in the tile
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
