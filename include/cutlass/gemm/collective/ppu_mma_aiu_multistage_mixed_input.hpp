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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gguf_packed_scale.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {
// Instantiates composition() ONLY when selected: naming both branches of a conditional_t instantiates both, and the
// unselected one fires cute's "Requires pow2 shape*stride".
// PER Scale_TileN, because the conflict map depends on it through the mma B operand's TV layout. A member template
// cannot be explicitly specialised inside the class, so the table lives here. A width with no entry gets the identity
// swizzle and keeps the plain map rather than pretending to be fixed.
// The pattern is closed-form -- Swizzle<2, 3, log2(TN) - 2> -- because the donor bits are always the group index's
// bits 1 and 2, and the group stride is TN halfs, so they sit at bit log2(TN)+1; MBase stays 3 so the low three bits
// are untouched and the 16 B cp.async keeps its contiguity. Written as explicit specialisations anyway, so the table
// claims exactly the three widths l98 measured and any other width gets the identity and keeps the plain map. A
// formula would have silently extended a result to widths nobody checked, which is how the first version of this
// table -- one entry, applied to all three widths -- left TN=32 at 4-way while advertising 1-way.
template <int TN> struct ScaleSwizzleFor      { using type = cute::Swizzle<0, 4, 4>; };   // identity
template <>       struct ScaleSwizzleFor<32>  { using type = cute::Swizzle<2, 3, 3>; };   // l98: 4-way -> 1-way
template <>       struct ScaleSwizzleFor<64>  { using type = cute::Swizzle<2, 3, 4>; };   // l98: 4-way -> 1-way
template <>       struct ScaleSwizzleFor<128> { using type = cute::Swizzle<2, 3, 5>; };   // l98: 4-way -> 1-way

template <bool On, class Swz, class L> struct MaybeScaleSwizzle { using type = L; };
template <class Swz, class L> struct MaybeScaleSwizzle<true, Swz, L> {
  using type = decltype(cute::composition(Swz{}, L{}));
};
}  // namespace cutlass::gemm::collective::detail



// ONE definition of the packed cube pitch, used by BOTH sides. The read's pitch is baked into A's atom by the
// builder and the write's is computed in the collective; as two separate literals they diverged -- 16 against 64 --
// and the kernel wrote at one spacing, read at another, and faulted with an invalid VA. 64 halfs = 128 B keeps every
// cube base and the whole span 128-B aligned, which smem_b's AIU descriptor needs (its alignment used to hold only
// because A's byte count happened to be a multiple of 32).
#ifndef PPU_A_PACK_PITCH
#define PPU_A_PACK_PITCH 64
#endif

namespace cutlass::gemm::collective {

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
    MainloopPPUAiuMixedInput<Stages, kContinous, KernelSchedule>,
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
  using DispatchPolicy =  MainloopPPUAiuMixedInput<Stages, kContinous, KernelSchedule>;
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

  // ---------------------------------------------------------------------------------------------------------------
  // THE NATIVE (PACKED) SCALE CHANNEL -- plan #20 option E. TYPES ONLY in this commit; nothing below consumes them yet,
  // and PPU_PACKED_SCALE off leaves every existing type byte-identical.
  //
  // The channel carries the gguf's own scale bytes instead of two pre-multiplied fp16 planes: for Q4_K one 16 B unit per
  // (superblock, column) holding d, dmin and the 8+8 six-bit codes, reordered offline into two self-contained 6-byte
  // halves so a k-tile covering half a superblock reads half the unit (`fold_derivation/l94`, PackBits layout).
  //
  // Every number here was measured locally, not assumed:
  //   * the tile is TN x 16 B = 2048 B at TN=128, i.e. THE SAME BYTES as today's scale tile alone -- the zero tile
  //     disappears, so the channel halves from 4.0 to 2.0 B per (group, column).
  //   * one thread per COLUMN, 16 B each: 16 B aligned, and consecutive threads on consecutive chunks, i.e. one fully
  //     coalesced burst, where today's (TN/8, SK) x (_8,_1) map is strided. l94 (6): contiguity/alignment/coalescing/
  //     coverage all 0 bad.
  //   * smem banks: 1-way on all 32, against 4-way on 4 today (l94 (2), counting DISTINCT addresses -- lanes sharing an
  //     address broadcast, and the scale fragment is deliberately k-broadcast). Config dependent: it holds because warp 0
  //     touches 8 CONSECUTIVE columns, so re-run l94 when the warp shape changes.
  //   * TileK=64 must stay on the fp16 path: 2 groups per tile would read 10 B, i.e. 5.0 B per (group, column), worse
  //     than fp16's 4.0. TileK >= 128 wins (2.5 B) and TileK == 256 is the best case (2.0 B).
#if defined(PPU_PACKED_FORMAT)
  static constexpr cutlass::gguf_packed::Fmt kPackedFmt = cutlass::gguf_packed::Fmt(PPU_PACKED_FORMAT);
#else
  static constexpr cutlass::gguf_packed::Fmt kPackedFmt = cutlass::gguf_packed::Fmt::Q4K;
#endif
  using PackedUnit = cutlass::gguf_packed::Unit<kPackedFmt>;
  // This collective has ONE weight plane. A format selector for Q3/Q5/Q6 must therefore leave every instantiation
  // here on fp16 metadata even when its low plane happens to have the same bit width as Q2/Q4. Besides preventing a
  // scale-pointer reinterpretation, the type-level match gives inactive formats a harmless legal staging type.
  static constexpr bool kPackedFormatMatchesElement =
      (kPackedFmt == cutlass::gguf_packed::Fmt::Q4K && std::is_same_v<ElementB, cutlass::int4b_t>) ||
      (kPackedFmt == cutlass::gguf_packed::Fmt::Q2K && std::is_same_v<ElementB, cutlass::uint2b_t>);
  static constexpr int kPackedScaleUnit = kPackedFormatMatchesElement ? PackedUnit::kUnitBytes : 16;
  using SmemLayoutScalePacked = Layout<Shape <Int<Scale_TileN>, Int<kPackedScaleUnit>>,
                                       Stride<Int<kPackedScaleUnit>, _1>>;
  // APPLICABILITY, NOT A REQUIREMENT. smem delta = TN*Stages*(16 - 4*Scale_TileK): strictly negative only for
  // Scale_TileK > 4, a wash at 4, and POSITIVE below it -- the unit always carries eight groups' codes while a k-tile
  // with small Scale_TileK needs fewer. Measured on ppu001 across the splitk bench's 11 units: three configs shrank by
  // exactly one zero tile (-8176, -6128, -4080 = -(8192, 6144, 4096) + 16 B of alignment) and at least one GREW (+1040).
  //
  // At gs=32, Scale_TileK == 8 is the same condition as TileK == 256, i.e. one superblock per k-tile -- so one test
  // covers both the smem win and the load cadence. A static_assert would be wrong here: the splitk bench generates 11
  // differently-shaped units into ONE binary, so a requirement fails the whole build instead of letting the other shapes
  // keep the fp16 path. Same mistake the PPU_SCALE_PREFETCH assert made earlier in this work.
  // THE FORMAT, AND EVERY CONSTANT DERIVED FROM IT. PPU_PACKED_FORMAT selects one of cutlass::gguf_packed::Fmt and
  // defaults to Q4_K, so a build that does not set it is byte-identical to what shipped. Before this the six numbers
  // below were literals -- 16 bytes per unit, Scale_TileK == 8, bias 0, has-min, ZMul 8, four 32-bit words -- and
  // schemes.py recorded the consequence: Q2_K is single-plane so the SHAPE fits and none of the constants did.
  //
  // The selected trait's unit size genuinely differs per format. Only Q4/Q2 can activate this one-plane collective;
  // a two-plane selector keeps an inert legal 16-byte staging type here and is consumed by the other collective.
  // Scale_TileK is the number of GROUPS a k-tile covers, and one unit carries a whole superblock's worth -- so the
  // applicability test is "the k-tile is exactly one superblock", which for Q4_K means 8 and for the 16-group
  // formats means 16. It was written as == 8, which is the same condition only for Q4_K.
  static constexpr bool kPackedScaleOn =
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
      kPackedFormatMatchesElement && (int(Scale_TileK) == PackedUnit::kGroups);
#else
      false;
#endif

  // THE STAGING TILE: the gguf's own bytes, cp.async'd in and decoded at the barrier. At Scale_TileK == 8 its size is
  // exactly one scale tile (TN*16 == TN*SK*2), so the channel goes from two smem tiles to three -- the price of keeping
  // the gmem side ASYNC. Without it the loader had load -> wait -> decode -> store serialised in one thread at the point
  // a cp.async used to be merely ISSUED, and that measured 24.17 us against a 20.2 baseline with an inner loop already
  // byte-identical to fp16.
  using SmemLayoutScaleRawStaged =
      Layout<Shape <Int<Scale_TileN>, Int<kPackedScaleUnit>, Int<DispatchPolicy::Stages>>,
             Stride<Int<kPackedScaleUnit>, _1, Int<Scale_TileN * kPackedScaleUnit>>>;

  // THE TRANSFER WIDTH FOLLOWS THE ACTIVE UNIT. A single uint128 per column is right for Q4; Q2's 20-byte unit is
  // five uint32 copies. Picking a width that does not divide would either drop bytes or read past the unit.
  // ppu.cp.async ACCEPTS 4, 8 OR 16 BYTES AND NOTHING ELSE (cute/arch/copy_ppu.hpp:262), so the width is the largest
  // of those that divides the unit: 16 for Q4_K and 4 for Q2_K. Q3/Q6's paired 28/36-byte transport lives in the
  // two-plane collective that owns their weight shapes, not in this file.
  static constexpr int kPackedColsPerThread = 1;
  static constexpr int kPackedCopySpan = kPackedColsPerThread * kPackedScaleUnit;
  static_assert(kPackedCopySpan % 4 == 0,
                "an active single-plane packed unit must divide into legal ppu.cp.async transfers");
  static_assert(int(Scale_TileN) % kPackedColsPerThread == 0,
                "the column tile must divide evenly among the copying threads");
  static constexpr int kPackedCopyBytes = (kPackedCopySpan % 16 == 0) ? 16
                                        : (kPackedCopySpan % 8  == 0) ? 8
                                        : (kPackedCopySpan % 4  == 0) ? 4 : 0;
  static_assert(kPackedCopyBytes != 0,
                "an active single-plane packed unit must divide into 4, 8 or 16-byte ppu.cp.async transfers");
  using PackedCopyElem =
      cute::conditional_t<kPackedCopyBytes == 16, cute::uint128_t,
      cute::conditional_t<kPackedCopyBytes == 8,  uint64_t, uint32_t>>;

  using GmemTiledCopyScalePacked = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<PackedCopyElem>, uint8_t>{},
                    Layout<Shape <Int<Scale_TileN / kPackedColsPerThread>, _1>>{},   // one thread per column GROUP
                    Layout<Shape <_1, Int<kPackedCopySpan>>>{}));    // its whole span, however many ops that takes
  // The staged view, which is what SharedStorage actually holds: stage s starts at s * TN * 16 bytes.
  using SmemLayoutScalePackedStaged =
      Layout<Shape <Int<Scale_TileN>, Int<kPackedScaleUnit>, Int<DispatchPolicy::Stages>>,
             Stride<Int<kPackedScaleUnit>, _1, Int<Scale_TileN * kPackedScaleUnit>>>;

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
  // PPU_A_CPASYNC: A's own gmem->smem copy, one row, plain cp.async instead of the AIU.
  //
  // The AIU/swzl pair cannot deliver a one-row A tile: l82 replays the read's simulator and shows the 16 rows come
  // from vreg_row_idx = (v/2)*8 + lane/4 + coord_h, the instruction's lane/vreg structure, with m16n16 in the
  // mnemonic. There is no stride operand to zero -- the asm takes only (base, coord_h, CUBE_H, channel offset) --
  // and CUBE_H scales the SLICE stride, so shrinking it aliases 360 of 512 reads instead of collapsing rows.
  //
  // A plain cp.async has none of that: cute computes the address from the layout, so a stride-0 M mode genuinely
  // aliases and one row is enough. Atom and the thread_idx % n slicing are copied from GmemTiledCopyScale, which
  // already moves a small operand this way in this same collective and rides the same cp_async_fence.
  // PPU_A_PACK geometry, all read off fold_derivation/l84-l86 rather than derived here:
  //   row 0 of a 16x64 fp16 cube occupies 4 runs of 16 halfs at half-offsets {0, 288, 528, 816}
  //   consecutive cube bases can sit 16 halfs apart with no collision (l85), the runs being 16 wide and aligned
  //   the last cube is still READ to its full 1024-half span, so that is the tail of the allocation
  static constexpr int kACubeH      = shape<0>(TileShape{});                  // CUBE_H = Block_MN = TileM for A
  static constexpr int kACubeW      = 64;                                    // AiuContElemSize for fp16
  static constexpr int kASlices     = kACubeW / 16;                          // 8 words per slice
  // PITCH 64 HALFS = 128 B, not the minimum 16. The tight pack faulted on the box in B's AIU load
  // (vmem.aiu.ld.tsm ... .b8) because smem_b FOLLOWS smem_a and cute::array_aligned only guarantees 16 B, while
  // PPU0010's AIU needs 32 -- the old alignment held only because A's byte count happened to be a multiple of 32.
  // At 64 halfs every cube base and the whole span are 128-B multiples, so the alignment is structural instead of
  // arithmetic. Costs 2944 B against 2272 at pitch 16, still 5.6x under the unpacked 16,384.
  static constexpr int kAPackPitch  = PPU_A_PACK_PITCH;                      // halfs; see PPU_A_PACK_PITCH
  static constexpr int kACubes      = shape<2>(TileShape{}) / kACubeW;       // cubes per stage = InstNum
  // Rounded up to 64 halfs so smem_b starts 128-B aligned whatever the cube geometry is.
  static constexpr int kAPackSpanRaw = kAPackPitch * (kACubes * DispatchPolicy::Stages - 1) + kACubeH * kACubeW;
  static constexpr int kAPackSpan    = ((kAPackSpanRaw + 63) / 64) * 64;
  static constexpr int kAWrThreads  = kACubes * kASlices * 2;                // cube x run x (16 halfs / 8 per thread)
  // Row 0's run offsets, derived from ppu_tsm_ld_swzl_sim rather than tabulated: row 0 means lane/4 == 0 and
  // v/2 == 0, which leaves vreg_line_idx = 0 and vreg_vec_idx = v%2, so the run starts at the slice base plus
  // swz2(v=0)*4 words and covers 8 words. A constexpr function, not a static array -- a static constexpr array
  // member has no device definition.
  CUTLASS_HOST_DEVICE static constexpr int aPackRunOff(int s) {
    int const ssv = (((s & 1) << 1) + ((s & 2) >> 1)) * 2;                    // 0, 4, 2, 6
    return 2 * (kACubeH * 8 * s + ssv * 4);                                  // halfs
  }
  // l85's collision check. Defined here, ASSERTED in mma(): a static_assert in the class body calls a member of an
  // incomplete class, which EDG accepts and hgcc rejects with "no type named 'SharedStorage'".
  CUTLASS_HOST_DEVICE static constexpr bool aPackDisjoint() {
    int const n = kACubes * DispatchPolicy::Stages;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        for (int a = 0; a < kASlices; ++a)
          for (int b = 0; b < kASlices; ++b) {
            int const x = kAPackPitch * i + aPackRunOff(a);
            int const y = kAPackPitch * j + aPackRunOff(b);
            if (x < y + 16 && y < x + 16) return false;
          }
    return true;
  }
  static constexpr int kACpElemsPerThr = 8;                                  // 128-bit, as the scale copy uses
  static constexpr int kACpThreads     = shape<2>(TileShape{}) / kACpElemsPerThr;
  using GmemTiledCopyACp = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, RealInternalElementA>{},
                    Layout<Shape<Int<kACpThreads>>>{},
                    Layout<Shape<Int<kACpElemsPerThr>>>{}));

  using RealInternalElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
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
  static_assert((size<1>(TileShape{}) % size<0>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  // A's SMEM CANNOT BE MADE STRIDE-0 IN M -- TRIED, IT FAULTS. Recorded because the arithmetic is compelling
  // and someone (me) will otherwise try it again.
  //
  // The motivation: at decode every expert has ONE row while TileM >= 16 (every MMA atom is Shape<_16,...>), so
  // 15/16 of this tile is padding whose results the epilogue's residue mask discards. SharedStorage sizes smem_a
  // by cosize_v<SmemLayoutA>, so a stride-0 M mode shrinks the allocation with no change there: 16,384 B ->
  // 1,024 B at (16,32,256) with 2 stages, block total 26,624 -> 11,264, 9 blocks/CU -> 23, 18 warps/CU -> 46,
  // theoretical occupancy 28% -> 72%.
  //
  // WHY IT FAILS, measured rather than inferred (fold_derivation/l74_swzl_coord_not_stride.cu). The mma-side read
  // is partition_S(make_mix_tensor_like(sA)), and a mix tensor carries (ptr, COORDINATE) rather than a resolved
  // pointer -- copy_unpack forwards src.data().coord_ to the asm, as the traits' own comment in
  // copy_traits_ppu0010_aiu.hpp states. Printing that coordinate for both layouts:
  //
  //     compact (16,256,2):(256,1,4096)   coord at (m,0,0) = (0,m,_0,0)   linear offsets 0 256 512 768
  //     bcast   (16,256,2):(  0,1, 256)   coord at (m,0,0) = (0,m,_0,0)   linear offsets 0   0   0   0
  //
  // The coordinates are IDENTICAL: m enters raw, unscaled by any stride. So the stride-0 layout altered exactly
  // the quantity this path does not use and left untouched the one it does, and the hardware still turned
  // coordinate m into base + m*(cube row pitch). On ppu001 that reads 16x past the 16x smaller allocation:
  //     tsm.ld.swzl.b32x4.s0.t1.trans0  vreg[64:67], [sreg63] @sreg27      (bases 512 B apart: 0xc00, 0xe00)
  // nvcc's front end accepted it with PPU_FORCE_INSTANTIATE and every static_assert passing, so the front end was
  // no evidence at all here.
  //
  // NO LAYOUT CHANGE CAN DO THIS. The fix has to pin the M COORDINATE, which is partition_S's output, not a
  // stride: copy(smem_tiled_copy_A, tCsA_p(_,0,k_block), tCrA_copy_view(_,i,k_block)) for each destination i.
  // Whether that is a saving at all depends on CPY_M = size<1>(tCrA_copy_view) exceeding 1 -- the box's three
  // tsm.ld.swzl at 512-byte-apart bases say it does, but that is an inference from three instructions.
  //
  // WHETHER IT IS WORTH DOING is now answerable, and the answer is probably not: the TileN ladder raised
  // warps/block 2 -> 8 at fixed total work and bought 1.066x within a single run (22.68 -> 21.28 us), so
  // occupancy is a weak lever for this kernel. 2.6x of theoretical occupancy would not be expected to behave
  // differently from 1.78x of it.
  // A's shared-memory tile. Its footprint is TileM x TileK x Stages and TileM >= 16 is forced by the MMA atom
  // shape, so at one row per expert 15/16 of it is padding -- but it cannot be shrunk HERE. Two box faults and one
  // silent-NaN round established why: the allocation is sized by cosize_v<SmemLayoutA>, so a stride-0 M mode does
  // collapse it, but the swzl read is addressed by COORDINATE (fold_derivation/l74) and by its cube geometry, not
  // by these strides, so the instruction keeps sourcing TileM rows and reads out of bounds. Shrinking the cube to
  // match changes the permutation rather than the footprint and delivers the right bits to the wrong registers.
  //
  // Bypassing shared memory for A was also built and measured, and lost by 1.14-1.85x: the mma fragment needs 128
  // elements per thread per k-tile, which is ~64 vector loads against this atom's InstNum = 4, and A's 16x reuse
  // is BETWEEN threads, which only shared memory can serve. It has been removed again.
  //
  // Nor is there gmem traffic left to save. AiuDesc::init takes dim_h from the problem's M -- per expert in the
  // grouped kernel -- and the instruction is ...padz..., so at one row per expert the AIU already fetches exactly
  // one row per k-tile and zero-fills the rest of the cube.
#if defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
  // Stride 0 on M: cosize_v drops from TileM*TileK*Stages to TileK*Stages, so SharedStorage allocates ONE row.
  // Legal here and not under the AIU because every A access on this path is a plain copy, which takes its address
  // from this layout. Sound only at Mmax == 1; launch() enforces that.
  using SmemLayoutA = decltype(make_layout(
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      make_stride(_0{}, _1{}, shape<2>(TileShape{}))));
  // A compact twin, never allocated, used only to shape the mma fragment: partition_fragment_A on the stride-0
  // layout would inherit the zero and allocate fewer registers than the mma reads.
  using SmemLayoutAFrag = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
#else
  using SmemLayoutA = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
#endif
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  // It is assumed that the scales and zero-points share the same smem layout
  // PPU_SCALE_PAD: break the bank period on the GROUP stride.
  //
  // The natural layout is (Scale_TileN, Scale_TileK, Stages) : (1, Scale_TileN, Scale_TileN*Scale_TileK), and at
  // Scale_TileN = 64 halfs the group stride is 128 B -- exactly 32 banks x 4 B, so consecutive groups start on the
  // SAME bank. N is contiguous inside a group, so a single group covers the banks once and is fine; the conflicts
  // come from accesses that step in the group or stage direction. acu, sz against pc: +252k conflicts on +272k scale
  // reads, about 1.02 each, and they double the channel's transactions (+504k) while shared memory sits at 28% of
  // peak -- so this is transactions times latency, and padding is free apart from the extra bytes.
  //
  // Padding by 8 halfs (16 B) shifts each group by 4 banks. The data, the gmem->smem copy and every read all go
  // through this layout, so nothing else changes.
  // PER TileN, not one constant. The conflict map depends on Scale_TileN through the mma B operand's TV layout, and
  // fold_derivation/l98 sweeps each width against the collective's own layout: Swizzle<2,3,5> takes TN=128 from 4-way
  // to 1-way but leaves TN=32 at 4-way, i.e. it was overfit to the one width l98 originally hardcoded. The table below
  // is filled from that sweep; a width with no entry keeps the plain layout rather than pretending to be fixed.
  using ScaleSwizzleT = typename detail::ScaleSwizzleFor<int(shape<0>(ScaleTileShape{}))>::type;

#if defined(PPU_SCALE_PAD) && (PPU_SCALE_PAD > 0)
  static constexpr int kScalePad = PPU_SCALE_PAD;
  using SmemLayoutScale = decltype(make_layout(
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{}),
    make_stride(_1{},
                Int<int(shape<0>(ScaleTileShape{})) + kScalePad>{},
                Int<(int(shape<0>(ScaleTileShape{})) + kScalePad) * int(shape<1>(ScaleTileShape{}))>{})));
#else
  // PPU_SCALE_SWIZZLE -- an XOR on the scale tile's address, chosen by sweeping the collective's OWN layout in
  // fold_derivation/l98 rather than derived here. Today's map is 4-way conflicted on 4 banks (l94 (2) and l98 agree);
  // Swizzle<2,3,5> takes it to 1-way on 16 banks. It moves two bits from position 8 -- inside the group field, since
  // the group stride is 128 halfs = bit 7 -- down to position 3, so it never touches the stage field and stays a
  // permutation of the allocation (cosize 2048 halfs is a power of two, which l98 checks).
  //
  // WHY A SWIZZLE AND NOT PADDING: PPU_SCALE_PAD added halfs to the group stride and LOST, because an additive pad
  // makes the address non-power-of-two and the multiply costs more than the conflict. An XOR is free.
  //
  // WHY IT IS WORTH TRYING, bounded by numbers already in TODO.md: SK_QUANT=0 prices the whole per-group scale reload
  // at 7.3%, and PPU_SCALE_PREFETCH -- which removes only the WAITING -- recovered 0.7%. So nine tenths of that
  // channel is work, not stall, and a 4-way conflict is work: four shared-pipe services for one instruction, which
  // prefetching provably cannot reach. This attacks the part prefetch left behind.
  //
  // IT MUST BE COMPOSED ON BOTH VIEWS. partition_extra_inputs builds sS with THIS layout while
  // partition_extra_mma_info builds a tensor also called sS with SmemCopyLayoutScale (n, 1, stage*Scale_TileK + g).
  // composition(Swz, L)(c) = Swz(L(c)), so the two stay equal iff they are equal unswizzled -- l98 (2) checks exactly
  // that, 0 bad over every coordinate, before and after. Swizzling one and not the other is the same class of bug as
  // the two-literals pitch that faulted with an invalid VA.
  //
  // APPLICABILITY, AND WHY IT IS A PARTIAL SPECIALISATION AND NOT conditional_t. cute requires a power-of-two
  // shape*stride to compose a swizzle, and this bench builds many units into one binary -- Stages = 3 alone makes
  // smem_scale_k = 3*Scale_TileK non-power-of-two. `conditional_t` does not help: BOTH branch types are instantiated
  // to be named, so the assert fires from the branch that was not taken. A partial specialisation instantiates only
  // the selected body. The local front-end check caught this on the first build; a static_assert here would instead
  // have failed the whole binary for the shapes that cannot carry it, which is the mistake PPU_SCALE_PREFETCH made.
  using PlainSmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{})));
  static constexpr bool kScaleSwizzleOkInner =
#if defined(PPU_SCALE_SWIZZLE) && (PPU_SCALE_SWIZZLE != 0)
      cute::is_static<PlainSmemLayoutScale>::value &&
      ((int(cute::cosize_v<PlainSmemLayoutScale>) & (int(cute::cosize_v<PlainSmemLayoutScale>) - 1)) == 0) &&
      ((int(DispatchPolicy::Stages) & (int(DispatchPolicy::Stages) - 1)) == 0) &&
      ((int(shape<0>(ScaleTileShape{})) & (int(shape<0>(ScaleTileShape{})) - 1)) == 0) &&
      ((int(shape<1>(ScaleTileShape{})) & (int(shape<1>(ScaleTileShape{})) - 1)) == 0);
#else
      false;
#endif
  using SmemLayoutScale = typename detail::MaybeScaleSwizzle<kScaleSwizzleOkInner, ScaleSwizzleT, PlainSmemLayoutScale>::type;
#endif
  // DECLARED IN BOTH BRANCHES, because partition_extra_mma_info uses them unconditionally. Putting them only in the
  // #else broke `PPU_SCALE_PAD=8` outright -- "identifier ScaleSwizzle is undefined" -- and the local front-end check
  // does not exercise that macro unless asked, so it shipped. Same shape as every other defect here: a definition and
  // its use governed by two different conditions.
  static constexpr bool kScaleSwizzleOk =
#if defined(PPU_SCALE_PAD) && (PPU_SCALE_PAD > 0)
      false;                       // padding already changes the map; the two are alternatives, not a stack
#else
      kScaleSwizzleOkInner;
#endif


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

  // PLACED HERE, AFTER KernelConversionMode, AND THAT IS NOT COSMETIC. This block first sat next to the swizzle
  // constants 70 lines above, where KernelConversionMode does not exist yet -- and because the offending conjunct
  // lives inside `#if defined(PPU_PACKED_SCALE_FUSED)`, EVERY build without the macro preprocessed it away and
  // compiled. The one configuration that used it was the only one that could not build, and nothing local covered
  // that configuration, so it reached the box as a define that quietly failed to apply. See ci/local_gates.py's
  // SYNTAX table, which now carries the macro, and l100_fused_active, which asserts the path is actually on.
  // -----------------------------------------------------------------------------------------------------------------
  // PPU_PACKED_SCALE_FUSED -- ONE INTERLEAVED (scale, zero) TILE INSTEAD OF TWO PLANES.
  //
  // WHAT IT IS FOR, and it is the STORE side only. The packed decoder writes `sS(n,G,st)` and `sZ(n,G,st)` as two
  // 16-bit stores; 32 lanes with consecutive n cover 32 adjacent 2-byte slots, which is 16 of the 32 four-byte banks
  // two deep, and stores cannot broadcast. acu measured the cost exactly: +73,728 conflicts against base, matching
  // `4 decoder warps x 8 groups x 2 planes x 9 passes x 128 CTAs` to the unit. Interleaving makes it ONE 32-bit store
  // whose 32 lanes hit all 32 banks once.
  //
  // WHY THIS ONE AND NOT THE OTHER TWO CANDIDATES, both already tried and both dead:
  //   * PAIRING ADJACENT COLUMNS to widen the store RACES. cp_async_wait is per thread, so between the wait and the
  //     publishing __syncthreads a thread may only read bytes it copied itself; the paired version read column 2p+1
  //     and rowC went to bad=128/4096, concentrated in odd columns. This is ownership-safe by construction: a thread
  //     derives BOTH halves from its OWN column's unit.
  //   * AN OFFLINE PERMUTATION of the scale tensor cannot do it at all. A reorder changes which VALUE sits at an
  //     address; a bank conflict is a property of the ADDRESSES the warp issues, which are unchanged -- and composing
  //     the permutation into the read view to keep it correct is exactly the runtime address arithmetic that made
  //     PPU_SCALE_SWIZZLE cost ~7% for zero conflicts removed.
  //
  // BYTES ARE UNCHANGED, in both memories. Stored bytes: the interleave is a pure rearrangement. Shared: scale 4 KiB
  // plus zero 4 KiB is 8 KiB either way -- the fused tile takes 2x the elements and the zero tile goes to zero, so
  // SharedStorage is the same size. Anyone expecting an occupancy gain here will not get one.
  //
  // THE READ SIDE IS DELIBERATELY UNTOUCHED. sS and sZ stay half-typed views over the fused buffer at offsets 0 and
  // 1 with every stride doubled, so all six tensors, the copy atom, the fragments and the four transform arms keep
  // their shapes and their code. The load bank map does not get worse: the current map puts 8 lanes on each of 4
  // banks in pairs; doubling the stride spreads them to 8 banks, still 4-way from the 256-element thread stride, so
  // the SERVICE count per pair of reads is what it was. Halving the read count is a SEPARATE change (one 32-bit read
  // plus a register deinterleave) and is not attempted here -- one variable at a time, and this one is the 73,728.
  static constexpr bool kFusedScaleZero =
#if defined(PPU_PACKED_SCALE_FUSED) && (PPU_PACKED_SCALE_FUSED != 0)
      kPackedScaleOn && (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) && !kScaleSwizzleOk;
#else
      false;
#endif

  // THE ASSUMPTION IS CHECKED, NOT ASSUMED. The fused layouts below are written out compactly rather than derived
  // from SmemLayoutScale, because a stride-doubling transform of an arbitrary (possibly swizzled, possibly padded)
  // layout is not a thing I can state in one line and be sure of. That is only sound while the layout it replaces IS
  // the compact one, so say so and let the build fail otherwise -- the alternative is the failure mode this file has
  // hit twice, where two functions build a tensor of the same name with different layouts and the second one faults
  // as "TSM out of range".
  static constexpr int kSZ_N   = int(shape<0>(ScaleTileShape{}));
  static constexpr int kSZ_G   = int(shape<1>(ScaleTileShape{}));
  static constexpr int kSZ_St  = int(DispatchPolicy::Stages);
  // THE GATE IS A TEMPLATE PARAMETER, NOT `!Fused || ...`. A disjunction does not stop the right-hand side from being
  // INSTANTIATED, and with PPU_SCALE_SWIZZLE on SmemLayoutScale is a ComposedLayout whose stride<> is deleted -- so the
  // first version of this check failed to compile the swizzle build while claiming to be inert there. `if constexpr` on
  // a template parameter is the only form that actually discards, which this file already says in as many words about
  // kPackedScaleOn. The local syntax gate caught it, which is the whole reason that gate exists.
  template <bool Fused, class L>
  static constexpr bool sz_layout_is_compact() {
    if constexpr (!Fused) { return true; }
    else {
      return cute::is_static<L>::value &&
             int(cute::cosize_v<L>) == kSZ_N * kSZ_G * kSZ_St &&
             int(cute::stride<0>(L{})) == 1 &&
             int(cute::stride<1>(L{})) == kSZ_N &&
             int(cute::stride<2>(L{})) == kSZ_N * kSZ_G;
    }
  }
  static_assert(sz_layout_is_compact<kFusedScaleZero, SmemLayoutScale>(),
                "PPU_PACKED_SCALE_FUSED assumes the compact (n, group, stage) scale layout");
  // Same discarding requirement for the flattened read view; see above.
  template <bool Fused, class L>
  static constexpr bool sz_copy_layout_is_compact() {
    if constexpr (!Fused) { return true; }
    else { return int(cute::stride<0>(L{})) == 1 && int(cute::stride<2>(L{})) == kSZ_N; }
  }

  // The WORD view the decoder stores through: one 32-bit slot per (n, group, stage), stride 1 in n so 32 lanes with
  // consecutive n write 32 consecutive words and touch all 32 banks once.
  using SmemLayoutScaleFusedWord = decltype(make_layout(
      make_shape(Int<kSZ_N>{}, Int<kSZ_G>{}, Int<kSZ_St>{}),
      make_stride(_1{}, Int<kSZ_N>{}, Int<kSZ_N * kSZ_G>{})));
  // The HALF views the readers keep using: same shape, every stride doubled. scale is the low half of each word and
  // zero the high half, so they differ only by the base pointer -- see scale_zero_base() below.
  using SmemLayoutScaleFusedHalf = decltype(make_layout(
      make_shape(Int<kSZ_N>{}, Int<kSZ_G>{}, Int<kSZ_St>{}),
      make_stride(_2{}, Int<2 * kSZ_N>{}, Int<2 * kSZ_N * kSZ_G>{})));
  // What every scale/zero TENSOR is built on. One name, so the six construction sites cannot disagree.
  using SmemLayoutScaleSZ = cute::conditional_t<kFusedScaleZero, SmemLayoutScaleFusedHalf, SmemLayoutScale>;

  static constexpr auto
  elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    }
    else if constexpr (ModeHasScales) {
      // FUSED TAKES BOTH PLANES' ELEMENTS AND THE ZERO TILE GOES TO ZERO, so the total is byte-identical. Written as
      // 2 * cosize of the UNFUSED layout, not cosize of the fused one: the fused layout's strides are doubled, so its
      // cosize is 2*cosize - 1 and the final zero slot would fall outside the allocation.
      return kFusedScaleZero ? 2 * int(cute::cosize_v<SmemLayoutScale>) : int(cute::cosize_v<SmemLayoutScale>);
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
      return kFusedScaleZero ? 0 : int(cute::cosize_v<SmemLayoutScale>);   // fused: zero lives in smem_scale's odd halfs
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
      assert(false);
    }
  }

  // THE ONE PLACE THAT KNOWS WHERE THE ZERO PLANE LIVES. Fused, it is the odd half of every 32-bit slot, i.e. the
  // scale buffer's base plus one element; unfused it is its own array. Three sites build a zero tensor and all three
  // go through this, because "six tensors, where I had claimed two" is the recorded way this goes wrong.
  template <class Storage>
  CUTLASS_DEVICE static NonVoidElementZero*
  zero_smem_base(Storage& storage) {
    if constexpr (kFusedScaleZero) {
      return reinterpret_cast<NonVoidElementZero*>(storage.smem_scale.begin()) + 1;
    } else {
      return storage.smem_zero.begin();
    }
  }

public:
  // OBSERVABLE ON PURPOSE. kFusedScaleZero and the layouts it selects sit in the private section because they live
  // beside KernelConversionMode, which they need. But a flag nobody outside can read is a flag that silently does
  // nothing, and that is not hypothetical here: PPU_PACKED_SCALE_FUSED shipped to the box in a state where the only
  // translation unit that used it could not compile, the define was reported as a WARNING nobody's gate checked, the
  // binary built without it, correctness passed, and acu reported the store conflicts unchanged at 81,920 (+0.00%).
  //
  // CORRECTION, and it matters because the wrong version of this sentence closed off the one probe that works.
  // This used to read "every observable the bench and the profiler have -- shared bytes, instruction counts,
  // results -- is identical whether this path is on or off, BY DESIGN". Shared BYTES and RESULTS are identical, and
  // the read side is untouched. The STORE INSTRUCTION COUNT IS NOT: the publication below emits ONE uint32_t
  // assignment where the unfused branch emits two half assignments, so at this launch's
  //     4 warps x 8 groups x 9 publications x 128 CTAs = 36,864
  // shared-store instructions disappear. acu's `Shared Store / Inst` is therefore a valid machine-code probe --
  // ~84,480 -> ~47,616 -- and reading only `Bank Conflicts` is what made the previous round look undecidable.
  // (A backend is still free to lower the 32-bit store into two 16-bit ones. That is exactly what the Inst count
  // detects, and it is why the count is the probe rather than the source.)
  //
  // What the type-level gate is still for: dev/fold_derivation/l100_fused_active.cu answers "is this path selected
  // for THIS configuration" without a device, which no counter can do, and it is what proves a null result is a
  // real null rather than an inactive path.
  static constexpr bool is_fused_scale_zero = kFusedScaleZero;
  static constexpr bool is_packed_scale     = kPackedScaleOn;
  static constexpr bool has_zero_channel    = (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero);
  using FusedScaleWordLayout = SmemLayoutScaleFusedWord;   // 32-bit slots, stride 1 in n: the conflict-free store
  using FusedScaleHalfLayout = SmemLayoutScaleSZ;          // what every reader sees; stride 2 in n when fused

  struct SharedStorage
  {
    static constexpr int scale_elements = elements_per_smem_scale();
    static constexpr int zero_elements = elements_per_smem_zero();
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
    // Packed: the cubes overlap, so the allocation is the packed span, not cosize of the logical tile.
    cute::ArrayEngine<RealInternalElementA, kAPackSpan> smem_a;
#else
    cute::ArrayEngine<RealInternalElementA, cute::cosize_v<SmemLayoutA>> smem_a;
#endif
    // If this member is ever resized or removed, note that smem_b follows it directly and array_aligned defaults
    // to 16-B alignment: at the full size (cosize*2 B, a multiple of 32) smem_b happens to land 32-B aligned,
    // which PPU0010's AIU load requires (align_bytes = 32 in gemm_operands.hpp). Shrinking smem_a to one element
    // once put smem_b at offset 2 and produced 'AIU_ld TSM size out of range'. The alignment holds by arithmetic,
    // not by declaration.
    cute::ArrayEngine<RealInternalElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    // PACKED SCALE CHANNEL (plan #20 option E). When on, the tile holds the gguf's own bytes -- one 16 B unit per
    // (superblock, column) carrying d, dmin and the codes -- so the ZERO TILE GOES TO ZERO ELEMENTS, not shrinks: `mn`
    // lives in the same unit. Chosen by TYPE rather than by #if, because one binary holds units of several shapes and
    // only those with Scale_TileK == 8 take this path (see kPackedScaleOn).
    //
    // smem_zero stays declared, at zero elements, rather than being deleted: it is the LAST member, so a zero-length
    // ArrayEngine cannot move anything (unlike smem_a, whose comment above records what shrinking a leading member did
    // to smem_b's 32 B alignment), and keeping the name lets the ScaleZero paths compile unchanged.
    // F CHANGES NOTHING HERE. The decode happens on the way IN (see packed_decode_stage), so smem still holds the
    // same two fp16 planes and the whole read side -- s2r, fragments, the four transform arms -- is untouched. That is
    // the point: llama.cpp's MMQ keeps the shared read and amortises the decode over its consumers, and the earlier
    // register-decode attempt did the opposite (measured 1.4M extra ALU to save 0.27M shared loads, with LSU only 6%
    // busy and IALU/FALU at 14%).
    cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
    cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    // LAST on purpose: at zero elements it cannot move anything above it, so a default build stays byte-identical.
    cute::ArrayEngine<uint8_t, kPackedScaleOn ? int(cute::cosize(SmemLayoutScaleRawStaged{})) : 0> smem_scale_raw;
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
    // The raw-byte cp.async beside the fp16 one; stateless, so carrying both costs nothing.
    GmemTiledCopyScalePacked gmem_tiled_copy_scale_packed;
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
  // THE PACKED COPY NEEDS ITS OWN N PREDICATE. scale_valid is derived from the fp16 copy's coordinate, whose
  // thread -> N map is 8*(p % 16) for a (16,8) x (_8,_1) layout, while the packed copy is partitioned ONE COLUMN PER
  // THREAD. Using one for the other is only invisible because every shape measured so far has N a multiple of TileN.
  // With a residue it fails both ways at once: at residue_n = 20 thread 3 owns packed column 3 (valid) but its fp16
  // coordinate is 24, so its cp.async is skipped while the decode loop still reads that column -- bytes nobody
  // copied; and thread 32 has fp16 coordinate 0, so it copies packed column 32 out of a tile that has twenty, an
  // out-of-bounds GLOBAL read. Neither depends on PPU_PACKED_SPLIT_GROUPS; the split only makes the first one a
  // stated invariant that is false rather than an accident that happens to hold.
  bool scale_valid_pk = true;
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
      // THE SAME TYPE THE MEMBER IS DECLARED AS, not a second construction of it. This line spelled the atom out as
      // uint128 while GmemTiledCopyScalePacked derived it from the unit, so the moment the unit stopped being 16
      // bytes the two disagreed -- and the failure surfaced as "TiledCopy uses too few vals" pointing at a copy that
      // looked correct where it was declared. One relation, one place; the member's own type is that place.
      p.gmem_tiled_copy_scale_packed = GmemTiledCopyScalePacked{};
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
#if (defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)) || (defined(PPU_A_PACK) && (PPU_A_PACK != 0))
    // PLAIN, not make_mix_tensor_like: that wrapper carries (ptr, coordinate) for the AIU descriptor and has NO
    // addressable strides (l74), so &gA(...) yields a meaningless address. Both macros write A with cp.async and
    // therefore need real strides -- PPU_A_PACK was missing from this condition and faulted at
    // s.wait commit_group(1), the async copy's own address check.
    Tensor gA = local_tile(mA_mkl(_,_,0), TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});            // (BLK_M,BLK_K,k)
#else
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,0));                                                         // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});                    // (BLK_M,BLK_K,k)
#endif

    // B init (include init aiu desc)
    auto mB_nk = load_init_B(mainloop_params, N, K, L, l_coord);                                                // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});                    // (BLK_N,BLK_K,k)

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

      // THE PACKED PLANE: [nsb][N][16] bytes, i.e. one 16 B unit per (superblock, column) holding d, dmin and the codes.
      // nsb is DERIVED, not a new parameter: at Scale_TileK groups per k-tile, scale_k / Scale_TileK is the superblock
      // count. Built unconditionally and appended, because `auto` return-type deduction sees BOTH branches of an
      // `if constexpr` whose condition is a class constant, so a macro-dependent tuple type would not compile.
      int const nsb_ = scale_k / int(Scale_TileK);
      Tensor mSp = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t const*>(mainloop_params.ptr_S)),
                               make_shape (N, Int<kPackedScaleUnit>{}, nsb_, L),
                               make_stride(Int<kPackedScaleUnit>{}, _1{},
                                           Int<kPackedScaleUnit>{} * N, Int<kPackedScaleUnit>{} * N * nsb_));
      Tensor gSp = local_tile(mSp(_,_,_,l_coord), Shape<Int<Scale_TileN>, Int<kPackedScaleUnit>>{},
                              make_coord(n_coord, 0, _));                                    // (BLK_N, 16, nsb)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA, gB, gS, gSp);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_Z), make_shape(N,scale_k,L));    // (n,scale_k,l)
        Tensor mZ_nk = mZ_nkl(_,_,l_coord);
        Tensor gZ = local_tile(mZ_nk, ScaleTileShape{}, make_coord(n_coord, _));
        return cute::make_tuple(gA, gB, gS, gZ, gSp);
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
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
    // l85's collision check, as a body-level assert. It CANNOT sit in the class body: it calls a member of the
    // same class, which is still incomplete there -- nvcc's EDG front end accepts that and hgcc rejects it with
    // "no type named 'SharedStorage'", which is how the local gate passed and the box build failed.
    static_assert(aPackDisjoint(), "PPU_A_PACK: packed row-0 runs collide -- raise kAPackPitch");
    static_assert(kACubeW == 64, "PPU_A_PACK: run offsets assume AiuContElemSize == 64 halfs");
    // The read's pitch and the write's now come from the same macro (PPU_A_PACK_PITCH), so they cannot diverge
    // the way they did when each side carried its own literal.
    static_assert(int(cute::size<0>(TileShape{})) == kACubeH, "PPU_A_PACK: CUBE_H must be TileM");
#endif
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
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

#if defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
    // ROW 0 ONLY. sA's M stride is 0, so row 0 IS the single physical row and every other m index aliases onto it.
    // thread_idx % kACpThreads mirrors the scale copy's slicing; the guard keeps the surplus threads out instead of
    // having them re-issue the same cp.async, which is what the scale path settles for.
    auto a_cp_thr = GmemTiledCopyACp{}.get_slice(thread_idx % kACpThreads);
    Tensor tAgA = a_cp_thr.partition_S(gA(Int<0>{},_,_));                      // (ACPY,ACPY_K,k)
    Tensor tAsA = a_cp_thr.partition_D(sA(Int<0>{},_,_));                      // (ACPY,ACPY_K,PIPE)
#else
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
#endif
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      auto k_iter_crd = cute::idx2crd(*k_tile_iter, k_iter_shape);
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
      copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,k_pipe), warp_idx);
      copy_A_packed_row0(gA, storage.smem_a.begin(), *k_tile_iter, k_pipe, thread_idx);
#elif defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
      copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,k_pipe), warp_idx);
      if (thread_idx < kACpThreads) copy(GmemTiledCopyACp{}, tAgA(_,_,*k_tile_iter), tAsA(_,_,k_pipe));
#else
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,k_pipe),
        warp_idx
      );
#endif
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
#if defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
    // Shape the fragment from the COMPACT twin: sA's M stride is 0, and partition_fragment_A would inherit it and
    // allocate fewer registers than the mma reads (measured on the gmem variant: ((2,2,2),1):((1,2,0),0), cosize 4
    // instead of 8). Null pointer -- layout only.
    Tensor sA_frag = make_tensor(make_smem_ptr(static_cast<RealInternalElementA*>(nullptr)), SmemLayoutAFrag{});
    Tensor tCrA     = thr_mma.partition_fragment_A(sA_frag(_,_,0));            // (MMA,MMA_M,MMA_K)
#else
    Tensor tCrA     = thr_mma.partition_fragment_A(sA(_,_,0));                // (MMA,MMA_M,MMA_K)
#endif
    Tensor tCrB_mma = thr_mma.partition_fragment_B(sB(_,_,0));                // (MMA,MMA_N,MMA_K)
#if defined(PPU_B_DEQUANT_NOP) && (PPU_B_DEQUANT_NOP != 0)
    // The ablation must not change what the MMA pipe is fed. partition_fragment_B does not initialise, and with the
    // conversion removed nothing else would either, so every atom would consume indeterminate bits -- which as fp16
    // are freely NaN or Inf, and a timing measurement taken over exceptional operands measures the exception
    // handling. One fill, outside the k-loop, so it costs nothing the measurement cares about.
    cute::fill(tCrB_mma, static_cast<typename decltype(tCrB_mma)::value_type>(1.0f));
#endif

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

#if defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
    // Plain vectorised copy, addressed from sA's layout -- so the stride-0 M mode aliases and every m slot of the
    // fragment gets row 0. Sliced per thread (not per warp-group like the AIU atom) because this one is a normal
    // load, and partition_S takes the PLAIN sA, not a mix tensor.
    auto smem_tiled_copy_A = make_tiled_copy_A(
        Copy_Atom<DefaultCopy, RealInternalElementA>{}, tiled_mma);   // DefaultCopy: width from the layouts, no
                                                                     // alignment assumed -- a per-thread offset
                                                                     // that is not 16-B aligned would fault under
                                                                     // AssumedAlignment<128>.
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);                                     // (CPY,CPY_M,CPY_K,PIPE)
#else
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                // (CPY,CPY_M,CPY_K,PIPE)
#endif
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
#if defined(PPU_SCALE_PREFETCH) && (PPU_SCALE_PREFETCH != 0)
    // GROUP-AHEAD SCALE PREFETCH. On the FINE path the scale and zero are reloaded from smem at each group's first
    // mma atom and used one or two instructions later -- 8 times per k-tile at gs=32 with TileK=256. With 14.2 warps
    // per CU and no spare (this shape is work-bound), every one of those is a Memory Dependency stall that costs
    // time directly, and Memory Dependency is the top warp state at 0.98.
    //
    // Priced by removing the channel entirely (SK_QUANT on the bench): per-group reload = 7.3% of the kernel, which
    // is this change's ceiling. Dropping the zero as well is another 11.5%, but that is a format question (#20), not
    // a scheduling one.
    //
    // A second register set lets a copy step issue BOTH its groups' loads up front. Built here, not in
    // partition_extra_mma_info, because that helper is shared with the fold and 2plane collectives; retile_D is only
    // a call, so a local pair costs nothing but registers. Passed as ONE named tuple parameter -- appending to
    // transform_B_kblock's cute::tuple<Ts...> does not deduce.
    // Only the WithZero tuple has a get<3>; on the ScaleOnly path that index is out of range, so the pack is built
    // inside an if constexpr and the other modes get an empty tuple.
    auto scale_pf = [&] {
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        auto thr_pf = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma).get_thread_slice(thread_idx);
        auto s_pf = cute::make_fragment_like(cute::get<1>(partitioned_extra_info));
        auto z_pf = cute::make_fragment_like(cute::get<3>(partitioned_extra_info));
        return cute::make_tuple(s_pf, z_pf, thr_pf.retile_D(s_pf), thr_pf.retile_D(z_pf));
      } else {
        return cute::tuple<>{};
      }
    }();
#else
    auto scale_pf = cute::tuple<>{};
#endif

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

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      // The staged bytes for smem_pipe_read have landed and the fp16 planes are still private; the __syncthreads below
      // publishes them. No barrier of its own.
      packed_decode_stage<kPackedScaleOn>(storage, smem_pipe_read, thread_idx, scale_residue_n);
      __syncthreads();
      // Prefetch the first rmem from the first k-tile
      copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read);
      // NO M-PINNING LOOP HERE, and that is a measured decision. CPY_M = size<1>(tCsA) is 1 both with and without
    // PPU_A_CUBE_H (fold_derivation/l77), so M does not live on mode 1 and a loop over it is a no-op. With
    // CUBE_H=1 cute instead moves mode 2 from basis 2 to basis 0 with stride 64 and halves the A register
    // fragment (ArrayEngine 128 -> 64, with a stride-0 component), i.e. it re-derives the geometry itself.
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB_mma, partitioned_extra_info, Int<0>{}, K_ATOM_PER_COPY,
          copy_partitions_extra_info, smem_pipe_read, scale_pf);
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
          packed_decode_stage<kPackedScaleOn>(storage, smem_pipe_read, thread_idx, scale_residue_n);
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, k_block_next, smem_pipe_read);
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB_mma, partitioned_extra_info, k_block_next, K_ATOM_PER_COPY,
          copy_partitions_extra_info, smem_pipe_read, scale_pf);

        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          auto k_iter_crd = cute::idx2crd(*k_tile_iter, k_iter_shape);
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
          copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,smem_pipe_write), warp_idx);
          copy_A_packed_row0(gA, storage.smem_a.begin(), *k_tile_iter, smem_pipe_write, thread_idx);
#elif defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
          copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,smem_pipe_write), warp_idx);
          if (thread_idx < kACpThreads)
            copy(GmemTiledCopyACp{}, tAgA(_,_,*k_tile_iter), tAsA(_,_,smem_pipe_write));
#else
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
#endif
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

        CUTLASS_PRAGMA_UNROLL
        for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
          auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
          // Transform before compute
          cute::transform(tCrA(_,_,atom_idx), TransformA{});
          cute::transform(tCrB_mma(_,_,atom_idx), TransformB{});
          // gemm for one tiled_mma atom on K
          cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB_mma(_,_,atom_idx), accum);
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
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B),
        make_shape(N, make_shape(kCon, K / kCon), L),
        // GROUPED FIX: interleaved desc base = (uint8_t*)raw_pointer_cast(mB_nk.data()) treats the packed
        // ELEMENT L-stride as a BYTE count, so the stride must be in bytes: one expert weight is
        // N*K * sizeof_bits<B>/8 bytes (int4 -> N*K/2, int2 -> N*K/4, int8 -> N*K). (Was Int<0> -> every expert
        // read plane 0; wrong scale -> e>=2 read OOB garbage. This lands each expert exactly. No effect on L=1.)
        make_stride(kCon, make_stride(cute::Int<1>{}, kCon * N), int64_t(N) * int64_t(K) * sizeof_bits<RealInternalElementB>::value / 8)
      );
      Tensor mB_nk = mB_nkl(_,_,l_coord);
      auto layout_counting = make_layout(
        mB_nk.shape(),
        make_stride(ScaledBasis<_1, 1>{}, make_stride(ScaledBasis<_1, 0>{}, ScaledBasis<int, 1>{N}))
      );
      Tensor mB_nk_counting = make_counting_tensor(layout_counting);
      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            (uint8_t*)(raw_pointer_cast(mB_nk.data())), N * K / kCon, kCon, mB_nk.stride());
      return mB_nk_counting;
    } else {
      Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B), make_shape(N,K,L), mainloop_params.dB);
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
      // The packed pair sits AFTER whatever the mode already appended -- 3,4 for ScaleOnly and 5,6 for ScaleZero. Named
      // once here so the two copy sites below cannot disagree about the index, which is the shape of bug that has cost
      // this work the most time.
      // (tSgSp, tSsSp): 3,4 for ScaleOnly and 5,6 for ScaleZero. Named ONCE so the two sites cannot disagree.
      static constexpr int kPkG = (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) ? 5 : 3;
      // per-column path
      if constexpr(DispatchPolicy::StaticGroupSize == -1) {
        // Packed: ONE cp.async of the gguf's own bytes into staging. The decode is at the barrier, in mma().
        if constexpr (kPackedScaleOn)
          copy(mainloop_params.gmem_tiled_copy_scale_packed,
               get<kPkG>(extra_input_partitions)(_,_,_,0), get<kPkG+1>(extra_input_partitions)(_,_,_,write_stage));
        else
          copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,0), tSsS(_,_,_,write_stage));
        // NOT under kPackedScaleOn: `mn` rides in the scale unit, and smem_zero is zero elements there, so issuing
        // this copy would write past the end of the allocation. Found by asking what the ScaleZero fixture would do,
        // not by a compiler -- a 0-length ArrayEngine is a valid pointer and cp.async would happily scribble past it.
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero && !kPackedScaleOn) {
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
        // kPackedScaleOn picks the predicate that belongs to the copy actually being issued.
        if ((kPackedScaleOn ? scale_valid_pk : scale_valid) && (scale_load_k * Scale_TileK < scale_residue_k)) {
          if constexpr (kPackedScaleOn)
            // scale_load_k IS ALREADY A TILE INDEX (partition_S leaves the last mode selecting which block of
            // Scale_TileK groups a call loads), and one k-tile is one superblock, so it indexes superblocks directly and
            // must NOT be divided. The k bound above already encoded that reading.
            copy(mainloop_params.gmem_tiled_copy_scale_packed,
                 get<kPkG>(extra_input_partitions)(_,_,_,scale_load_k),
                 get<kPkG+1>(extra_input_partitions)(_,_,_,write_stage));
          else
            copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,scale_load_k), tSsS(_,_,_,write_stage));
          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero && !kPackedScaleOn) {
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
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScaleSZ{});
      Tensor gS = get<2>(load_inputs);
      // Construct identity layout for sS
      constexpr static Tensor cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));

      auto gmem_thr_copy_scale = mainloop_params.gmem_tiled_copy_scale.get_slice(thread_idx);

      Tensor tSgS = gmem_thr_copy_scale.partition_S(gS);
      Tensor tSsS = gmem_thr_copy_scale.partition_D(sS);
      Tensor tScS = gmem_thr_copy_scale.partition_S(cS);

      // THE PACKED g2s, partitioned beside the fp16 one and appended to the tuple for the same return-type reason as
      // gSp. One thread per column, its whole 16 B unit: l94 (6) measured this as a fully coalesced 2048 B burst.
      static constexpr int kPackedLoadIdx =
          (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) ? 4 : 3;
      Tensor sSraw = make_tensor(make_smem_ptr(shared_tensors.smem_scale_raw.begin()), SmemLayoutScaleRawStaged{});
      // THE MODULO IS NOT COSMETIC. cute does NOT wrap an out-of-range thread index, and this copy's thread layout is
      // Scale_TileN wide (128) while get_slice is called by all size(TiledMma) = 256 threads. fold_derivation/l97
      // measured where the extras land on the real layout: thread 128 partitions at byte 2048 of a 4096-byte staging
      // tile whose stage stride is Scale_TileN*16 = 2048 -- i.e. EXACTLY stage 1's first byte -- and thread 255 at
      // 4080. So while stage 0 is being copied, half the CTA writes over stage 1; while stage 1 is being copied, the
      // same half writes PAST the whole allocation, and smem_scale_raw is the last member of SharedStorage.
      //
      // That is an unconditional out-of-bounds write, and its damage depends on whether the clobbered stage is read
      // before the real copy overwrites it -- which is what an intermittent, partial, build-sensitive failure looks
      // like. rowC went bad=128, bad=724, then MATCH with no semantic source change, and restoring "=r" did not bring
      // it back, so the constraint was never the cause.
      //
      // `% n` rather than a guard, because it is the idiom this file already uses for exactly this
      // (GmemTiledCopyACp at the A cp.async path): the extra threads then redundantly copy the same bytes from the
      // same source to the same destination, which is benign, instead of partitioning outside the tile.
      auto gmem_thr_copy_raw = mainloop_params.gmem_tiled_copy_scale_packed.get_slice(
          thread_idx % int(cute::size(GmemTiledCopyScalePacked{})));
      Tensor tSgSp = gmem_thr_copy_raw.partition_S(cute::get<kPackedLoadIdx>(load_inputs));
      Tensor tSsSp = gmem_thr_copy_raw.partition_D(sSraw);

      clear(tSsS);      // both paths: the loader also relies on out-of-range columns staying zero

      // THE K BOUND. The fp16 path counts GROUPS: scale_k - <this thread's first group>. The packed path consumes one
      // UNIT per k-tile, so the bound is the superblock count -- and it is written as nsb * Scale_TileK precisely so the
      // use site (`scale_load_k * Scale_TileK < scale_residue_k`) needs no change: dividing both sides by Scale_TileK
      // leaves `scale_load_k < nsb`. The N bound is unchanged; mode 0 is still the column in both layouts, while mode 1
      // is BYTES here and groups there, which is why only this line moves.
      if constexpr (kPackedScaleOn)
        scale_residue_k = int64_t(mainloop_params.scale_k / int(Scale_TileK)) * int64_t(Scale_TileK);
      else
        scale_residue_k = mainloop_params.scale_k - get<1>(tScS(0,0,0));
      scale_valid = get<0>(tScS(0,0,0)) < scale_residue_n;
      // From the PACKED partition's own identity tensor rather than from thread_idx arithmetic: the guard and the
      // partition it guards then come from one object, which is the only form of this that cannot drift.
      if constexpr (kPackedScaleOn) {
        Tensor cSp   = make_identity_tensor(make_shape(Int<Scale_TileN>{}, Int<kPackedScaleUnit>{}));
        Tensor tScSp = gmem_thr_copy_raw.partition_S(cSp);
        scale_valid_pk = get<0>(tScSp(0,0,0)) < scale_residue_n;
      }

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS, tScS, tSgSp, tSsSp);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ  = make_tensor(make_smem_ptr(zero_smem_base(shared_tensors)), SmemLayoutScaleSZ{});
        Tensor gZ = get<3>(load_inputs);

        auto gmem_thr_copy_zero = mainloop_params.gmem_tiled_copy_zero.get_slice(thread_idx);

        Tensor tZgZ = gmem_thr_copy_zero.partition_S(gZ);
        Tensor tZsZ = gmem_thr_copy_zero.partition_D(sZ);
        // Same reason as the copies: with the packed path on, smem_zero holds zero elements.
        if constexpr (!kPackedScaleOn) clear(tZsZ);

        return cute::make_tuple(tSgS, tSsS, tScS, tZgZ, tZsZ, tSgSp, tSsSp);
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
          make_tensor(make_smem_ptr((NonVoidElementScale*)nullptr), SmemLayoutScaleSZ{})))>;
    } else {
      return 0;
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // THE PACKED SCALE CHANNEL'S DEVICE STEPS. Defined unconditionally: they are templates, so an off unit never
  // instantiates them, and the call sites select with `if constexpr (kPackedScaleOn)` per UNIT rather than per binary.
  // THE PACKED SCALE CHANNEL'S TWO DEVICE STEPS. Derived facts, all from fold_derivation/l94 on the collective's own
  // objects (l95 proves the probe's mma IS this one):
  //   * a lane's scale fragment touches exactly kPackedSlots DISTINCT columns -- 2 at TN=128/w16x16, the same for all
  //     256 lanes -- so the lane holds that many 16 B units and nothing more.
  //   * value -> slot is periodic with period 8 and is `(v >> 2) & 1` at this config; expressed here as a division by
  //     the run length rather than a stored table, and CHECKED at runtime against the coordinate tensor below, because
  //     the run length is the one number that changes with the warp shape.
  //   * every value sharing a slot shares its column, hence its scale: per group a lane decodes kPackedSlots values,
  //     not one per fragment element.
  static constexpr int kPackedSlots = 2;
  // Q4_K for now: unsigned codes with no centre, and a min channel. Q3_K would be <32, false> and Q6_K
  // <0, false>; they are template parameters of group_of precisely so no second decode is written.
  static constexpr int  kPackedScaleBias = PackedUnit::kScaleBias;
  static constexpr bool kPackedHasMin    = PackedUnit::kHasMin;
  // 8 cancels the int4 converter's own -8, which this path leaves in place: see group_of's comment for why that is the
  // better of the two ways to reconcile them.
  // ZMul CANCELS THE WEIGHT CONVERTER'S OWN SHIFT, so it follows the weight's element width -- and writing it as a
  // literal 8 was wrong for exactly the reason I gave for keeping it one. The int4 converter emits q-8, so int4
  // needs 8; the uint2 converter emits q in [0,3] with NO bias ("the per-group affine 'zero' term absorbs the
  // offset", fast_numeric_conversion_for_mix_gemm.h at the W2A16 specialisation), so a 2-bit weight needs 0.
  // Q2_K's weights are 2-bit, so the literal would have shifted every one of them by 8*scale.
  static constexpr int  kPackedZMul      =
      (cute::sizeof_bits_v<RealInternalElementB> == 4) ? 8 : 0;
  // ROUNDED UP. Q3_K's unit is 14 bytes and Q6_K's 18, i.e. 3.5 and 4.5 words, and truncating loses the tail --
  // which for Q3_K is groups 12..15 and for Q6_K the last two scales, read as zero. The staging tile is padded to
  // whole words for the same reason, so reading the extra bytes is in-bounds.
  static constexpr int kPackedUnitWords = (kPackedScaleUnit + 3) / 4;

  // THE PACKED (f16x2) PER-GROUP DECODE, which needs both fields to pack against each other and the 6-bit unsigned
  // extraction code_pair_from_words performs. Q4_K is the only format with a min, so this is exactly Q4_K today; the
  // scalar group_of_words stays for everything else and is what the `else` arm below still calls. The bias/mask/OR
  // identity it rests on is NOT restricted to unsigned codes -- l96 (A0) pinned its true bound at [-128, 895] -- so a
  // future signed format only needs its own extraction, not its own arithmetic.
  //
  // PPU_PACKED_PAIR=0 FORCES THE SCALAR DECODE BACK, and it exists to BISECT rather than to tune. rowC of
  // test_q4k_packed_gemm -- the only row where kPackedScaleOn is true -- regressed when the packed pair landed, and
  // the two candidate causes cannot be separated by reading: (i) the f16x2 asm, which has zero local coverage because
  // the local gate compiles under nvcc where the scalar fallback is selected, and (ii) anything else the same commits
  // touched. One build each answers it: PAIR=0 restoring MATCH indicts the packed arithmetic/asm, PAIR=0 still failing
  // exonerates it and points at the rest of the change.
#if defined(PPU_PACKED_PAIR) && (PPU_PACKED_PAIR == 0)
  static constexpr bool kPackedPairFast = false;
#else
  // AND SIX-BIT FIELDS. The pair path folds the field width into kMagic1152x2, so Q2_K (4 bits) and Q6_K
  // (8 bits) must take the scalar arm -- which they would otherwise enter, since Q2_K has a min and a zero
  // bias and satisfies the old condition exactly.
  static constexpr bool kPackedPairFast = kPackedHasMin && (kPackedScaleBias == 0)
                                      && (PackedUnit::kScaleBits == 6) && (PackedUnit::kMinBits == 6);
#endif


  // ON IS A TEMPLATE PARAMETER, and that is the whole point: `if constexpr` only skips INSTANTIATING a discarded branch
  // when its condition is value-dependent. With kPackedScaleOn (a class constant) both branches were instantiated, so
  // every unit paid for the packed tensors even with the path off -- measured as ~5% on the TK=64 control rows, which are
  // supposed to be byte-identical between the two builds. Routing the gate through a template parameter makes the
  // discarding real.

  // THE DECODING LOADER: gmem native unit -> registers -> the SAME fp16 smem planes the fp16 path uses.
  //
  // This is llama.cpp's shape (ggml-cuda/mmq.cuh, load_tiles_q4_K) and the reason it is right is amortisation, not
  // instruction count in isolation: there the decode runs once per (row, group) for the whole CTA and the result goes to
  // SHARED memory, so every lane that needs it reads it rather than re-deriving it. Doing it per lane in registers, which
  // is what the previous attempt did, multiplies the work by the number of lanes sharing a column (four, per l94 (7))
  // times the warps -- measured as ~1.4M extra ALU against 0.27M shared loads saved, and LSU was only 6% busy.
  //
  // Here: one thread per column reads its 16 B unit (d, dmin and all 8 groups' 6-bit codes), decodes with the bit
  // positions as compile-time constants, and stores the fp16 scale and zero. 64 columns x 8 groups = 512 decodes per CTA
  // per k-tile, against 2048 for the per-lane version.
  //
  // WHAT IT COSTS: this channel is no longer cp.async, because arithmetic between gmem and smem is exactly what cp.async
  // cannot do. The load is issued where the cp.async used to be and the barrier that already guards the stage
  // (cp_async_wait + __syncthreads at the last k_block) still guards it, so no new sync is added -- but the LDG's latency
  // is now exposed rather than overlapped, and only measurement says whether the B cp.async in flight covers it.
  // DECODE ONE STAGE, smem -> smem, at the barrier the pipeline already has.
  //
  // llama.cpp's shape (ggml-cuda/mmq.cuh, load_tiles_q4_K): the decode runs once per (column, group) for the whole CTA
  // and the result goes to SHARED memory, so every lane reads it instead of re-deriving it. The per-lane register version
  // did the opposite and paid ~1.4M extra ALU to save 0.27M shared loads, with LSU at 6% busy.
  //
  // Called AFTER cp_async_wait (the staged bytes have landed) and BEFORE __syncthreads (the planes are still private, and
  // the sync publishes them), so it adds no barrier of its own. mma() already holds shared_tensors and thread_idx, which
  // is why this needs no plumbing through any tuple.
  template <bool On, class Storage>
  CUTLASS_DEVICE static void
  packed_decode_stage(Storage& storage, int stage, int thread_idx, int64_t residue_n) {
    if constexpr (On) {
      constexpr int kTN      = int(Scale_TileN);
      constexpr int kGrp     = int(Scale_TileK);
      constexpr int kThreads = int(cute::size(TiledMma{}));
      Tensor sRaw = make_tensor(make_smem_ptr(storage.smem_scale_raw.begin()), SmemLayoutScaleRawStaged{});
      Tensor sS   = make_tensor(make_smem_ptr(storage.smem_scale.begin()),  SmemLayoutScaleSZ{});
      Tensor sZ   = make_tensor(make_smem_ptr(zero_smem_base(storage)),     SmemLayoutScaleSZ{});
      // THE FUSED STORE'S OWN VIEW: 32-bit slots, stride 1 in n. Built beside sS/sZ rather than instead of them so
      // the unfused arm below is byte-identical and the two cannot drift apart.
      Tensor sSZw = make_tensor(make_smem_ptr(reinterpret_cast<uint32_t*>(storage.smem_scale.begin())),
                                SmemLayoutScaleFusedWord{});
      (void)sSZw;
      // ONE THREAD PER COLUMN, AND THAT IS A CORRECTNESS CONSTRAINT, NOT A CHOICE. cp_async_wait is PER THREAD and the
      // __syncthreads that publishes this stage comes AFTER this function, so between the wait and the sync a thread
      // may only read bytes IT ITSELF copied. GmemTiledCopyScalePacked's thread layout is
      // Layout<Shape<Int<Scale_TileN>,_1>> with a 16 B value layout -- thread t copies column t -- so thread t reading
      // column t is exactly the guarantee the wait gives, and reading any other column is a race.
      //
      // MEASURED, NOT ARGUED: a version of this loop paired two adjacent columns per thread to turn the 2 B store into
      // a 4 B one (worth ~0.6%: acu put the scalar form at +71,680 tsm.st with +73,728 bank conflicts and 1.83
      // transactions per instruction, since 32 lanes x 2 B covers 16 of 32 banks). It read column 2p+1, which thread
      // 2p+1 copied, and test_q4k_packed_gemm's rowC -- the ONLY row where kPackedScaleOn is true -- went to
      // bad=128/4096 with the failures concentrated in odd columns. rowA and rowB kept passing because at TK=128 they
      // are on the fp16 path, which is exactly why the gate has a row per Scale_TileK.
      //
      // To get the wide store back, the COPY must be paired too (thread layout Scale_TileN/2 over a 32 B value layout)
      // so the two maps coincide again; the alternative, a __syncthreads before the decode, costs more than the store
      // saves (8 barriers per CTA against 0.6%).
      //
      // SPLIT THE GROUPS, NOT THE COLUMNS -- PPU_PACKED_SPLIT_GROUPS. Half the CTA is idle here: only threads
      // [0, Scale_TileN) enter the loop, so four warps decode while four wait at the __syncthreads below, and the
      // slowest decoding warp sets the publication latency for all eight.
      //
      // The fix falls out of a fact that was sitting in the call path unnoticed. mma() wraps the index before
      // partitioning -- `thread_idx % (Scale_GmemCopyThrLayoutH * Scale_GmemCopyThrLayoutW)` at the
      // partition_extra_inputs call -- so threads n and n + Scale_TileN are DUPLICATE OWNERS of the same column:
      // both issue its cp.async, to the same destination, and each waits on a copy it issued itself. The ownership
      // invariant above therefore permits BOTH of them to read that unit, which the paired-column attempt did not.
      //
      // So thread t decodes column `t % Scale_TileN`, taking the low half of the groups if t < Scale_TileN and the
      // high half otherwise. Eight warps decode four groups each instead of four warps decoding eight. Identical
      // decode count, identical stores, identical bank conflicts, identical ownership; the only new cost is that the
      // 16 B unit is read twice per column instead of once.
      //
      // It is also the cleanest available test of placement-versus-volume: if halving the critical path helps, the
      // barrier placement is what costs; if it does not, aggregate issue demand is, and no amount of rebalancing will
      // reach it. Off by default so the configuration everything has been measured against does not move silently.
      constexpr int kWrapTh = int(Scale_GmemCopyThrLayoutH{} * Scale_GmemCopyThrLayoutW{});
      constexpr int kOwnTh  = int(cute::size(GmemTiledCopyScalePacked{}));
      // Every premise checked, because a split that is wrong about ownership is a race, not a slowdown: the CTA must
      // be exactly two owner-sets wide, both wrap moduli must agree with the tile width, and the groups must halve.
      constexpr bool kSplitGroups =
#if defined(PPU_PACKED_SPLIT_GROUPS) && (PPU_PACKED_SPLIT_GROUPS != 0)
          (kThreads == 2 * kTN) && (kWrapTh == kTN) && (kOwnTh == kTN) && (kGrp % 2 == 0);
#else
          false;
#endif
      constexpr int kHalfG = kGrp / 2;

      // ONE THREAD, ITS OWN COLUMNS. With kPackedColsPerThread > 1 the copy gave thread t the span starting at
      // t * cpt, so t decodes exactly that span -- which is why this is ownership-safe where reading a neighbour's
      // column was not. At cpt == 1 this is the original loop unchanged.
      constexpr int kCPT = kPackedColsPerThread;
      for (int base = (kSplitGroups ? (thread_idx % kTN) : thread_idx) * kCPT; base < kTN;
           base += (kSplitGroups ? kTN : kThreads) * kCPT)
      for (int sub = 0; sub < kCPT; ++sub) {
        int const n = base + sub;
        if (n >= kTN) break;
        if (n >= residue_n) continue;                        // the same N bound the fp16 path predicates on
        uint8_t const* unit = reinterpret_cast<uint8_t const*>(&sRaw(n, cute::Int<0>{}, stage));
        uint32_t u[kPackedUnitWords];
        // THE LAST WORD MAY BE PARTIAL. With a unit that is not a multiple of four bytes the final read would run
        // past it, so the tail is assembled byte by byte -- the bytes beyond the unit are never referenced by any
        // field, and reading them would be out of bounds on the last column of the tile.
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < kPackedUnitWords; ++w) {
          if constexpr (kPackedScaleUnit % 4 == 0) {
            u[w] = *reinterpret_cast<uint32_t const*>(unit + 4 * w);
          } else {
            uint32_t acc = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < 4; ++b) {
              int const idx = 4 * w + b;
              if (idx < kPackedScaleUnit) acc |= uint32_t(unit[idx]) << (8 * b);
            }
            u[w] = acc;
          }
        }
        // half2(d, -dmin) for the whole column: one xor, hoisted out of the group loop. Replaces head_of_words'
        // two bitcasts AND the per-group negate of the min term.
        uint32_t const m2 = cutlass::gguf_packed::mul2_of_words(u);
        auto const h = cutlass::gguf_packed::head_of_words(u);
        (void)m2; (void)h;
        // ONE body, called from both halves with a compile-time G. The group index has to be a template argument --
        // every bit position in the unit is derived from it -- so the two halves cannot share a runtime offset.
        auto decode_group = [&] (auto g_) {
          constexpr int G = decltype(g_)::value;
          // TIMING-ONLY ABLATION. PPU_PACKED_SCALE_NOP=1 keeps the native transport and the shared STORES but drops
          // the decode ARITHMETIC, so three builds decompose the +12.9% instead of attributing all of it at once:
          //     baseline B   fp16 planes, cp.async writes them, no decode
          //     nop      N   native 16 B transport + the same stores, no arithmetic
          //     full     P   everything
          // giving arithmetic = P - N and transport+stores = N - B. RESULTS ARE DELIBERATELY WRONG under this flag.
          // The store still consumes the unit's first half so the 16 B smem load cannot be dead-code eliminated --
          // an ablation the compiler optimises away measures the compiler, not the kernel.
          //
          // A switch by this name was DOCUMENTED in PLAN_task20_scale.md and HANDOFF_packed_scale.md while its code
          // no longer existed: it did not survive the F/F' rewrites and nothing noticed, because a timing flag has no
          // gate that fails. Same defect shape as the rest of this task, one level up.
#if defined(PPU_PACKED_SCALE_NOP) && (PPU_PACKED_SCALE_NOP != 0)
          // NORMAL VALUES, NOT d AND dmin. Writing the unit's own d and dmin kept the 16 B load alive but made the
          // planes depend on the input, and d IS subnormal for 80.5% of superblocks on the real fixture.
          //
          // THAT IS NOT WHY packnop CAME OUT SLOWER, and I asserted it was. test_moe_splitk_bench fills its scale
          // buffer with a constant 0.0625 and its zero buffer with -0.0625 and hands the same allocation to the
          // packed path, which reinterprets it as 16-byte units -- so the bench's d and dmin are 0.0625, normal, and
          // the fixture's subnormals never reach it. Second time I attached that measured fact to the wrong effect.
          // The packnop anomaly is still unexplained; the leading candidates are timing noise at 2.3% against a
          // documented 13% dispersion, and a different compiler schedule, since the full decoder consumes all four
          // u[] words while this consumes only u[0] and the other shared loads may simply be eliminated.
          //
          // The change is kept because input-independent constants are the right shape for an ablation regardless --
          // it just does not buy what the first version of this comment claimed.
          //
          // 0x3C00 is 1.0 and 0x0000 is +0; ORing one bit of u[0] into the mantissa keeps the load from being dead
          // while staying firmly normal. The planes are still wrong on purpose -- read the time, never the MATCH.
          cutlass::gguf_packed::GroupScale sz;
          sz.scale = cutlass::half_t::bitcast(uint16_t(0x3C00u | (u[0] & 1u)));
          sz.zero  = cutlass::half_t::bitcast(uint16_t((u[0] >> 16) & 1u));
          if (false)
#else
          cutlass::gguf_packed::GroupScale sz;
#endif
          if constexpr (kPackedPairFast) {
            // BOTH FIELDS OF THE GROUP IN ONE 32-BIT LANE PAIR: one integer add carries the bias, the mask and the
            // magic OR for scale AND min together, then one ppu.sub.f16x2 and one ppu.fma.rtte.f16x2. 15 opcodes per
            // group down to ~11, and bit-identical rather than close -- l96 (A) checks that over 32768 real Q4_K
            // groups and (A0) checks each of the four identities it rests on separately. This touches only the
            // thread's OWN column, so it is independent of the constraint above.
            sz = cutlass::gguf_packed::group_pair_of_words<G, kPackedZMul, kPackedScaleBias>(u, m2);
          } else {
            sz = cutlass::gguf_packed::group_of_words<G, kPackedScaleBias, kPackedHasMin, kPackedZMul, kPackedFmt>(u, h);
          }
          // (n, group, stage): SmemLayoutScale's own modes. NOT the read side's flattened (n, 1, stage*SK+g) -- two
          // functions build a tensor called sS with DIFFERENT layouts, and using the wrong one faulted as
          // "TSM out of range" once already.
          if constexpr (kFusedScaleZero) {
            // ONE 32-BIT STORE. scale in the low half, zero in the high half -- the order the readers' even/odd views
            // above assume, and the order a little-endian half2 already has.
            //
            // sz.zero IS THE VALUE TO STORE, NOT y2. group_pair_of_words computes half2(d*sc, -dmin*mn) and then adds
            // `kPackedZMul * scale` to the zero AFTER splitting it (gguf_packed_scale.h, the ZMul arm), which cancels
            // the int4 converter's own -8. Packing the pre-correction pair back into 32 bits looks like it saves the
            // split and is simply WRONG -- it drops that cancellation.
            sSZw(n, cute::Int<G>{}, stage) = cutlass::gguf_packed::pack_h2(sz.scale, sz.zero);
          } else {
            sS(n, cute::Int<G>{}, stage) = sz.scale;
            if constexpr (kPackedHasMin) sZ(n, cute::Int<G>{}, stage) = sz.zero;
          }
        };
        if constexpr (kSplitGroups) {
          if (thread_idx < kTN)
            cute::for_each(cute::make_int_sequence<kHalfG>{},
                           [&] (auto i_) { decode_group(cute::Int<decltype(i_)::value>{}); });
          else
            cute::for_each(cute::make_int_sequence<kHalfG>{},
                           [&] (auto i_) { decode_group(cute::Int<kHalfG + decltype(i_)::value>{}); });
        } else {
          cute::for_each(cute::make_int_sequence<kGrp>{},
                         [&] (auto i_) { decode_group(cute::Int<decltype(i_)::value>{}); });
        }
      }
    }
  }


  // ON IS A TEMPLATE PARAMETER, and that is the whole point: `if constexpr` only skips INSTANTIATING a discarded branch
  // when its condition is value-dependent. With kPackedScaleOn (a class constant) both branches were instantiated, so
  // every unit paid for the packed tensors even with the path off -- measured as ~5% on the TK=64 control rows, which are
  // supposed to be byte-identical between the two builds. Routing the gate through a template parameter makes the
  // discarding real.

  // THE DECODING LOADER: gmem native unit -> registers -> the SAME fp16 smem planes the fp16 path uses.
  //
  // This is llama.cpp's shape (ggml-cuda/mmq.cuh, load_tiles_q4_K) and the reason it is right is amortisation, not
  // instruction count in isolation: there the decode runs once per (row, group) for the whole CTA and the result goes to
  // SHARED memory, so every lane that needs it reads it rather than re-deriving it. Doing it per lane in registers, which
  // is what the previous attempt did, multiplies the work by the number of lanes sharing a column (four, per l94 (7))
  // times the warps -- measured as ~1.4M extra ALU against 0.27M shared loads saved, and LSU was only 6% busy.
  //
  // Here: one thread per column reads its 16 B unit (d, dmin and all 8 groups' 6-bit codes), decodes with the bit
  // positions as compile-time constants, and stores the fp16 scale and zero. 64 columns x 8 groups = 512 decodes per CTA
  // per k-tile, against 2048 for the per-lane version.
  //
  // WHAT IT COSTS: this channel is no longer cp.async, because arithmetic between gmem and smem is exactly what cp.async
  // cannot do. The load is issued where the cp.async used to be and the barrier that already guards the stage
  // (cp_async_wait + __syncthreads at the last k_block) still guards it, so no new sync is added -- but the LDG's latency
  // is now exposed rather than overlapped, and only measurement says whether the B cp.async in flight covers it.
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
      // THE SAME swizzle as SmemLayoutScale, for the reason spelled out there: one buffer, two views, and they are
      // equal only while both carry it.
      using PlainSmemCopyLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{},
          make_shape(shape<0>(ScaleTileShape{}), Int<1>{}, Int<smem_scale_k>{})));
      using UnfusedSmemCopyLayoutScale =
          typename detail::MaybeScaleSwizzle<kScaleSwizzleOk, ScaleSwizzleT, PlainSmemCopyLayoutScale>::type;
      // THE READ SIDE'S FLATTENED VIEW, FUSED. Same shape (n, 1, stage*Scale_TileK + g), strides doubled, so a reader
      // written against the unfused layout keeps its code and only walks 4 bytes per element instead of 2. Checked
      // against the unfused layout rather than assumed, for the reason given at SmemLayoutScaleFusedWord: this is the
      // SECOND view of the same buffer, and the two going out of step is the documented failure here.
      static_assert(sz_copy_layout_is_compact<kFusedScaleZero, UnfusedSmemCopyLayoutScale>(),
                    "PPU_PACKED_SCALE_FUSED assumes the compact flattened scale copy layout");
      using FusedSmemCopyLayoutScale = decltype(make_layout(
          make_shape(Int<kSZ_N>{}, Int<1>{}, Int<smem_scale_k>{}),
          make_stride(_2{}, _0{}, Int<2 * kSZ_N>{})));
      using SmemCopyLayoutScale =
          cute::conditional_t<kFusedScaleZero, FusedSmemCopyLayoutScale, UnfusedSmemCopyLayoutScale>;
      Tensor sS   = make_tensor(make_smem_ptr(storage.smem_scale.begin()), SmemCopyLayoutScale{});
      Tensor tCsS = smem_thr_copy_S.partition_S(sS);
      Tensor tCrS = make_scale_fragment(thr_mma, sS);

      // APPENDED, never inserted: every existing site reads this tuple positionally (get<0>..get<3>), so the packed
      // extras go after them and no index moves. Appended UNCONDITIONALLY so the arity never depends on a macro -- with
      // `if constexpr` selecting the fill, a discarded branch still has to name get<4>, and a macro-dependent arity
      // would make that ill-formed on the other configuration.
      // Built only when the path is on; an empty tuple otherwise, so an off unit constructs nothing at all. The
      // reinterpret is needed because with the path off that engine is half_t.
      // Built unconditionally. packed_or_empty (a template-parameter gate around a generic lambda) is the standard way
      // to avoid that, and EDG instantiates the lambda body anyway -- so the empty stand-in reached partition_S. Its
      // cost is measured, not guessed: ~1.5 us on the TK=64 control rows. Second-order next to the decode, and revisited
      // only after the decode's win is on the table; one variable at a time.
      // POINTED AT THE BUFFER ITS LAYOUT DESCRIBES. This tensor is dead -- ScaleOnly puts it at tuple index 2 and that
      // arm reads only 0 and 1; ScaleZero puts it at 4 and nothing reads past 3 -- but it was built on smem_scale, the
      // fp16 plane, with SmemLayoutScalePackedStaged, which describes the 16-byte staging. That is a leftover from
      // when the staging lived inside smem_scale, and it is exactly the kind of thing that is revived and then wrong
      // twice over: wrong buffer, and (since PPU_SCALE_SWIZZLE) an unswizzled view of a buffer every live view now
      // swizzles. Repointed rather than deleted so the tuple indices, which every consumer reads positionally, do not
      // move. Found by enumerating every tensor built on smem_scale/smem_zero -- six of them, where I had claimed two.
      Tensor sSp  = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t*>(
                                    kPackedScaleOn ? (void*)storage.smem_scale_raw.begin()
                                                   : (void*)storage.smem_scale.begin())),
                                SmemLayoutScalePackedStaged{});
      Tensor tCcS = smem_thr_copy_S.partition_S(make_identity_tensor(shape(sS)));
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS, sSp, tCcS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ   = make_tensor(make_smem_ptr(zero_smem_base(storage)), SmemCopyLayoutScale{});
        Tensor tCsZ = smem_thr_copy_S.partition_S(sZ);
        Tensor tCrZ = make_scale_fragment(thr_mma, sZ);
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ, sSp, tCcS);
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
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ           = cute::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view = cute::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_,_,0,scale_k_idx), tCrZ_copy_view(_,_,0));
        }
        if constexpr (false) {} else {
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
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  /// A's row 0 into the PACKED cube layout. Four contiguous 16-half runs per cube at the offsets l86 exported, and
  /// inside a run the logical k advances with the physical word, so each run is a plain copy with no shuffle.
  /// cp.async and not the AIU: the AIU write is .padz and would write each cube's 15 zero rows over its packed
  /// neighbour's row 0. One 128-bit transfer per thread, kAWrThreads threads, so one instruction on one warp.
  template <class GA, class EA>
  CUTLASS_DEVICE
  void copy_A_packed_row0(GA const& gA, EA* smem_a, int k_tile, int pipe, int thread_idx) {
    if (thread_idx >= kAWrThreads) return;
    int const per = kASlices * 2;
    int const c   = thread_idx / per;            // which cube in this stage
    int const run = (thread_idx % per) / 2;      // which of row 0's runs
    int const h   = thread_idx % 2;              // which 8-half half of the run
    // The raw op, not a Copy_Atom: the atom's SrcLayout only matches when it is reached through a TiledCopy, and
    // the thread -> (cube, run, half) map here is not a tiler. One 128-bit ppu.cp.async.cg per thread.
    auto const& gsrc = *reinterpret_cast<cute::uint128_t const*>(
        &gA(Int<0>{}, c * kACubeW + run * 16 + h * 8, k_tile));
    auto&       sdst = *reinterpret_cast<cute::uint128_t*>(
        smem_a + kAPackPitch * (c + kACubes * pipe) + aPackRunOff(run) + h * 8);
    PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(gsrc, sdst);
  }
#endif

  // PPU_B_DEQUANT_NOP -- TIMING ONLY, RESULTS ARE DELIBERATELY WRONG. It answers the one question the packed-scale
  // NOP cannot: how much of the 20.11 us baseline is the int4->fp16 dequant pipeline itself? That chain is 1,898,496
  // instructions against 131,072 mma, i.e. 43% of dynamic instruction count -- but an instruction share is not a
  // cycle share, and those instructions can issue while memory operations are outstanding. Only removing them says.
  //
  // WHAT IT KEEPS, because an ablation that changes memory traffic answers a different question: the B s2r loads
  // (cvt_in stays live), the scale/zero smem reads (one element of each fragment is still consumed, so the copies
  // cannot be dead-code eliminated), the mma count, the tile shapes and every barrier. What it drops: the conversion
  // for all but one word, and N-1 of the N elementwise scale/zero applications per atom.
  //
  // The one-element form is deliberate. Skipping the transforms entirely would let the compiler delete the scale
  // copies with them, and the run would then measure a kernel that also stopped reading shared memory -- the same
  // trap PPU_PACKED_SCALE_NOP avoids by keeping its unit load consumed.
  template <class TA, class TB, class TC, class Op>
  CUTLASS_DEVICE static void bdq_transform(TA&& a, TB&& b, TC&& c, Op op) {
#if defined(PPU_B_DEQUANT_NOP) && (PPU_B_DEQUANT_NOP != 0)
    c(0) = op(a(0), b(0));
#else
    cute::transform(a, b, c, op);
#endif
  }

  /// Utilities to transform B.
  // FINE-grained scale (gs < B-copy-step K, i.e. Scale_TileK > K_BLOCK_MAX): a single copy step's K_ATOM_PER_COPY
  // mma atoms straddle MORE than one scale group, so one pre-loaded scale reg can't cover the step (and the coarse
  // GroupK = K_BLOCK_MAX/Scale_TileK is 0). Here each mma atom reloads ITS group's scale straight from smem:
  // atom `a` (0..mma_K_atoms) belongs to group a/(mma_K_atoms/Scale_TileK), at smem slot read_stage*Scale_TileK+g.
  // This needs tiled_copy_and_views (smem_tiled_copy_S + reg-copy dst views) + read_stage, so both are passed in.
  template <typename RealInternalElementB,
            class TCrB_load,
            class TCrB_mma,
            int K_ATOM_PER_COPY,
            class... Ts,
            class CopyViews,
            class KBlockT,
            class PfPack>
  CUTLASS_DEVICE
  void transform_B_kblock(
    TCrB_load const& tCrB_load,
    TCrB_mma& tCrB_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    // KBlockT, NOT int: the callers already hold a static k_block (for_each gives Int<x>, and k_block_next is
    // (Int<x> + _1) % K_BLOCK_MAX, also static). Taking it as an int erased that, so atom_idx, g = atom_idx/APG_
    // and the `% APG_ == 0` guard all turned into runtime work -- s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 +
    // s.shra 0.09 + s.mull 0.16 per mma in the profile, ~1.8 of 34. Kept static, the guard folds to if constexpr
    // and vanishes with the division and modulo. Constant folding only; numerics unchanged.
    KBlockT const& k_block,
    cute::Int<K_ATOM_PER_COPY> k_atom,
    CopyViews const& tiled_copy_and_views,
    int const read_stage,
    // The second scale/zero register set, or an empty tuple. A separate template parameter and NOT an extension of
    // the Ts... pack above: appending to cute::tuple<Ts...> fails deduction.
    PfPack const& pf) {

    static constexpr int K_BLOCK_STATIC = int(KBlockT{});
    Tensor cvt_in  = recast<RealInternalElementB>(tCrB_load(_, _, k_block));
    Tensor cvt_out = make_tensor(tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(), cvt_in.layout());

    using CPY_VEC = Int<4 * 32 / sizeof_bits<RealInternalElementB>::value>;
#if defined(PPU_B_DEQUANT_NOP) && (PPU_B_DEQUANT_NOP != 0)
    // Nothing. The earlier version copied one 32-bit word of cvt_in into cvt_out to keep the B load alive, on the
    // belief that the rest of cvt_out held the previous k-tile's converted halfs -- it does not, the fragment is
    // never initialised, and those raw int4 bits read as fp16 are NaN or Inf about as often as not. The B s2r does
    // not need that crutch: it is a TSM_LD_SWZL implemented as asm volatile, so it survives its results going unused.
    (void)cvt_in; (void)cvt_out;
#else
    convert_tensor(cvt_in, cvt_out, CPY_VEC{});
#endif

    constexpr int MMA_KA_ = decltype(cute::size<2>(tCrB_mma))::value;   // total mma-K atoms in the tile
    constexpr int KBM_    = MMA_KA_ / K_ATOM_PER_COPY;                  // K_BLOCK_MAX (copy steps)
    constexpr bool FINE   = (int(Scale_TileK) > KBM_);                  // gs < copy-step K -> per-atom scale
    constexpr int APG_    = FINE ? (MMA_KA_ / int(Scale_TileK)) : 1;    // mma atoms per scale group (FINE only)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // do nothing
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      if constexpr (!FINE) {
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + decltype(i_)::value;
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
        });
      } else {
        // FINE: write via tCrS_copy_view (a retile VIEW of the ORIGINAL fragment) then read the ORIGINAL back --
        // NOT the local `tCrS` copy above (make_fragment_like is owning, so `auto tCrS` snapshots stale rmem).
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCsS              = cute::get<0>(partitioned_extra_info);
        auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
    // COMPILE-TIME ATOM INDEX. i used to be a runtime int, so atom_idx, g = atom_idx / APG_ and the
    // `atom_idx % APG_ == 0` guard all became runtime work even though K_ATOM_PER_COPY, APG_ and k_block are
    // constants: the profile showed s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 + s.shra 0.09 + s.mull 0.16 per mma,
    // about 1.8 of 34. With for_each the index is Int<i>, the guard folds to if constexpr and disappears, and the
    // division and modulo fold away. Same reason the main loop uses for_each -- see its comment about needing
    // k_block to be Int<x>. Numerics are unchanged; this is constant folding only.
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int I = decltype(i_)::value;
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
          constexpr int g = atom_idx / APG_;                            // this atom's scale group within the tile
          if constexpr (atom_idx % APG_ == 0)                             // reload only at a group's first atom
            copy(smem_tiled_copy_S, tCsS(_,_,0, read_stage * int(Scale_TileK) + g), tCrS_copy_view(_,_,0));
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
                          tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
        });
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      auto tCrZ = cute::get<3>(partitioned_extra_info);
      if constexpr (!FINE) {
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + decltype(i_)::value;
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrZ(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
        });
      } else {
        // FINE: see ConvertAndScale note -- write via the copy VIEWs, read the ORIGINAL fragments back.
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCsS              = cute::get<0>(partitioned_extra_info);
        auto tCsZ              = cute::get<2>(partitioned_extra_info);
        auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
        auto tCrZ_copy_view    = cute::get<2>(tiled_copy_and_views);
    // COMPILE-TIME ATOM INDEX. i used to be a runtime int, so atom_idx, g = atom_idx / APG_ and the
    // `atom_idx % APG_ == 0` guard all became runtime work even though K_ATOM_PER_COPY, APG_ and k_block are
    // constants: the profile showed s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 + s.shra 0.09 + s.mull 0.16 per mma,
    // about 1.8 of 34. With for_each the index is Int<i>, the guard folds to if constexpr and disappears, and the
    // division and modulo fold away. Same reason the main loop uses for_each -- see its comment about needing
    // k_block to be Int<x>. Numerics are unchanged; this is constant folding only.
        // PREFETCH IS AN APPLICABILITY, NOT A REQUIREMENT. Two register sets cover a copy step with at most two
        // scale groups; TileK=64 configs have more and must keep the original per-group reload. Writing that as a
        // static_assert made three units fail to compile instead of falling back -- the same shape of mistake as a
        // boundary check that only guards one end.
        constexpr int GRP = (K_ATOM_PER_COPY % APG_ == 0) ? (K_ATOM_PER_COPY / APG_) : 0;
        // Prefetching a smem read that no longer happens: where the packed path is on, a group costs kPackedSlots
        // decodes from registers and there is no load latency to hide. Per UNIT, not per binary.
        constexpr bool kPfOk = !kPackedScaleOn &&
#if defined(PPU_SCALE_PREFETCH) && (PPU_SCALE_PREFETCH != 0)
            (GRP == 2) && (cute::tuple_size<PfPack>::value == 4);
#else
            false;
#endif
        if constexpr (kPfOk) {
          // BOTH groups load before any transform, group gi into register set gi % 2, so the second group's data has
          // a whole group of atoms to arrive in instead of being used one instruction after its load. pf carries the
          // second set: fragments 0,1 and their copy views 2,3.
          constexpr int g0 = (K_BLOCK_STATIC * K_ATOM_PER_COPY) / APG_;
          cute::for_each(cute::make_int_sequence<GRP>{}, [&] (auto gi_) {
            constexpr int GI = decltype(gi_)::value;
            const int sk = read_stage * int(Scale_TileK) + g0 + GI;
            if constexpr (GI % 2 == 0) {
              copy(smem_tiled_copy_S, tCsS(_,_,0,sk), tCrS_copy_view(_,_,0));
              copy(smem_tiled_copy_S, tCsZ(_,_,0,sk), tCrZ_copy_view(_,_,0));
            } else {
              copy(smem_tiled_copy_S, tCsS(_,_,0,sk), cute::get<2>(pf)(_,_,0));
              copy(smem_tiled_copy_S, tCsZ(_,_,0,sk), cute::get<3>(pf)(_,_,0));
            }
          });
          cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
            constexpr int I        = decltype(i_)::value;
            constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
            constexpr int GI       = I / APG_;
            if constexpr (GI % 2 == 0) {
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::multiplies{});
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::plus{});
            } else {
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<0>(pf)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::multiplies{});
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<1>(pf)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::plus{});
            }
          });
                } else {
          cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
            constexpr int I = decltype(i_)::value;
            constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
            constexpr int g = atom_idx / APG_;
            if constexpr (atom_idx % APG_ == 0) {                        // reload only at a group's first atom
              const int sk = read_stage * int(Scale_TileK) + g;
              copy(smem_tiled_copy_S, tCsS(_,_,0,sk), tCrS_copy_view(_,_,0));
              copy(smem_tiled_copy_S, tCsZ(_,_,0,sk), tCrZ_copy_view(_,_,0));
            }
            bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
                            tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
            bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_, _, 0),
                            tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
          });
        }
      }
    }
    else {
      assert(false);
    //   static_assert(cutlass::detail::dependent_false<KernelSchedule>, "No A data is loaded.");
    }
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


public:
  // A PUBLIC FORWARDER FOR THE PUBLICATION STEP, so it can be compiled in isolation and its emitted stores counted.
  // packed_decode_stage is private and is only ever reached from mma(), which drags in the whole pipeline -- so the
  // question "does the fused branch actually emit ONE 32-bit shared store" had no local answer, and the flag shipped
  // to the box in a state where the counter it must move did not move. Byte-neutral changes cannot be seen in any
  // run (same shared bytes, same results, same instruction mix on every other path), so the only observables left
  // are the type, which l100_fused_active.cu asserts, and the generated code, which this makes reachable.
  template <bool On, class Storage>
  CUTLASS_DEVICE static void probe_packed_decode_stage(Storage& storage, int stage, int thread_idx,
                                                       int64_t residue_n) {
    packed_decode_stage<On>(storage, stage, thread_idx, residue_n);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
