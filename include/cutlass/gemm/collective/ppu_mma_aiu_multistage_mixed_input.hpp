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
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

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
#if defined(PPU_SCALE_PAD) && (PPU_SCALE_PAD > 0)
  static constexpr int kScalePad = PPU_SCALE_PAD;
  using SmemLayoutScale = decltype(make_layout(
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{}),
    make_stride(_1{},
                Int<int(shape<0>(ScaleTileShape{})) + kScalePad>{},
                Int<(int(shape<0>(ScaleTileShape{})) + kScalePad) * int(shape<1>(ScaleTileShape{}))>{})));
#else
  using SmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{})));
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
    cute::tuple<Ts...> const& partitioned_extra_info,
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
    convert_tensor(cvt_in, cvt_out, CPY_VEC{});

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
          cute::transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
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
          if constexpr (atom_idx % APG_ == 0)                            // reload only at a group's first atom
            copy(smem_tiled_copy_S, tCsS(_,_,0, read_stage * int(Scale_TileK) + g), tCrS_copy_view(_,_,0));
          cute::transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
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
          cute::transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
          cute::transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrZ(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
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
        constexpr bool kPfOk =
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
              cute::transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::multiplies{});
              cute::transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::plus{});
            } else {
              cute::transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<0>(pf)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::multiplies{});
              cute::transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<1>(pf)(_,_,0),
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
            cute::transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
                            tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
            cute::transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_, _, 0),
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

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
