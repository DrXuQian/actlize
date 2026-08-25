/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/arch/copy_ppu.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/layout.hpp>

namespace cute
{

template <class NumBitsPerAIU, typename Element, bool Trans, bool Swzl>
struct Copy_Traits<PPU0010_AIU_LOAD<NumBitsPerAIU, Element, Trans, Swzl>>
{
  using ThrID   = Layout<_1>;

  // (i) WHY THERE IS NO 2-D LAYOUT HERE, and where the structure actually lives. The AIU bulk load is issued by ONE
  // thread and moves a whole (cube_h x cube_w) tile, so from cute's point of view it is an opaque blob of NumBitsPerAIU
  // bits -- hence the flat Src/Dst layouts. The cube's 2-D structure is not missing, it is in AiuDesc (dim_h, dim_w,
  // cube_h, cube_w), which is why the collective's load_init_B* now derives those extents from the gmem TENSOR rather
  // than restating the folded row/column counts (see the note there for what restating them cost).
  //
  // AND WHY NO LogicalTV. The read atom (PPU0010_TSM_LD_SWZL) carries one, but it is already the map of write-then-read:
  // both instructions carry `.swzl` and the two swizzles cancel, so what its LogicalTV describes is the GMEM-RELATIVE
  // word, not an smem address. A separate write-side LogicalTV would therefore be the identity on the cube -- vacuous --
  // and decomposing the cancellation into two halves would mean inventing a factorisation neither instruction exposes.
  // The checkable statement is the one the read side already makes: its LogicalTV is a bijection onto the cube's words,
  // verified for fp16 (where no converter or offline exists to hide a compensating error) in
  // fold_derivation/l17_fp16_identity.cu.
  //
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerAIU>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerAIU>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // can't use reference since CopyAtom has no constructor
  AiuDesc desc_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void *dst_ptr, const void *src_ptr,
               Coord const& src_coord, seq<Is...>) const
  {
    PPU0010_AIU_LOAD<NumBitsPerAIU, Element, Trans, Swzl>::copy(dst_ptr, src_ptr, desc_, get<Is>(src_coord)...);
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TD>::value, "Expected smem dst for AIU_LOAD");

    // if (cute::thread0()) {
    //   print("        copy_unpack, src = "); print(src); print("\n");
    //   print("                     dst = "); print(dst); print("\n");
    // }

    if constexpr (is_mix_iterator<typename TS::iterator>::value) {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().ptr_.get(), src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    } else {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), traits.desc_.gmem_ptr, src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
  }

};

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum, int CubePitch>
struct Copy_Traits<PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum, CubePitch>>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  // ---------------------------------------------------------------------------------------------------------
  // LogicalTV -- (thread, vreg) -> the LOGICAL 32-bit word of the cube that vreg reads.
  //
  // DESCRIPTIVE ONLY. It is deliberately NOT SrcLayout/DstLayout: those feed partition_S/retile_D, whose result
  // becomes the coord_ handed to the asm, so changing them changes runtime addressing on every path (fp16, dense,
  // MoE, every quant format). This member exists so that offline placement generators and static checks have ONE
  // source of truth instead of each re-deriving the delivery pattern. Wiring cute up to compute the addresses is a
  // separate, behaviour-changing step that needs a box regression.
  //
  // Derived from ppu_tsm_ld_swzl_sim's SWAP=true branch with the two swizzle terms stripped -- they cancel against
  // the AIU write's matching .swzl, so what is left is the gmem-relative word:
  //     row = (v/2)*8 + lane/4 ,  word-in-row = (v%2)*4 + lane%4 ,  index = row*WordsPerRow + word
  // Validated in Kernels/general/w4a16_gemm/cutlass_w4a16/fold_derivation/:
  //   l2l3_layouts.cu  0 mismatch over all 128 (lane,vreg), bijective, against the sim's own arithmetic
  //   l17_fp16_identity.cu  the fp16 chain composes to the IDENTITY -- no converter and no offline exist there to
  //                         hide a compensating error, so this is the strongest available check
  //   l7/l10/l12/l13/l16    0 mismatch against the real offline for int1/int2/int4, fold and unfolded, whole buffer
  //
  // MULTI-SLICE. A cube wider than 32B is read as several 32B slices, and the sim carries a cross-slice term
  // (vec + slice_start_vec) % 8 with ssv in {0,4,2,6} -- a rotate that carries for slices 2 and 3, so NOT a plain
  // stride. It looked like that would force a custom swizzle functor. It does not: the rotate belongs to the
  // SWIZZLE, and the AIU write applies the matching one, so it cancels and the LOGICAL map has no rotate at all.
  // Measured, not assumed -- fold_derivation/l19_multislice.cu tries {strip, keep} x {slice-major, rowblock-major}
  // and only "strip + slice-major" yields the fp16 identity, at 2 slices (0/2048) and 4 slices (0/4096), with a
  // 1-slice control that all four pass because ssv is 0 there.
  //
  // So the slice is just one more mode with stride 8 words, and stock cute covers ANY cube width. The copy
  // instances walk slice-major: all slices of a row-block before the next row-block.
  static constexpr int LogicalWordsPerRow = CUBE_W * sizeof_bits<Element>::value / 32;
  static constexpr int LogicalSlices      = LogicalWordsPerRow / 8;
  static constexpr bool LogicalTVValid    = (LogicalWordsPerRow % 8 == 0);
  // ((lane%4, lane/4), (v%2, v/2), slice) -> logical 32-bit word index within the cube
  using LogicalTV = Layout<Shape <Shape<_4, Int<8>>,                    Shape<_2, _2>,                     Int<LogicalSlices>>,
                           Stride<Stride<_1, Int<LogicalWordsPerRow>>,  Stride<_4, Int<8 * LogicalWordsPerRow>>, _8>>;

  void *smem_base_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void *dst_ptr, void* src_ptr,
               Coord const& src_coord, seq<Is...>) const
  {
    PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum, CubePitch>::copy(dst_ptr, src_ptr, get<Is>(src_coord)...);
  }

  template <class TS, class SLayout,
          class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    if constexpr (is_mix_iterator<typename TS::iterator>::value) {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().ptr_.get(), src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    } else {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), traits.smem_base_, src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
  }
};

// Logical two-register view of the physical PPU0010 m8n8.x4 swizzle load used
// by m8n16k16's A operand.  v2/v3 are intentionally absent from every CuTe
// layout: they are an implementation detail contained by
// PPU0010_TSM_LD_SWZL_M8 and must never enlarge or overwrite the m8 A
// fragment.
template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans,
          int InstNum, int CubePitch, int StagePitch>
struct Copy_Traits<PPU0010_TSM_LD_SWZL_M8<
    Element, CUBE_H, CUBE_W, Swap, Trans, InstNum, CubePitch, StagePitch>>
{
  using Operation = PPU0010_TSM_LD_SWZL_M8<
      Element, CUBE_H, CUBE_W, Swap, Trans, InstNum, CubePitch, StagePitch>;

  using ThrID = Layout<_32>;

  // Four fp16 values (64 bits) per lane are the semantic source and
  // destination of this projected copy.  The raw instruction's other two
  // registers never enter the logical partition.
  using SrcLayout = Layout<Shape<_32, _64>, Stride<_64, _1>>;
  using DstLayout = Layout<Shape<_32, Shape<_32, _2>>,
                           Stride<_32, Stride<_1, _1024>>>;
  using RefLayout = DstLayout;

  static_assert(size<0>(SrcLayout{}) == 32 && size<1>(SrcLayout{}) == 64,
                "PPU0010 m8 A projected source must be 64 bits per lane");
  static_assert(size<0>(DstLayout{}) == 32 && size<1>(DstLayout{}) == 64 &&
                    size<1, 1>(DstLayout{}) == Operation::kLogicalRegisters,
                "PPU0010 m8 A projected destination must contain exactly two registers per lane");
  static_assert(size(SrcLayout{}) == size(DstLayout{}),
                "PPU0010 m8 A projection must preserve its logical bit count");

  void *smem_base_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void *dst_ptr, void *src_ptr,
               Coord const& src_coord, seq<Is...>) const
  {
    Operation::copy(dst_ptr, src_ptr, get<Is>(src_coord)...);
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    if constexpr (is_mix_iterator<typename TS::iterator>::value) {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().ptr_.get(),
                          src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    } else {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), traits.smem_base_,
                          src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
  }
};

} // namespace cute
