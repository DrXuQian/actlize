/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*!
    \file
    \brief Boost-like numeric conversion operator for int8 and CUTLASS int4b_t interleaved in a register
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"
// MixGemmEmit is a cute Layout now (it was a closed-form loop), so the algebra has to be available here.
#include "cute/layout.hpp"
#include <utility>

namespace cutlass
{

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.

// ===================== THE EMISSION MAP, hoisted above the converters that now USE it =====================
// ================= EMISSION LAYOUT of the mixed-input converters, unified across widths =================
//
// Every converter below is 1:1 in element COUNT -- N sub-byte codes in, N fp16 out -- and only the value width
// changes. But the position map is NOT the identity: each writes its results to h2[base + off] with a hand-chosen
// base and offset table, and that permutation is exactly what the offline weight relayout has to compensate. This
// struct is that permutation, and it turns out to be ONE map for all three widths, parameterised only by how many
// codes fit in a vreg:
//
//     e(code, vreg) = half + 2*c0 + sum_{i>=1} 4*2^i * ci + (2*CodesPerVreg)*(vreg%2) + 4*(vreg/2)
//
// where the code index's TOP bit is `half` (which fp16 of the half2) and c0..c_{nb-2} are the rest. Read off:
//     int1  CodesPerVreg=32  code bits -> 2, 8, 16, 32, 1     vreg -> 64, 4
//     int2  CodesPerVreg=16  code bits -> 2, 8, 16, 1          vreg -> 32, 4
//     int4  CodesPerVreg= 8  code bits -> 2, 8, 1              vreg -> 16, 4
// int4 looks structurally different in the source -- four base-8 calls, result_ptr[0,2,1,3], a 4-element block swap
// -- only because it folds the vreg into a GLOBAL code index. Split it back out (vreg = c/8, code = c%8) and it is
// the same map.
//
// NO LONGER DESCRIPTIVE. int2 and int1 now index h2[] THROUGH MixGemmEmit<Bits>::index(t, v)/2, and int4 -- whose
// composed base8-o-permutation-o-block-swap form is all risk to rewrite on the 55.9% production path -- is PINNED to
// it by a static_assert instead. So a new bit width inherits a map that is bijective by construction and agrees with
// three existing converters, rather than being hand-derived again. That hand-derivation is where the int2
// sigma_n = n & ~8 bug came from. The point was
// a single source of truth for offline placement generators, which previously each re-derived this by hand.
// Validated 0 mismatch against all three converters' own mask constants and sub-vector permutations in
// Kernels/general/w4a16_gemm/cutlass_w4a16/fold_derivation/ (l2l3_layouts.cu, l12_widths.cu), and end to end
// against the real offline for fold and unfolded paths, whole buffer (l13, l16).
//
// NOTE the +8 that add_bias_and_interleave_int4s applies is a VALUE transform, not a position one, so it is
// deliberately absent here -- a layout describes where a code goes, not what it becomes.
template <int Bits>
struct MixGemmEmit {
  static constexpr int kCodesPerVreg = 32 / Bits;
  static constexpr int kNumCodeBits  = (Bits == 1) ? 5 : (Bits == 2) ? 4 : (Bits == 4) ? 3 : 0;
  static_assert(kNumCodeBits != 0, "MixGemmEmit: only 1/2/4-bit codes");

  // THIS IS A cute LAYOUT, not a closed-form loop. It used to be the loop; the formula is a sum of per-bit weights,
  // which is precisely what a Layout is, and expressing it as one lets it COMPOSE with the other layouts in the
  // chain (pi = right_inverse(frag.layout()), LogicalTV, partition_B) instead of being a function every consumer has
  // to call specially. Verified equal to the loop for all three widths in
  // fold_derivation/l30_should_have_been_cute.cu.
  //
  //   shape  ((2, ..., 2), (2, 2))          nb code bits, then the 2 vreg bits
  //   stride ((2, 8, 16, ..., 1), (2*CPW, 4))
  //          code bit 0 -> 2 ; bit i in [1, nb-2] -> 4<<i ; bit nb-1 -> 1 (the half2 lo/hi selector)
  //
  // The 1-D DOMAIN IS `code + kCodesPerVreg * vreg` on purpose: cute decomposes an integer coordinate
  // colexicographically through the nested shape, and our modes are exactly code's bits low-to-high followed by
  // vreg's, so a flat index needs no manual splitting and index()'s signature is unchanged.
  template <int I>
  using CodeStride = cute::Int<(I == kNumCodeBits - 1) ? 1 : (I == 0 ? 2 : (4 << I))>;

  template <std::size_t... I>
  static constexpr auto make_emit_layout(std::index_sequence<I...>) {
    return cute::make_layout(
        cute::make_shape (cute::make_shape((void(I), cute::_2{})...), cute::make_shape(cute::_2{}, cute::_2{})),
        cute::make_stride(cute::make_stride(CodeStride<int(I)>{}...),
                          cute::make_stride(cute::Int<2 * kCodesPerVreg>{}, cute::_4{})));
  }
  using EmitLayout = decltype(make_emit_layout(std::make_index_sequence<kNumCodeBits>{}));

  // (code within a vreg, vreg) -> fp16 element index within the converter's 4*kCodesPerVreg outputs
  static constexpr int index(int code, int vreg) {
    return int(EmitLayout{}(code + kCodesPerVreg * vreg));
  }
  static constexpr int kNumOutputs = 4 * kCodesPerVreg;

  // a permutation or the model is wrong; checked at compile time so it cannot rot
  static constexpr bool bijective() {
    bool seen[4 * 32] = {};
    for (int v = 0; v < 4; ++v)
      for (int c = 0; c < kCodesPerVreg; ++c) {
        const int e = index(c, v);
        if (e < 0 || e >= kNumOutputs || seen[e]) return false;
        seen[e] = true;
      }
    return true;
  }
  static_assert(bijective(), "MixGemmEmit: emission map is not a permutation");
};

// int4's wide converter does NOT use a flat table: it composes the base-8 converter's internal order, a vreg
// permutation (result_ptr[0]=s0,[2]=s1,[1]=s2,[3]=s3) and a 4-element block swap (elt_b16 1<->2, 5<->6). Rewriting
// that to be MixGemmEmit-driven is pure risk on the 55.9% production path, so it is PINNED instead: if anyone
// changes either the composition or MixGemmEmit, this fails at compile time. Verified independently in
// fold_derivation/l29_emit_is_converter.cu, where the same three permutations compose onto the closed form with 0
// mismatch -- which is much stronger evidence for the model than a flat table agreeing.
namespace detail {
constexpr int mixgemm_int4_composed(int c, int v) {
  const int l = 2 * (c & 3) + (c >= 4 ? 1 : 0);        // base-8: h[c&3], low lane c<4, high lane c>=4
  const int p[4] = {0, 2, 1, 3};
  const int g0 = 8 * p[v] + l;
  int b = g0 / 4;
  b = (b == 1) ? 2 : (b == 2) ? 1 : (b == 5) ? 6 : (b == 6) ? 5 : b;
  return 4 * b + (g0 % 4);
}
constexpr bool mixgemm_int4_agrees() {
  for (int v = 0; v < 4; ++v)
    for (int c = 0; c < 8; ++c)
      if (mixgemm_int4_composed(c, v) != MixGemmEmit<4>::index(c, v)) return false;
  return true;
}
static_assert(mixgemm_int4_agrees(),
              "int4's composed emission (base8 o vreg-perm o block-swap) no longer matches MixGemmEmit<4>");
}  // namespace detail

// ================== CHUNK-AWARE EMISSION, one implementation for int1 AND int2 ==================
//
// The constants are DERIVED from Bits, not tabulated. int1 had 16 hardcoded (mask, mul, add) triples and int2 another
// 8 -- two tables encoding one rule, which also meant chunking int2 would mean copying the whole emitter. Verified
// against both widths' actual constants (24 rows, 0 differ) in fold_derivation/l36_emit_constants.cu:
//
//     pairs per vreg = 16/Bits          per level = 8/Bits    (level 1 reads reg>>8)
//     b              = (t % (8/Bits)) * Bits
//     mask           = m | (m<<16),  m = ((1<<Bits)-1) << b
//     mul            = fp16(2^-b)       = (15 - b) << 10, duplicated
//     add            = fp16(-2^(10-b))  = 0x8000 | ((25 - b) << 10), duplicated
//
// int4 is deliberately NOT covered: it uses the +8 bias with the 1032/72 magic on nibbles, a different scheme -- and
// it needs no chunking, since slots/delivery = 4 leaves it one k-atom per copy step, i.e. NChunk = 1.
//
// WHY CHUNKING EXISTS. One 16 B swzl delivery carries 16/Bits mma atom-slots of B, which is simultaneously int1's
// advantage (one read feeds 4x the mma of int4) and its handicap: the fp16 fragment must hold a whole delivery, so
// B_regs = 4*MMA_N*MMA_K >= 64 for int1, and those 64 push the best config past the power-of-two register billing
// boundary, halving warps/CU from 32 to 16. Chunking decouples how many atom-slots the delivery COVERS from how many
// are fp16 AT ONCE: all of them still convert and get used, in NChunk batches. Measured 186 -> 142 registers.
//
// at() AND keep() ARE LAYOUT COMPOSITIONS over FragLayout, which is tCrB_mma's OWN layout passed in by the caller --
// not restated. Restating it by hand (assuming a compact (8,32) where the fragment is ((1,2,4),32,8)) is what produced
// "MMA_N atom 0 right, atoms 1-3 wrong" on the box.
//     L_dst = shape(L) : (val strides, size(val), 0)   compact in n, stride-0 in k = ONE k-atom
//     L_k   = shape(L) : (zeros,       0,        1)    which k-atom an element belongs to
// composed with right_inverse(L); cute simplifies both symbolically. l35 checks them against the arithmetic they
// replace on all 256 (chunk, t, v) combinations.
//
// V AND T ARE TEMPLATE PARAMETERS, not loop variables: the gate is `if constexpr (keep(T, V))`, and a plain `if`
// relying on unrolling plus dead-code elimination would make the register saving depend on the compiler folding
// branches -- the assumption the scale-broadcast episode punished.
// (e) THE K-PERMUTATION RULE, in ONE place. The builder computes this at ppu_mma_builder.inl and every offline
// generator needs the same answer, so it used to be restated by hand -- and a plane_map that hardcoded the non-fold
// branch silently broke a FOLDED plane: at Block_K=64 with F=2 the composition covered 4160 of 8192 logical elements
// instead of being a bijection. Unfolded it is the K span one 32 B swzl delivery carries (int4 64 / int2 128 / int1 256);
// folded, the fragment must keep ordinary N x K register semantics because the fold lives only in the load layer, so it
// is TileShape.K. Both the builder and the generators now read this.
template <int Bits, int BlockK, int FoldF>
struct MixGemmMmaPermK {
  static constexpr int value = (FoldF > 1) ? BlockK : (32 * 8 / Bits);
};

// WHERE A CHUNKED OUTPUT GOES, as two compositions over the fragment layout. Extracted so the 2-plane converter can
// share it: the two converters have DIFFERENT emission orders (MixGemmEmit for one plane, the _E2 lines' AtLayout for
// two), but the PLACEMENT rule is the same, and re-deriving it for the second one is exactly how this session lost a
// day. Both of the 2-plane defects found were hand-written indices, one of them a second copy of a rule already fixed
// elsewhere.
//
//   ka(e)     the k-atom e belongs to  -- mode2 stride 1, everything else 0, so only the K coordinate survives
//   at_h2(e)  the compact destination  -- mode0 keeps its stride, mode1 (N) gets size<0>, mode2 (K) gets 0, which is
//             exactly the layout of a one-k-atom fragment. THIS IS WHY NAPC > 1 NEEDS NO CALLER ARITHMETIC: which
//             n-atom an output belongs to is something the fragment layout already knows.
template <class FragLayout>
struct ChunkPlace {
  using AtL = decltype(cute::composition(
      cute::make_layout(cute::shape(FragLayout{}),
                        cute::make_stride(cute::stride<0>(FragLayout{}),
                                          cute::Int<cute::size<0>(FragLayout{})>{}, cute::_0{})),
      cute::right_inverse(FragLayout{})));
  using KaL = decltype(cute::composition(
      cute::make_layout(cute::shape(FragLayout{}),
                        cute::make_stride(cute::repeat_like(cute::stride<0>(FragLayout{}), cute::_0{}),
                                          cute::_0{}, cute::_1{})),
      cute::right_inverse(FragLayout{})));
  static constexpr int ka(int e)    { return int(KaL{}(e)); }
  static constexpr int at_h2(int e) { return int(AtL{}(e)) / 2; }
};

// Bias: the integer subtracted from every code, folded into the converter's additive constant instead of costing a
// separate pass. int4's -8 (add_bias_and_interleave_int4s pre-adds 8 offline) has always used this; it is now a
// PARAMETER so a symmetric format can name its own centre and become ScaleOnly. Default reproduces the old value
// exactly, so nothing changes unless a caller asks.
//
// Q3_K's int2 plane is the case that motivates it: its dequant is dl*(low2 - 4) + 4*dl*high1, so with Bias=4 the
// int2 plane's zero becomes 0 and the plane needs no zero channel at all. Plan #20 phase 1.
template <int Bits, int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::Shape<cute::_2,cute::_2,cute::_2>, cute::_4, cute::_4>,
                                          cute::Stride<cute::Stride<cute::_1,cute::_2,cute::_4>, cute::_32, cute::_8>>,
          int Bias = (Bits == 4) ? 8 : 0>
struct MixGemmChunkEmit {
  static constexpr int kCodesPerVreg = 32 / Bits;
  static constexpr int kOut          = 4 * kCodesPerVreg;      // 128 for int1, 64 for int2
  static constexpr int kPairsPerVreg = kCodesPerVreg / 2;
  static constexpr int kPerLevel     = 8 / Bits;
  static constexpr int kPer          = kOut / NChunk;
  static_assert(Bits == 1 || Bits == 2 || Bits == 4, "MixGemmChunkEmit covers int1, int2 and int4");
  // int4 CARRIES A BIAS. add_bias_and_interleave_int4s pre-adds 8 offline (signed -> unsigned), so the converter must
  // subtract it. That is the ONLY thing that used to make int4 "a different scheme": its mask and mul already match
  // this rule exactly. With x = 1024 + c*2^bpos in the fp16 mantissa and mul = 2^-bpos,
  //     y = 1024*2^-bpos + c + add        so   add = -(2^(10-bpos) + Bias)
  // and 2^(10-bpos) + 8 = 2^(10-bpos) * (1 + 2^(bpos-7)), i.e. exponent field 25-bpos with mantissa 2^(bpos+3). The
  // two int4 constants that falls out of -- 0xE408 (-1032) at bpos 0 and 0xD480 (-72) at bpos 4 -- are exactly the
  // FP16_TOP_MAGIC_NUM/NEG_72 pair the shipped int4 converter hardcodes. Gated in fold_derivation/l65.
  static constexpr int kBias = Bias;
  static_assert(NChunk >= 1 && kOut % NChunk == 0, "MixGemmChunkEmit: NChunk must divide the output count");
  static_assert(Chunk < NChunk, "MixGemmChunkEmit: Chunk out of range");
  static_assert(Chunk < 0 || cute::size(FragLayout{}) == kOut,
                "MixGemmChunkEmit: FragLayout must cover all outputs when chunking");

  using Place = ChunkPlace<FragLayout>;
  using AtL = typename Place::AtL;                 // names kept: several harnesses print them
  using KaL = typename Place::KaL;

  static constexpr bool keep(int t, int v) {
    return Chunk < 0 || Place::ka(MixGemmEmit<Bits>::index(t, v)) == Chunk;
  }
  static constexpr int at(int t, int v) {
    const int e = MixGemmEmit<Bits>::index(t, v);
    return (Chunk < 0 || !Rebase) ? e / 2 : Place::at_h2(e);
  }
  static constexpr int kHalf2 = (Chunk < 0 || !Rebase ? kOut : kPer) / 2;

  static constexpr uint32_t dup(uint32_t h) { return h | (h << 16); }   // also used by MixGemm2Plane
  static  constexpr int bpos_of(int t)             { return (t % kPerLevel) * Bits; }
  template <int T> static constexpr int      bpos() { return bpos_of(T); }
  template <int T> static constexpr uint32_t mask() { return dup(uint32_t(((1u << Bits) - 1u) << bpos<T>())); }
  template <int T> static constexpr uint32_t mul()  { return dup(uint32_t((15 - bpos<T>()) << 10)); }
  // add = -(2^(10-bpos) + kBias), as an fp16 bit pattern: sign, exponent field 25-bpos, mantissa field kBias<<bpos.
  // The old form wrote the mantissa as 1<<(bpos+3), which is 8<<bpos -- the kBias=8 special case. Exactness needs
  // the mantissa to fit its 10 bits, i.e. kBias << bpos_max < 1024, which is asserted rather than assumed.
  //   kBias=8, Bits=4  bpos 0,4     -> 0x6408 / 0x5480          (the shipped int4 constants, unchanged)
  //   kBias=4, Bits=2  bpos 0,2,4,6 -> 0x6404 / 0x5C10 / 0x5440 / 0x4D00
  static_assert(kBias == 0 || (uint32_t(kBias) << (8 - Bits)) < 1024u,
                "Bias is not exactly representable in the fp16 additive constant at the top bpos");
  template <int T> static constexpr uint32_t add()  {
    return dup(uint32_t(0x8000u | ((25 - bpos<T>()) << 10) | (uint32_t(kBias) << bpos<T>())));
  }

  template <int T, int V>
  CUTLASS_DEVICE static void emit_one(uint32_t reg, uint32_t* h2) {
    const uint32_t src = (T / kPerLevel) ? (reg >> 8) : reg;
    uint32_t x;
    asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n"
                 : "=r"(x) : "r"(src), "n"(mask<T>()), "n"(0x64006400u), "n"(0xEAu));
    asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n"
                 : "=r"(x) : "r"(x), "r"(mul<T>()), "r"(add<T>()));
    h2[at(T, V)] = x;
  }

  template <int V>
  CUTLASS_DEVICE static void emit_v(uint32_t reg, uint32_t* h2) {
    cute::for_each(cute::make_int_sequence<kPairsPerVreg>{}, [&] (auto t) {
      constexpr int T = decltype(t)::value;
      if constexpr (keep(T, V)) emit_one<T, V>(reg, h2);
    });
  }

  CUTLASS_DEVICE static void emit(uint32_t const* s, uint32_t* h2) {
    emit_v<0>(s[0], h2); emit_v<1>(s[1], h2); emit_v<2>(s[2], h2); emit_v<3>(s[3], h2);
  }
};


template <typename T, typename S, int N>
struct MixGemmNumericArrayConverter
{
};

template <>
struct MixGemmNumericArrayConverter<half_t, int8_t, 4>
{
    using result_type = Array<half_t, 4>;
    using source_type = Array<int8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        uint32_t* h = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

        static constexpr uint32_t mask_for_elt_01 = 0x5250;
        static constexpr uint32_t mask_for_elt_23 = 0x5351;
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
        asm volatile("ppu.prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
        asm volatile("ppu.prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

        // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed integer as fp16.
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
        asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <>
struct MixGemmNumericArrayConverter<bfloat16_t, int8_t, 4>
{
    using result_type = Array<bfloat16_t, 4>;
    using source_type = Array<int8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

        uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

        static constexpr uint32_t fp32_base = 0x4B000000;
        float fp32_intermediates[4];

        // Construct FP32s, bfloat does not have enough mantissa for IADD trick
        uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
        fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
        fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7652);
        fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7651);
        fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

        // Subtract out fp32_base + 128 to make the unsigned integer signed.
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 4; ++ii)
        {
            fp32_intermediates[ii] -= 8388736.f;
        }

        // Truncate the fp32 representation and pack up as bfloat16s.
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 2; ++ii)
        {
            bf16_result_ptr[ii]
                = __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
        }
#else
        result.clear(); // Suppress compiler warning
#endif
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// MixGemmNumericArrayConverter for u8 -> fp16 on AIU
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DstType>
struct MixGemmNumericArrayConverter<DstType, int8_t, 16> {
  using result_type = Array<DstType, 16>;
  using source_type = Array<int8_t, 16>;

  static_assert(platform::is_same<DstType, half_t>::value || platform::is_same<DstType, bfloat16_t>::value,
                "DstType must be fp16 or bf16");

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
      MixGemmNumericArrayConverter<DstType, int8_t, 4>
          convert_vector_;

      result_type result;
      using vec_result = Array<DstType, 4>;
      using vec_source = Array<int8_t, 4>;

      vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);
      vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

      result_ptr[0] = convert_vector_(source_ptr[0]);
      result_ptr[2] = convert_vector_(source_ptr[1]);
      result_ptr[1] = convert_vector_(source_ptr[2]);
      result_ptr[3] = convert_vector_(source_ptr[3]);

      return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) {
      return convert(s);
  }
};

template <>
struct MixGemmNumericArrayConverter<half_t, int4b_t, 8>
{
    using result_type = Array<half_t, 8>;
    using source_type = Array<int4b_t, 8>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        uint32_t* h = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
        static constexpr uint32_t TOP_MASK = 0x00f000f0;
        static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

        // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
        // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
        // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
        // elt_67 to fp16 without having to shift them to the bottom bits before hand.

        // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
        // immediately before required.
        const uint32_t top_i4s = i4s >> 8;
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
        asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[1])
                     : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
        asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[2])
                     : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
        asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[3])
                     : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

        // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
        // half2 ctor. In this case, I chose performance reliability over code readability.

        // This is the half2 {1032, 1032} represented as an integer.
        static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
        // This is the half2 {1 / 16, 1 / 16} represented as an integer.
        static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
        // This is the half2 {-72, -72} represented as an integer.
        static constexpr uint32_t NEG_72 = 0xd480d480;

        // Finally, we construct the output numbers.
        // Convert elt_01
        asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_23
        asm volatile("ppu.fma.rtte.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
        // Convert elt_45
        asm volatile("ppu.sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_67
        asm volatile("ppu.fma.rtte.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <>
struct MixGemmNumericArrayConverter<bfloat16_t, int4b_t, 8>
{
    using result_type = Array<bfloat16_t, 8>;
    using source_type = Array<int4b_t, 8>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

        uint32_t* h = reinterpret_cast<uint32_t*>(&result);
        uint32_t const source_i4s = reinterpret_cast<uint32_t const&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t MASK = 0x000f000f;
        static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

        // We don't have enough mantissa to remove as much shift overhead as FP16, so we must loop.
        // No shift needed for first item.
        uint32_t i4s = source_i4s;
        asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 1; ii < result_type::kElements / 2; ++ii)
        {
            i4s >>= sizeof_bits<typename source_type::Element>::value;
            // (i4s & 0x000f000f) | 0x43004300
            asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                         : "=r"(h[ii])
                         : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        }

        // This is the BF16 {-136, -136} represented as an integer.
        static constexpr uint32_t BF16_BIAS = 0xC308C308;
        static constexpr uint32_t BF16_ONE = 0x3F803F80;

        // Finally, we construct the output numbers.
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < result_type::kElements / 2; ++ii)
        {
            asm("ppu.fma.rtte.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[ii]) : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
        }
#else
        result.clear(); // Suppress compiler warning.
#endif
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <typename DstType>
struct MixGemmNumericArrayConverter<DstType, int4b_t, 32>
{
    using result_type = Array<DstType, 32>;
    using source_type = Array<int4b_t, 32>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        MixGemmNumericArrayConverter<DstType, int4b_t, 8> convert_vector_;

        result_type result;
        using vec_result = Array<DstType, 8>;
        using vec_source = Array<int4b_t, 8>;

        vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        result_ptr[0] = convert_vector_(source_ptr[0]);
        result_ptr[2] = convert_vector_(source_ptr[1]);
        result_ptr[1] = convert_vector_(source_ptr[2]);
        result_ptr[3] = convert_vector_(source_ptr[3]);

        // transform layout
        using vec_trans = Array<DstType, 4>;
        vec_trans* elt_b16 = reinterpret_cast<vec_trans*>(&result);
        vec_trans tmp_b16 = elt_b16[1];
        elt_b16[1] = elt_b16[2];
        elt_b16[2] = tmp_b16;
        tmp_b16 = elt_b16[5];
        elt_b16[5] = elt_b16[6];
        elt_b16[6] = tmp_b16;

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

// ============================ W2A16 : uint2b_t -> fp16 ============================
// Q2 base plane (also the high plane of Q6, low plane of Q3). q2 in [0,3] UNSIGNED (no +bias:
// the per-group affine 'zero' term absorbs the offset). Correctness-first: plain shift-extract,
// NO lop3 magic-trick (perf later). Numeric validated in low_bit/w2a16_smoke.cu (bad=0 on ppu001).
template <>
struct MixGemmNumericArrayConverter<half_t, uint2b_t, 16>   // 16 x uint2 = 32 bits -> 16 half
{
    using result_type = Array<half_t, 16>;
    using source_type = Array<uint2b_t, 16>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
        uint32_t const bits = reinterpret_cast<uint32_t const&>(source);   // 16 x 2-bit, LSB-first
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i)
            result[i] = half_t(float((bits >> (2 * i)) & 0x3u));           // elt i = bits[2i:2i+2]
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s) { return convert(s); }
};

// Wide converter used by convert_tensor (CPY_VEC = 4*32/2 = 64 for uint2). Composes base-16 x4.
template <>
struct MixGemmNumericArrayConverter<half_t, uint2b_t, 64>
{
    using result_type = Array<half_t, 64>;
    using source_type = Array<uint2b_t, 64>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // STAGE-2 PERF: pure lop3 magic (like int4's base-8), zero scalar packing. DEPENDS on the OFFLINE register
        // relayout in add_bias_and_interleave_int2s_inplace (mirror of int4's, split at 8 -> dest d gets src crumb
        // d<8?2d:2(d-8)+1). After that relayout the lop3 h[t]=(crumb 2t, crumb 2t+1) i.e. exactly Stage-1's
        // adjacent pairs, so the N-half placement below is Stage-1's VALIDATED interleave verbatim.
        //   lop3 dst = (src & MASK) | 0x64006400  [immLut 0xEA = (a&b)|c]  -> fp16 1024 + K*crumb, K = mask position
        //   weight, undone by fma(value, 1/K, -1024/K): 0x03->K=1, 0x0c->4, 0x30->16, 0xc0->64.
        //   src0,src1 = low-N (n%16<8), src2,src3 = high-N (n%16>=8) [MEASURED]. For vreg v the 8 pairs (c0,c1)..
        //   (c14,c15) land at h2[base + 4*(t/2) + (t%2)], base = 16*(v&1) + 2*(v>=2). 32 lop3 + 32 fma, no temp.
        // Delegates to MixGemmChunkEmit<2>, whose (mask, mul, add) are DERIVED from Bits. The eight triples that
        // used to live here were the same rule as int1's sixteen at a different stride -- verified identical, 24 rows,
        // 0 differ, in fold_derivation/l36_emit_constants.cu. One rule, one implementation.
        result_type result;
        MixGemmChunkEmit<2, -1, 1>::emit(reinterpret_cast<uint32_t const*>(&source),
                                         reinterpret_cast<uint32_t*>(&result));
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s) { return convert(s); }
};



// ============================ W1A16 : uint1b_t -> fp16 ============================
// Q1 base plane (high plane of Q3/Q5). bit in {0,1} UNSIGNED (affine 'zero' absorbs offset). CORRECTNESS-FIRST
// magic-OR (0x6400|bit == fp16(1024+bit), one vectorized f16x2 sub of 1024) -- NO lop3 yet; optimize after the
// N-half map is validated on ppu001. CPY_VEC = 4*32/1 = 128. N-half by ANALOGY to the (validated) uint2b_t wide
// converter, scaled 2x (int1 packs 32 values/vreg vs int2's 16): src0,src1 = low-N, src2,src3 = high-N; 16 atoms,
// atom a (g=a/8, j=a%8): frag[8a+0..3] <- bits 4j..4j+3 of vreg g (low-N), frag[8a+4..7] <- vreg g+2 (high-N).
// MUST verify with test_w1a16_diag; if the permutation differs, re-derive with the controlled-input probe.
// ================== W1A16 emission, CHUNK-AWARE. One source of truth for chunked and unchunked. ==================
//
// WHY. One swzl delivery is a fixed 16 B, so it carries 16/Bits mma atom-slots of B -- 16 for int1 against int4's 4.
// That is simultaneously int1's advantage (one read feeds 4x the mma; the width-isolation run measured 49.9 / 48.1 /
// 45.9 for int1 / int2 / int4 at one shared config, monotone in bits) and its handicap: the fp16 fragment must hold a
// whole delivery, so MMA_N*MMA_K >= 16/Bits and B_regs = 4*MMA_N*MMA_K >= 64 for int1. Those 64 registers are what
// push the best int1 config from 164 to over the power-of-two billing boundary, costing half the occupancy
// (warps/CU 32 -> 16). Chunking DECOUPLES "how many atom-slots the delivery covers" from "how many are fp16 at
// once": all 16 are still converted and used, in NChunk batches, so B drops to 64/NChunk with no wasted delivery and
// no change to the mma count or total converter work. Only the emission ORDER changes.
//
// THE GATE IS COMPILE-TIME, which is the whole reason MixGemmEmit had to become the emission source first (l29):
// against the old hand-written offset table there would be nothing to gate on. tCrB_mma is compact
// (8, MMA_N, MMA_K), so k-atom a owns the contiguous range [kPer*a, kPer*(a+1)), and from MixGemmEmit<1>
//     e = bit4 + 2*b0 + 8*b1 + 16*b2 + 32*bit3 + 64*(v&1) + 4*(v>>1)
// every term except 32*bit3 and 64*(v&1) is below 32, so at kPer=32 the chunk is exactly
//     e / 32 == bit3(code) + 2*(vreg & 1)
// a STATIC function of (code, vreg). Verified for all three widths and MMA_N in 4/2, with the pairs splitting exactly
// evenly, in fold_derivation/l32_chunk_predicate.cu.
//
// V IS A TEMPLATE PARAMETER, not a loop variable. The first attempt at this kept `for (int v = 0; v < 4; ++v)` and
// gated with `if constexpr`, which cannot depend on a runtime variable. Relying on a plain `if` plus unrolling and
// dead-code elimination would have made the register saving depend on the compiler folding the branches -- exactly
// the assumption the scale-broadcast episode punished. So the per-vreg emission is templated instead.
// Rebase=false writes at the FULL output index instead of the chunk-local one. That is the bisection mode: chunked
// EMISSION into the whole fragment, so a mismatch localises to the gating rather than to the small buffer.
template <>
struct MixGemmNumericArrayConverter<half_t, uint1b_t, 128>
{
    using result_type = Array<half_t, 128>;
    using source_type = Array<uint1b_t, 128>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        // STAGE-2: pure lop3 magic (mirror of int2's, scaled to 1-bit / 32-bit-register). DEPENDS on the OFFLINE
        // split-at-16 relayout in add_bias_and_interleave_int1s (unfused_weight_dequantize.hpp): after it, lop3
        // h[t]=(bit 2t, bit 2t+1) = the validated (magic-OR) adjacent pairs. Per vreg: 8 masks (1<<b)|(1<<(16+b))
        // b=0..7, x2 levels (reg / reg>>8); bit lands at fp16 mantissa bit b -> value=1024+2^b*bit, undone by
        // fma(2^-b, -2^(10-b)). N-half placement = validated Stage-1: base = 32*(v&1) + 2*(v>=2); pair t -> h2[base
        // + {0,1,4,5,8,9,12,13}[t%8] + 16*(t>=8)].
        // Delegates to MixGemmInt1Emit<-1,1> so the chunked and unchunked paths cannot drift apart. The
        // emission comment lives on that struct.
        result_type result;
        MixGemmChunkEmit<1, -1, 1>::emit(reinterpret_cast<uint32_t const*>(&source),
                                         reinterpret_cast<uint32_t*>(&result));
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s) { return convert(s); }
};

// ================= TWO-PLANE (bit-plane concat) convert: low plane + high plane -> ONE fp16 =================
// Produces fp16 of the COMBINED integer  q = low + 2^bl * high  (Q3: bl=2 -> q in 0..7). The collective's affine
// then applies dl*q + zero, with the format's center folded into zero (Q3_K: zero = -4*dl).
//
// The concat MUST happen in the fp16 domain, not the packed domain: the low plane's codes are densely packed at
// bl-bit spacing, so there is no bit gap to OR the high bit into (leaving one == padding, which is the route we
// rejected because it costs 33% memory).
//
// Both planes keep their ALREADY-VALIDATED offline relayouts -- no new bit derivation is needed. But BEWARE the
// trap that cost a full debug cycle here: scratchpad/xplane.py verified (bad=0/4096) the relation
//     high-plane MEMORY vector j1 == low-plane MEMORY vector j2 >> 1 ;  pair index += 8*(j2 & 1)
// in MEMORY-VECTOR space. The converter's `v` is the SWZL-DELIVERED register index, and swzl permutes, so j2 != v.
// Applying the memory-space relation to v is what produced the 50%-mis-sourced bug (rung8: 25% of recovered high
// bits wrong under a random witness == 50% mis-sourced, flat across every n%16 and k%16, no k/n permutation fitting).
//
// The truth for DELIVERED indices comes from aligning the two VALIDATED single-plane converters, whose output h2
// index is their ground truth:
//     int2:  o2(v,t) = 16*(v&1) + 2*(v>=2) + off8[t]                    , o2 in [0,32)  = ONE k_block
//     int1:  o1(v,t) = 32*(v&1) + 2*(v>=2) + off8[t%8] + 16*(t>=8)      , o1 in [0,64)  = TWO k_blocks
// int1's 64 h2 span exactly twice int2's mma-fragment range, so int1's 32*(v&1) IS the k_block selector. Requiring
// both planes to land on the same h2 index within a k_block gives, for low vreg v in low k_block kb:
//     (v_hi & 1) = kb ,  (v_hi >= 2) = (v >= 2) ,  t_hi = t + 8*(v & 1)
//   => v_hi = kb + 2*(v >> 1)            (NOT 2*kb + (v>>1), which is half wrong in BOTH k_blocks)
// so the collective offsets the high pointer by kb (stride 1) and this converter indexes hi[2*(v>>1)] (stride 2).
//
// Placement trick that keeps ONE correction per half2: the low code sits at mantissa base b = 2*(t%4) (the four
// int2 masks 0x03/0x0c/0x30/0xc0), so the high bit is placed at mantissa b+2, i.e. it contributes exactly
// 4 * 2^b == 2^bl * 2^b. The single per-mask fma (2^-b, -2^(10-b)) then yields (low + 4*high) directly.
// CHUNKING (task #12): Chunk >= 0 emits ONLY the half2 belonging to k-atom `Chunk`, into a 4-half2 one-atom buffer,
// so the mainloop can hold 4*MMA_N fp16 registers for B instead of 4*MMA_N*MMA_K. Chunk = -1 is the full delivery and
// is bit-identical to the version that shipped -- the eight destinations below used to be hardcoded as
// h2[base + {0,1,4,5,8,9,12,13}] with base = 16*(v&1) + 2*(v>=2), and AtLayout reproduces all 32 of them exactly
// (fold_derivation/l41_2plane_chunk_gate.cu, 0/32 differ).
//
// WHY THE GATE IS A DIVISION AND NOT A right_inverse COMPOSITION (unlike the single-plane MixGemmChunkEmit). l40
// derived the real layouts from the builder's own type algebra and found cvt_in's mode-1 stride (128) EQUAL to
// tCrB_mma's MMA_N stride (128) at the 2-plane's TileShape, so the flat pointer write the collective already does is
// an exact partition of the fragment: per `ii` the converter fills 64 CONTIGUOUS fp16 = 8 k-atoms of ONE n, with k
// inner. Hence k-atom == at_plain/4 and the in-atom slot == at_plain%4. The single-plane fold path needs the
// composition because there MmaPermK = TileShape.K puts the fragment in a different mode order; here MmaPermK is the
// 32 B run span (ppu_mma_builder.inl:588). That one constant is the whole difference.
// TWO BIT PLANES INTO ONE fp16 FRAGMENT, for any (low, high) width pair. Q3 = int2+int1, Q6 = int4+int2,
// Q5 = int4+int1. Everything below is ONE closed form; there is no per-format index algebra, because writing a second
// and third hand-derived pairing is precisely what cost this session a day.
//
// THE ARITHMETIC. The low plane's crumb lands in the fp16 mantissa at position bpos<T> exactly as in a single-plane
// conversion, and the high plane's bits are OR'd LowBits positions above it, so the mantissa holds
// lo + 2^LowBits * hi. The (mask, mul, add) that follow are therefore the LOW plane's own -- MixGemmChunkEmit<LowBits>
// -- unchanged, including int4's +8 bias removal. Nothing about the second plane touches them.
//
// THE PAIRING, derived once. A delivery is 16 B/thread either way, so the low plane brings 128/LowBits codes in 4 vregs
// while the high plane's 16 B covers 128/HiBits of them, i.e. P2_DIV = LowBits/HiBits times what one low delivery needs.
// Inside one high vreg the two half2 lanes must sit 16 bits apart, so a pair of high codes is (a, a + 16/HiBits), and
//     idx(v_local, c) = (c % kPairs) + kPairs*v_local + (16/HiBits)*(c / kPairs),   kPairs = 16/LowBits
// is a bijection onto that vreg's codes because kPairs * P2_DIV == 16/HiBits. Hence
//     hshift(T, V) = HiBits * (T + kPairs * (V % P2_DIV))        hi vreg = P2_DIV * (V / P2_DIV)
//
// GATED AGAINST SHIPPED CODE, which is the only reason to trust it: at (2,1) the form reproduces Q3's hand-written
// constants exactly -- hshift == 8*(V&1)+T, hi vreg == 2*(V>>1), and at_plain == MixGemmEmit<2>::index/2, so even the
// hand-written AtLayout turns out to be the low plane's own emission. It is a bijection for Q6 and Q5 too.
// fold_derivation/l65, five checks.
//
// FragLayout is the DELIVERY's view of tCrB_mma -- shape (mode0, NAPC, KAPC), strides taken from tCrB_mma -- so `Chunk`
// indexes the k-atom and which n-atom an output belongs to is resolved by the LAYOUT, never by the caller.
template <int LowBits, int HiBits, int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::Int<4 * (16 / LowBits) / 4>>>>
struct MixGemm2Plane
{
    static_assert(LowBits == 2 || LowBits == 4, "low plane is int2 (Q3) or int4 (Q5/Q6)");
    static_assert(HiBits == 1 || HiBits == 2, "high plane is int1 (Q3/Q5) or int2 (Q6)");
    static_assert(LowBits > HiBits && LowBits % HiBits == 0, "the high plane must be the sparser one");

    static constexpr int kCodesPerVreg = 32 / LowBits;        // low codes in one vreg    (int2 16, int4 8)
    static constexpr int kPairs        = kCodesPerVreg / 2;   // half2 pairs per vreg     (int2 8,  int4 4)
    static constexpr int kOut          = 4 * kCodesPerVreg;   // fp16 outputs per delivery(int2 64, int4 32)
    static constexpr int kPerAtom      = 4;                   // half2 per mma B k-atom (4 half2 == 8 fp16)
    static constexpr int kAtoms        = (kOut / 2) / kPerAtom;// mma atoms one delivery covers (int2 8, int4 4)
    // THE VREG RATIO INSIDE ONE DELIVERY -- deliberately NOT called P2_DIV, which in the collective means the COPY STEP
    // ratio DL1/DL2. The two coincide only when neither plane folds, and conflating them broke the first version of the
    // offline generalisation while leaving Block_K=256 (where they agree) passing.
    static constexpr int kVregRatio    = LowBits / HiBits;
    static constexpr int kHiStride     = 16 / HiBits;          // high codes between the two half2 lanes
    static constexpr int kHiVregs      = 4 * HiBits / LowBits; // high vregs one low delivery consumes
    static constexpr int P2_DIV        = kVregRatio;           // old name, kept for the harnesses that print it
    static constexpr int kPerLevel     = 8 / LowBits;          // low codes per byte level
    static constexpr int kLowBitsPub   = LowBits;              // so host-side emulation can read it back
    static_assert(Chunk < NChunk, "MixGemm2Plane: Chunk out of range");
    static_assert(Chunk < 0 || cute::size(FragLayout{}) == kOut,
                  "FragLayout must cover exactly one delivery's outputs");

    // (mask, mul, add, bpos) are the LOW plane's, unchanged. ONE rule, shared with the single-plane converter.
    using E = MixGemmChunkEmit<LowBits, -1, 1>;
    // Placement is the low plane's own emission, composed with the fragment layout by the SAME ChunkPlace the
    // single-plane path uses. The emission ORDERS differ between formats; the placement rule does not.
    using Place = ChunkPlace<FragLayout>;
    static constexpr int  at_plain(int t, int v) { return MixGemmEmit<LowBits>::index(t, v) / 2; }
    static constexpr bool keep(int t, int v)     { return Chunk < 0 || Place::ka(2 * at_plain(t, v)) == Chunk; }
    static constexpr int  at(int t, int v)       { return Chunk < 0 ? at_plain(t, v) : Place::at_h2(2 * at_plain(t, v)); }
    // A chunk touches only some of the four vregs (l41), so the others are not even READ -- gate the vreg, not just its
    // lines, or a chunked pass still pays for all four lo[] loads and their hi[] companions.
    static constexpr bool vreg_used(int v) {
      for (int t = 0; t < kPairs; ++t) if (keep(t, v)) return true;
      return false;
    }
    // (g) THE PAIRING AS LAYOUTS, and as the SINGLE definition both sides use. The converter used to state it as
    // arithmetic while the offline (xplane::tile_map_hi) stated the SAME rule as cute Layouts -- two forms of one rule,
    // which is exactly the class that produced both of this session's 2-plane defects: hi_vreg0 was one rule in two
    // copies with only one fixed, and PDcopy/kVregRatio were two quantities sharing a name. So the layouts live here and
    // the offline reads them.
    //
    //   LoCodeL : (T, half)     -> the LOW code index inside a low vreg          T + kPairs*half
    //   HiCodeL : (T, V, half)  -> the HIGH code index inside its high vreg      T + kPairs*(V % VR) + hstride*half
    //   HVregL  : (V)           -> which high vreg low vreg V reads               VR * (V / VR)
    //
    // V's two roles are the nested mode (VR, 4/VR): the low part selects which of the VR low vregs share a high vreg
    // (stride kPairs in HiCodeL, 0 in HVregL) and the high part selects the high vreg (stride 0 in HiCodeL, VR in
    // HVregL). At (2,1) these evaluate to T + 8*(V&1) and 2*(V>>1) -- Q3's shipped constants (fold_derivation/l65).
    using LoCodeL = cute::Layout<cute::Shape <cute::Int<kPairs>, cute::_2>,
                                 cute::Stride<cute::_1,          cute::Int<kPairs>>>;
    using HiCodeL = cute::Layout<cute::Shape <cute::Int<kPairs>, cute::Shape<cute::Int<kVregRatio>, cute::Int<4 / kVregRatio>>, cute::_2>,
                                 cute::Stride<cute::_1,          cute::Stride<cute::Int<kPairs>,    cute::_0>,                 cute::Int<16 / HiBits>>>;
    using HVregL  = cute::Layout<cute::Shape <cute::Shape<cute::Int<kVregRatio>, cute::Int<4 / kVregRatio>>>,
                                 cute::Stride<cute::Stride<cute::_0,             cute::Int<kVregRatio>>>>;
    static constexpr int  lo_code(int t, int half)        { return int(LoCodeL{}(t, half)); }
    static constexpr int  hi_code(int t, int v, int half) { return int(HiCodeL{}(t, v, half)); }
    static constexpr int  hi_vreg(int v)                  { return int(HVregL{}(v)); }
    // the converter needs the same thing in BITS, for lane 0; the dup'd mask reaches lane 1 sixteen bits up, which is
    // hstride codes and therefore HiCodeL's `half` mode -- so this is one layout, not two.
    static constexpr int  hshift(int t, int v) { return HiBits * hi_code(t, v, 0); }
    static constexpr uint32_t himask() { return E::dup(uint32_t((1u << HiBits) - 1u)); }

    // THE HIGH PLANE'S BITS, LowBits mantissa positions above the low plane's. Written the plain way on purpose.
    //
    // I replaced this with an explicit single-shift + second lop3 (immLut 0xF8 == a | (b & c)) on the theory that the
    // source form costs 6 machine operations against a single plane's 2. It costs 4, because the compiler already
    // composes the two compile-time shifts and already fuses the and/or into a lop3: nvcc -arch=sm_80 -O3 gives
    // BYTE-IDENTICAL SASS for both forms -- LOP3=4, SHIFT=2, 32 instructions. So the explicit version was zero gain and
    // more code, and it is reverted. (git history has it if a future acu ever shows hgcc failing to fuse.)
    //
    // What survives from that analysis is the real number: at SASS level a 2-plane half2 costs
    //     lop3 + shift + lop3 + fma = 4      against a single plane's      lop3 + fma = 2
    // a structural 2x, and it does not come down. The first lop3's three operands are already src, mask and the 0x6400
    // base, and the high plane's bits live in a DIFFERENT register, so they must be shifted and OR'd. Removing the shift
    // would mean storing the high plane pre-shifted to its destination bit position, which wastes field width in a 1-2
    // bit plane and buys the ALU back in traffic. So the Q3/Q5/Q6 residual over a single int4 GEMM (1.15x / 1.18x /
    // 1.24x, ordered by total bit width) is NOT reducible by cutting converter instructions; the remaining axis is
    // amortisation, cvt/mma, and WM=128 is out because accum = WM*WN/32 would be 256 registers.
    template <int T, int V>
    CUTLASS_DEVICE static void emit_one(uint32_t reg, uint32_t r8, uint32_t hreg, uint32_t* h2) {
      const uint32_t src = (T / kPerLevel) ? r8 : reg;        // the upper byte level of the low vreg
      uint32_t x;
      asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(x)
                   : "r"(src), "n"(E::template mask<T>()), "n"(0x64006400u), "n"(0xEAu));
      x |= ((hreg >> hshift(T, V)) & himask()) << (E::template bpos<T>() + LowBits);
      asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n" : "=r"(x)
                   : "r"(x), "r"(E::template mul<T>()), "r"(E::template add<T>()));
      h2[at(T, V)] = x;
    }

    template <int V>
    CUTLASS_DEVICE static void emit_v(uint32_t const* lo, uint32_t const* hi, uint32_t* h2) {
      const uint32_t reg = lo[V], r8 = reg >> 8;
      // hi already points at the high fragment offset by this k_block's parity (HiPlaneSrc in the collective); the
      // vregs one k_block owns are P2_DIV apart, so this picks the one serving low vreg V.
      const uint32_t hreg = hi[hi_vreg(V)];                   // (g) HVregL, shared with the offline
      cute::for_each(cute::make_int_sequence<kPairs>{}, [&] (auto t) {
        constexpr int T = decltype(t)::value;
        if constexpr (keep(T, V)) emit_one<T, V>(reg, r8, hreg, h2);
      });
    }

    // lo: the delivery's 4 low vregs. hi: base already offset by the low k_block parity.
    // out: kOut/2 half2 for Chunk < 0, else kPerAtom half2 for k-atom `Chunk`.
    CUTLASS_DEVICE
    static void convert(uint32_t const* lo, uint32_t const* hi, uint32_t* h2)
    {
      cute::for_each(cute::make_int_sequence<4>{}, [&] (auto v) {
        constexpr int V = decltype(v)::value;
        if constexpr (vreg_used(V)) emit_v<V>(lo, hi, h2);
      });
    }
};

// Q3's name, kept so nothing downstream changes. Q6 and Q5 are the same object at different widths.
template <int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_8>>>
using MixGemm2Plane_uint2_uint1 = MixGemm2Plane<2, 1, Chunk, NChunk, Rebase, FragLayout>;
template <int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_4>>>
using MixGemm2Plane_int4_uint2 = MixGemm2Plane<4, 2, Chunk, NChunk, Rebase, FragLayout>;   // Q6
template <int Chunk = -1, int NChunk = 1, bool Rebase = true,
          class FragLayout = cute::Layout<cute::Shape<cute::_8, cute::_1, cute::_4>>>
using MixGemm2Plane_int4_uint1 = MixGemm2Plane<4, 1, Chunk, NChunk, Rebase, FragLayout>;   // Q5

// The collective picks the pair from the two plane ELEMENTS, so a new format needs no dispatch table.
template <class LowElem, class HiElem, int Chunk = -1, int NChunk = 1, bool Rebase = true, class FragLayout = void>
struct MixGemm2PlaneFor {
  using type = MixGemm2Plane<cutlass::sizeof_bits<LowElem>::value, cutlass::sizeof_bits<HiElem>::value,
                             Chunk, NChunk, Rebase, FragLayout>;
};

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
