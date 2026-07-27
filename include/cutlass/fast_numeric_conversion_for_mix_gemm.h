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

  // (code within a vreg, vreg) -> fp16 element index within the converter's 4*kCodesPerVreg outputs
  static constexpr int index(int code, int vreg) {
    int e = ((code >> (kNumCodeBits - 1)) & 1)                  // the half2 lo/hi bit -> weight 1
          + (2 * kCodesPerVreg) * (vreg & 1)                    // vreg low  bit -> top weight
          + 4 * (vreg >> 1);                                    // vreg high bit -> weight 4
    for (int i = 0; i < kNumCodeBits - 1; ++i)
      e += ((code >> i) & 1) * (i == 0 ? 2 : (4 << i));         // 2, 8, 16, 32, ...
    return e;
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
        result_type result;
        uint32_t const* s  = reinterpret_cast<uint32_t const*>(&source);   // 4 swzl vregs, 16 crumbs each
        uint32_t*       h2 = reinterpret_cast<uint32_t*>(&result);         // 32 half2 (uint32 view)
        #define _E(dst, src, MASK, MUL, ADD) do {                                                         \
            uint32_t _x;                                                                                  \
            asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(_x) : "r"(src), "n"(MASK), "n"(0x64006400u), "n"(0xEAu)); \
            asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n" : "=r"(_x) : "r"(_x), "r"((uint32_t)(MUL)), "r"((uint32_t)(ADD))); \
            (dst) = _x; } while (0)
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < 4; ++v) {
          uint32_t reg = s[v], r8 = reg >> 8;
          // The h2 index is DERIVED now: MixGemmEmit<2>::index(t, v)/2 equals the hardcoded
          // base + {0,1,4,5,8,9,12,13}[t] it replaces, 0 mismatch over all (t, v) -- see
          // fold_derivation/l29_emit_is_converter.cu. The MASKS stay literal on purpose: they encode a code's BIT
          // POSITION, which is a value transform, and a layout describes where a code goes, not what it becomes.
          _E(h2[MixGemmEmit<2>::index(0, v) / 2], reg, 0x00030003u, 0x3c003c00u, 0xe400e400u);  // pair0 (c0,c1)   K=1  : v-1024
          _E(h2[MixGemmEmit<2>::index(1, v) / 2], reg, 0x000c000cu, 0x34003400u, 0xdc00dc00u);  // pair1 (c2,c3)   K=4  : v/4-256
          _E(h2[MixGemmEmit<2>::index(2, v) / 2], reg, 0x00300030u, 0x2c002c00u, 0xd400d400u);  // pair2 (c4,c5)   K=16 : v/16-64
          _E(h2[MixGemmEmit<2>::index(3, v) / 2], reg, 0x00c000c0u, 0x24002400u, 0xcc00cc00u);  // pair3 (c6,c7)   K=64 : v/64-16
          _E(h2[MixGemmEmit<2>::index(4, v) / 2], r8,  0x00030003u, 0x3c003c00u, 0xe400e400u);  // pair4 (c8,c9)
          _E(h2[MixGemmEmit<2>::index(5, v) / 2], r8,  0x000c000cu, 0x34003400u, 0xdc00dc00u);  // pair5 (c10,c11)
          _E(h2[MixGemmEmit<2>::index(6, v) / 2], r8,  0x00300030u, 0x2c002c00u, 0xd400d400u);  // pair6 (c12,c13)
          _E(h2[MixGemmEmit<2>::index(7, v) / 2], r8,  0x00c000c0u, 0x24002400u, 0xcc00cc00u);  // pair7 (c14,c15)
        }
        #undef _E
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
        result_type result;
        uint32_t const* s  = reinterpret_cast<uint32_t const*>(&source);   // 4 swzl vregs, 32 bits each
        uint32_t*       h2 = reinterpret_cast<uint32_t*>(&result);         // 64 half2 (uint32 view)
        #define _E(dst, src, MASK, MUL, ADD) do {                                                        \
            uint32_t _x;                                                                                 \
            asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(_x) : "r"(src), "n"(MASK), "n"(0x64006400u), "n"(0xEAu)); \
            asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n" : "=r"(_x) : "r"(_x), "r"((uint32_t)(MUL)), "r"((uint32_t)(ADD))); \
            (dst) = _x; } while (0)
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < 4; ++v) {
          uint32_t reg = s[v], r8 = reg >> 8;
          // N-GROUP ROUTING, bases {0,32,2,34}. Do NOT read this as "the converter can only reach 2 N-groups, so
          // a 4-way variant would unlock the F=4 fold" -- that theory was tested and is FALSE. The fold factor is
          // capped at 2 by the swzl delivery grid, not by anything here; see fold_traits.hpp for the derivation
          // (scratchpad/leg1_runword.cpp + leg2_frag.cu). The short version: the two extra columns of an F=4 run
          // are delivered to DIFFERENT LANES, and a converter only relabels registers inside one thread.
          // Derived: MixGemmEmit<1>::index(t, v)/2 equals base + {0,1,4,5,8,9,12,13}[t%8] + 16*(t>=8).
          // NOTE the comment ABOVE this struct describes the STAGE-1 magic-OR implementation, not this one --
          // encoding it instead of this code cost 120/128 mismatches on l29's first run.
          //         dst              src  MASK          MUL(2^-b)     ADD(-2^(10-b))   b : mantissa bit, K=2^b
          _E(h2[MixGemmEmit<1>::index(0, v) / 2], reg, 0x00010001u, 0x3c003c00u, 0xe400e400u); // b0 pair0
          _E(h2[MixGemmEmit<1>::index(1, v) / 2], reg, 0x00020002u, 0x38003800u, 0xe000e000u); // b1 pair1
          _E(h2[MixGemmEmit<1>::index(2, v) / 2], reg, 0x00040004u, 0x34003400u, 0xdc00dc00u); // b2 pair2
          _E(h2[MixGemmEmit<1>::index(3, v) / 2], reg, 0x00080008u, 0x30003000u, 0xd800d800u); // b3 pair3
          _E(h2[MixGemmEmit<1>::index(4, v) / 2], reg, 0x00100010u, 0x2c002c00u, 0xd400d400u); // b4 pair4
          _E(h2[MixGemmEmit<1>::index(5, v) / 2], reg, 0x00200020u, 0x28002800u, 0xd000d000u); // b5 pair5
          _E(h2[MixGemmEmit<1>::index(6, v) / 2], reg, 0x00400040u, 0x24002400u, 0xcc00cc00u); // b6 pair6
          _E(h2[MixGemmEmit<1>::index(7, v) / 2], reg, 0x00800080u, 0x20002000u, 0xc800c800u); // b7 pair7
          _E(h2[MixGemmEmit<1>::index(8, v) / 2], r8,  0x00010001u, 0x3c003c00u, 0xe400e400u); // b0 pair8
          _E(h2[MixGemmEmit<1>::index(9, v) / 2], r8,  0x00020002u, 0x38003800u, 0xe000e000u); // b1 pair9
          _E(h2[MixGemmEmit<1>::index(10, v) / 2], r8,  0x00040004u, 0x34003400u, 0xdc00dc00u); // b2 pair10
          _E(h2[MixGemmEmit<1>::index(11, v) / 2], r8,  0x00080008u, 0x30003000u, 0xd800d800u); // b3 pair11
          _E(h2[MixGemmEmit<1>::index(12, v) / 2], r8,  0x00100010u, 0x2c002c00u, 0xd400d400u); // b4 pair12
          _E(h2[MixGemmEmit<1>::index(13, v) / 2], r8,  0x00200020u, 0x28002800u, 0xd000d000u); // b5 pair13
          _E(h2[MixGemmEmit<1>::index(14, v) / 2], r8,  0x00400040u, 0x24002400u, 0xcc00cc00u); // b6 pair14
          _E(h2[MixGemmEmit<1>::index(15, v) / 2], r8,  0x00800080u, 0x20002000u, 0xc800c800u); // b7 pair15
        }
        #undef _E
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
struct MixGemm2Plane_uint2_uint1
{
    // lo: 4 swzl vregs (64 crumbs). hi: base already offset by the low k_block parity; this k_block's two vregs
    // are STRIDE 2 apart, so low vreg v uses hi[2*(v>>1)], half selected by v&1 (see the derivation above).
    // out: 32 half2 (= 64 fp16), laid out exactly like the validated single-plane uint2 converter.
    CUTLASS_DEVICE
    static void convert(uint32_t const* lo, uint32_t const* hi, uint32_t* h2)
    {
        #define _E2(dst, losrc, MASK, MUL, ADD, HISRC, HISH, HIPOS) do {                                   \
            uint32_t _x;                                                                                   \
            asm volatile("ppu.lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(_x)                                       \
                         : "r"(losrc), "n"(MASK), "n"(0x64006400u), "n"(0xEAu));                           \
            _x |= (((HISRC) >> (HISH)) & 0x00010001u) << (HIPOS);   /* high bit -> mantissa b+2 */          \
            asm volatile("ppu.fma.rtte.f16x2 %0,%1,%2,%3;\n" : "=r"(_x)                                     \
                         : "r"(_x), "r"((uint32_t)(MUL)), "r"((uint32_t)(ADD)));                            \
            (dst) = _x; } while (0)
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < 4; ++v) {
          uint32_t reg = lo[v], r8 = reg >> 8;
          // WHICH high vreg serves low vreg v -- DERIVED from the two validated single-plane converters (see the
          // header block above), not guessed. hi points at the high fragment already offset by the low k_block
          // parity, and the two vregs this k_block owns are STRIDE 2 apart, so hi[2*(v>>1)] == absolute vreg
          // kb + 2*(v>=2).  hs is unchanged: pair index t + 8*(v&1) is what the alignment yields.
          uint32_t hreg = hi[2 * (v >> 1)];
          int hs   = 8 * (v & 1);
          int base = 16 * (v & 1) + 2 * (v >= 2); // same N-half placement as the single-plane uint2 converter
          //   pair t : low mask (mantissa base b=2*(t%4)), level reg/r8 for t<4 / t>=4, high bit at b+2
          _E2(h2[base + 0 ], reg, 0x00030003u, 0x3c003c00u, 0xe400e400u, hreg, hs + 0, 2);   // t=0 b=0
          _E2(h2[base + 1 ], reg, 0x000c000cu, 0x34003400u, 0xdc00dc00u, hreg, hs + 1, 4);   // t=1 b=2
          _E2(h2[base + 4 ], reg, 0x00300030u, 0x2c002c00u, 0xd400d400u, hreg, hs + 2, 6);   // t=2 b=4
          _E2(h2[base + 5 ], reg, 0x00c000c0u, 0x24002400u, 0xcc00cc00u, hreg, hs + 3, 8);   // t=3 b=6
          _E2(h2[base + 8 ], r8,  0x00030003u, 0x3c003c00u, 0xe400e400u, hreg, hs + 4, 2);   // t=4 b=0
          _E2(h2[base + 9 ], r8,  0x000c000cu, 0x34003400u, 0xdc00dc00u, hreg, hs + 5, 4);   // t=5 b=2
          _E2(h2[base + 12], r8,  0x00300030u, 0x2c002c00u, 0xd400d400u, hreg, hs + 6, 6);   // t=6 b=4
          _E2(h2[base + 13], r8,  0x00c000c0u, 0x24002400u, 0xcc00cc00u, hreg, hs + 7, 8);   // t=7 b=6
        }
        #undef _E2
    }
};

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
