/***************************************************************************************************
 * The PACKED GGUF scale unit, and its decode. Plan #20 option E.
 *
 * WHY THIS LIVES IN actlize AND NOT NEXT TO THE HARNESSES: the mainloop needs it, and the mainloop is here. It used to
 * live only in general/w4a16_gemm/cutlass_w4a16/gguf_scale_decode.hpp, which the collective cannot see; copying it would
 * have made one relation exist twice, the failure mode this work keeps hitting. So the FORMAT-INDEPENDENT part moved
 * here and that harness header now includes this one.
 *
 * The unit is 16 bytes per (superblock, column), written by the offline in this order:
 *
 *     byte 0-1  d    byte 2-3  dmin    byte 4-9  half0 (groups 0-3)    byte 10-15  half1 (groups 4-7)
 *
 * with each half self-contained -- 4 scale + 4 min codes as 6-bit fields, 48 bits exactly. The gguf's OWN packing is not
 * separable (get_scale_min_k4 takes sc[4..7] from bytes 8-11 and bytes 0-3's top two bits), so a k-tile covering half a
 * superblock could not read half a block; this order fixes that at no cost in bytes.
 *
 * One Layout gives every bit position, so no shift arithmetic is written down:
 *     bit(g, which) = 32 + 6*(g%4) + 48*(g/4) + 24*which        which = 0 scale, 1 min
 *
 * Gated in fold_derivation/l94 (4) as a round trip: native 12 B -> reference sc/mn (l91, 4096 real superblocks) -> this
 * packing -> this decode -> the reference, 0 bad over 128 columns x 8 groups, and 0 bits outside their own half.
 **************************************************************************************************/
#pragma once

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cute/layout.hpp"

namespace cutlass {
namespace gguf_packed {

using PackBits = cute::Layout<cute::Shape <cute::_4, cute::_2, cute::_2>,
                              cute::Stride<cute::_6, cute::Int<48>, cute::Int<24>>>;
static constexpr int kBitBase  = 32;      // d and dmin occupy the first four bytes
static constexpr int kUnitBytes = 16;

CUTLASS_HOST_DEVICE constexpr int bit_of(int g, int which) {
  return kBitBase + int(PackBits{}(g % 4, g / 4, which));
}

// A 6-bit field may straddle a byte boundary (half0 starts at byte 4, so the field starts at bit 32 + 6i), hence a
// 16-bit read and one shift. Stated once, here.
CUTLASS_HOST_DEVICE int code_of(uint8_t const* unit, int g, int which) {
  int const bit = bit_of(g, which);
  uint32_t const w = uint32_t(unit[bit >> 3]) | (uint32_t(unit[(bit >> 3) + 1]) << 8);
  return int((w >> (bit & 7)) & 0x3Fu);
}
CUTLASS_HOST_DEVICE void put_code(uint8_t* unit, int g, int which, int v) {     // the offline side, same one rule
  int const bit = bit_of(g, which);
  unit[bit >> 3]       |= uint8_t((uint32_t(v) << (bit & 7)) & 0xFFu);
  unit[(bit >> 3) + 1] |= uint8_t((uint32_t(v) << (bit & 7)) >> 8);
}

// REGISTER-RESIDENT EXTRACTION. The field's bit position is a compile-time constant, so this is a shift and a mask on
// values that stay in registers.
//
// This exists because code_of() takes a `uint8_t const*`, which is right for smem and gmem and WRONG for registers:
// byte-addressing a register array forces it to local memory. Measured on ppu001, that one mistake made the packed path
// 70% slower than the fp16 one it replaces (16x64:256 S=4: 34.02 -> 57.83 us) -- the 16 shared loads it saves per k-tile
// were traded for 64 local ones.
// Bits DEFAULTS TO 6 so every existing call site is unchanged; Q2_K's fields are 4 and Q6_K's are 8, and a mask
// written as a literal 0x3F would silently truncate the latter and pick up a neighbour's bits for the former.
template <int Bit, int NWords, int Bits = 6>
CUTLASS_HOST_DEVICE int code_from_words(uint32_t const (&u)[NWords]) {
  constexpr int w = Bit >> 5, off = Bit & 31;
  static_assert(w < NWords, "the field lies outside the unit");
  static_assert(Bits > 0 && Bits <= 16, "field widths beyond 16 bits need a second word unconditionally");
  uint32_t v = u[w] >> off;
  // A field straddles a word exactly when it starts within Bits-1 of the top, so the condition follows the width
  // rather than assuming six.
  if constexpr (off > 32 - Bits && w + 1 < NWords) v |= u[w + 1] << (32 - off);
  return int(v & ((1u << Bits) - 1u));
}

// ---------------------------------------------------------------------------------------------------------------
// THE UNIT, GENERALISED. Everything above describes exactly one format; these traits derive the same numbers for all
// five, and the static_assert at the bottom checks that the general rule REPRODUCES the hand-written Q4_K bit
// positions -- which is the only evidence worth having that this is a generalisation and not a second scheme.
//
// WHY IT LIVES HERE. The mainloop needs it and the mainloop is here; the harness-side header re-exports rather than
// copying, for the reason the top of this file gives -- one relation, one definition.
enum class Fmt { Q4K, Q5K, Q2K, Q3K, Q6K };

template <Fmt F> struct UnitTraits;
//                          groups scale_bits min_bits has_min signed scale_bias
template <> struct UnitTraits<Fmt::Q4K> { static constexpr int kGroups=8,  kScaleBits=6, kMinBits=6, kScaleBias=0;
                                          static constexpr bool kHasMin=true,  kSigned=false; };
template <> struct UnitTraits<Fmt::Q5K> : UnitTraits<Fmt::Q4K> {};        // same scale block as Q4_K
template <> struct UnitTraits<Fmt::Q2K> { static constexpr int kGroups=16, kScaleBits=4, kMinBits=4, kScaleBias=0;
                                          static constexpr bool kHasMin=true,  kSigned=false; };
template <> struct UnitTraits<Fmt::Q3K> { static constexpr int kGroups=16, kScaleBits=6, kMinBits=0, kScaleBias=32;
                                          static constexpr bool kHasMin=false, kSigned=false; };
template <> struct UnitTraits<Fmt::Q6K> { static constexpr int kGroups=16, kScaleBits=8, kMinBits=0, kScaleBias=0;
                                          static constexpr bool kHasMin=false, kSigned=true;  };

// The derived shape. kUnitBytes is one superblock's metadata share and DIFFERS PER FORMAT -- 16, 16, 20, 14, 18.
// A copyable unit pairs two consecutive superblocks of the SAME column when that share is 2 mod 4; this keeps the
// stored bytes neutral and preserves thread/column ownership while satisfying ppu.cp.async's 4-byte minimum.
template <Fmt F> struct Unit {
  using T = UnitTraits<F>;
  static constexpr int  kGroups      = T::kGroups;
  static constexpr int  kScaleBits   = T::kScaleBits;
  static constexpr int  kMinBits     = T::kMinBits;
  static constexpr bool kHasMin      = T::kHasMin;
  static constexpr bool kSigned      = T::kSigned;
  static constexpr int  kScaleBias   = T::kScaleBias;
  static constexpr int  kHeaderBytes = kHasMin ? 4 : 2;                 // d, plus dmin only if there is a min
  static constexpr int  kCodeBits    = kGroups * (kScaleBits + kMinBits);
  static constexpr int  kUnitBytes   = kHeaderBytes + kCodeBits / 8;
  static constexpr int  kSbBytes     = kUnitBytes;
  static constexpr int  kSbPerUnit   = (kSbBytes % 4 == 0) ? 1 : 2;
  static constexpr int  kUnitTotal   = kSbPerUnit * kSbBytes;
  // Fields lie group-major inside a RUN, and each run is self-contained so a k-tile covering part of a superblock
  // reads a contiguous byte range -- the entire reason the unit is reordered at all.
  static constexpr int  kRunGroups   = kHasMin ? (kGroups / 2) : kGroups;
  static constexpr int  kRunBits     = kRunGroups * (kScaleBits + kMinBits);
  static_assert(kCodeBits % 8 == 0, "the code field must fill whole bytes");
  static_assert(kRunBits % 8 == 0, "a run must be whole bytes or it is not self-contained in memory");

  // CUTE OWNS THE FIELD ADDRESS. Scale and min have different strides within a run, so keep one truthful layout per
  // field and pass the selected layout to field_bit_of. This is the device-side twin of the offline packed-unit
  // writer's layouts; activating a new format must not introduce a second hand-expanded run/field bit formula.
  using RunShape = cute::Shape<cute::Int<kRunGroups>, cute::Int<kGroups / kRunGroups>>;
  using ScaleBitLayout = cute::Layout<RunShape,
                                      cute::Stride<cute::Int<kScaleBits>, cute::Int<kRunBits>>>;
  using MinBitLayout = cute::Layout<RunShape,
                                    cute::Stride<cute::Int<kMinBits>, cute::Int<kRunBits>>>;

  template <class BitLayout>
  CUTLASS_HOST_DEVICE static constexpr int field_bit_of(int g, BitLayout const& layout) {
    return int(layout(g % kRunGroups, g / kRunGroups));
  }

  CUTLASS_HOST_DEVICE static constexpr int bit_of(int g, int which) {
    return kHeaderBytes * 8 + (which
        ? kRunGroups * kScaleBits + field_bit_of(g, MinBitLayout{})
        : field_bit_of(g, ScaleBitLayout{}));
  }
};

// THE GENERAL RULE MUST REPRODUCE THE SHIPPED ONE. If this ever fires, the generalisation has invented a different
// layout and every Q4_K artifact on disk is misread -- which is why it is a compile-time check and not a test.
static_assert(Unit<Fmt::Q4K>::kUnitBytes == kUnitBytes, "generalised Q4_K unit size differs from the shipped one");
static_assert(Unit<Fmt::Q5K>::kUnitTotal == 16 && Unit<Fmt::Q2K>::kUnitTotal == 20 &&
              Unit<Fmt::Q3K>::kUnitTotal == 28 && Unit<Fmt::Q6K>::kUnitTotal == 36,
              "copyable packed units must remain byte-neutral 16/20/28/36-byte metadata records");
static_assert(Unit<Fmt::Q4K>::bit_of(0,0) == bit_of(0,0) && Unit<Fmt::Q4K>::bit_of(3,0) == bit_of(3,0) &&
              Unit<Fmt::Q4K>::bit_of(4,0) == bit_of(4,0) && Unit<Fmt::Q4K>::bit_of(7,0) == bit_of(7,0) &&
              Unit<Fmt::Q4K>::bit_of(0,1) == bit_of(0,1) && Unit<Fmt::Q4K>::bit_of(3,1) == bit_of(3,1) &&
              Unit<Fmt::Q4K>::bit_of(4,1) == bit_of(4,1) && Unit<Fmt::Q4K>::bit_of(7,1) == bit_of(7,1),
              "generalised Q4_K bit positions differ from the shipped ones");

// A field is at most 8 bits, so two bytes always suffice -- and the second is read ONLY when the field straddles,
// because reading it unconditionally runs off the end of the unit for the last field of the last run (Q4_K's ends at
// bit 122 of 16 bytes; Q6_K's byte-aligned scales put the last at byte 17 of 18).
template <Fmt F>
CUTLASS_HOST_DEVICE int code_of_fmt(uint8_t const* unit, int g, int which) {
  int const bit = Unit<F>::bit_of(g, which);
  int const bits = which ? Unit<F>::kMinBits : Unit<F>::kScaleBits;
  int const off = bit & 7;
  uint32_t w = uint32_t(unit[bit >> 3]);
  if (off + bits > 8) w |= uint32_t(unit[(bit >> 3) + 1]) << 8;
  return int((w >> off) & ((1u << bits) - 1u));
}

struct GroupScale {
  half_t scale;
  half_t zero;
};

// int -> fp16 by the B converter's identity at bpos 0: v+128 in the mantissa of 0x6400, minus 1152. The +128 is what
// lets the SAME pair of instructions carry signed codes (Q6_K's int8 scales) with no second path. Exact for
// v in [-128, 895] and checked over that whole range in fold_derivation/l93.
CUTLASS_HOST_DEVICE half_t int_to_half_small(int v) {
  return half_t::bitcast(uint16_t(0x6400u | uint32_t((v + 128) & 0x3FF))) - half_t(1152.f);
}

// Q4_K: codes are unsigned with no centre of their own, and the affine term is -dmin*mn. `ScaleBias` and `HasMin` are
// template parameters rather than hardcoded so Q3_K (bias 32, no min) and Q6_K (signed, no min) need no second function.
// ZMul CANCELS THE CONVERTER'S OWN BIAS, and it is not optional. The mainloop computes scale*emitted + zero, and the
// int4 converter emits nib - 8, so a zero of -dmin*mn would leave the product short by 8*scale. With ZMul = 8 the zero is
// 8*scale - dmin*mn and the product is scale*nib - dmin*mn, which is Q4_K's dequant.
//
// The alternative -- setting the converter's Bias to 0 so the codes pass through as nib and the zero stays -dmin*mn --
// was measured against fp64 truth in real_weight/dump_packed_scale.py gate (c): 0.0148 step of error versus 0.0128 for
// this cancelling form, i.e. the two are equivalent and this one is if anything slightly better. It also leaves the
// shipped int4 converter untouched, which the other route would not.
template <int ScaleBias = 0, bool HasMin = true, int ZMul = 0>
CUTLASS_HOST_DEVICE GroupScale group_of(uint8_t const* unit, int g) {
  half_t const d    = half_t::bitcast(uint16_t(unit[0]) | uint16_t(uint16_t(unit[1]) << 8));
  GroupScale out;
  out.scale = d * int_to_half_small(code_of(unit, g, 0) - ScaleBias);
  if constexpr (HasMin) {
    half_t const dmin = half_t::bitcast(uint16_t(unit[2]) | uint16_t(uint16_t(unit[3]) << 8));
    out.zero = -(dmin * int_to_half_small(code_of(unit, g, 1)));
  } else {
    out.zero = half_t(0.f);
  }
  if constexpr (ZMul != 0) out.zero = out.zero + half_t(float(ZMul)) * out.scale;
  return out;
}

// d AND dmin, EXTRACTED ONCE PER K-TILE. They are the unit's first four bytes and constant for every group in the
// superblock, so re-deriving them inside a per-group decode is pure waste: acu put it at roughly 0.5M extra instructions
// on the decode band (8 groups x 2 slots repeating the same two bitcasts).
struct UnitHead {
  half_t d;
  half_t dmin;
};
template <int NWords>
CUTLASS_HOST_DEVICE UnitHead head_of_words(uint32_t const (&u)[NWords]) {
  return UnitHead{half_t::bitcast(uint16_t(u[0] & 0xFFFFu)), half_t::bitcast(uint16_t(u[0] >> 16))};
}

// ===================================================================================================================
// PACKED FORM -- both of a group's fields in the two halves of one 32-bit register. Same arithmetic, four fewer
// instructions per group (15 -> 11 measured as opcodes), and BIT-IDENTICAL to group_of_words rather than merely close.
// Each of the four savings is an identity, not an approximation:
//
//  (1) `+128`, `& 0x3FF` and `| 0x6400` collapse into ONE integer add:
//        0x6400 | ((v + 128) & 0x3FF)  ==  0x6480 + v      for every v in [-128, 895]
//      0x6400's low ten bits are zero so the OR is an add; (v+128) lands in [0, 1023] so the mask is a no-op; and
//      0x6480 + 895 = 0x67FF with 0x6480 - 128 = 0x6400, so the value never leaves its own 16-bit lane and one
//      `+ 0x64806480` serves BOTH fields at once.
//      THE PRECONDITION IS THE RANGE, NOT THE SIGN. I first wrote this as "unsigned codes only" and l96's boundary
//      probe refuted it: the bound is exactly int_to_half_small's own [-128, 895], so signed codes are fine and a
//      format's kScaleBias folds into the constant (Q3_K would use 0x6480 - 32 in the scale lane). What actually
//      restricts group_pair_of_words to the Q4_K shape is narrower and lives elsewhere: code_pair_from_words masks
//      6-bit fields, and only a format WITH a min has a second field to pack against in the first place.
//
//  (2) half(1152) IS 0x6480, so the constant that biases up and the constant that subtracts down are the same word.
//      The subtraction is exact in both lanes -- 1024+m and 1152 are integers in fp16 and their difference is an
//      integer of magnitude <= 895 -- which is the same reason int_to_half_small is exact.
//
//  (3) the multiplier half2(d, -dmin) is `u[0] ^ 0x80000000`. u[0] already holds half2(d, dmin) in the right lanes
//      (the unit's first four bytes, little endian), and flipping the top bit negates the high lane. So hoisting the
//      min's negation onto the multiplier -- exact, since negation is exact -- costs one xor per COLUMN, and the
//      per-group negate disappears.
//
//  (4) the two products become one ppu.fma.rtte.f16x2 with addend half2(-0, -0). THE ISA HAS NO ppu.mul.f16x2 (all
//      four f16x2 mnemonics in this tree are cvt, fma, sub and tanh), and fma(x, m, -0) is not an approximation of
//      the multiply -- it IS the multiply, for every input including signed zeros: p + (-0) = p for p != 0,
//      (+0) + (-0) = +0, and (-0) + (-0) = -0.
//
// The asm is gated on the PPU compiler AND on not-nvcc: fold_derivation's local gates compile these headers with
// nvcc (l95 even with -D__HGGCCC__), where a ppu mnemonic reaching ptxas is a hard error. The scalar fallback is
// what the host-side bit-identity gate in l96 actually checks, and it performs the same two half_t operations in the
// same order, so agreeing with it is agreeing with group_of_words.
#if defined(__HGGCCC__) && !defined(__NVCC__)
#  define CUTLASS_GGUF_PACKED_F16X2_ASM 1
#else
#  define CUTLASS_GGUF_PACKED_F16X2_ASM 0
#endif

// SUBNORMAL FLUSHING IS THE HAZARD, AND Q4_K WALKS STRAIGHT INTO IT. Measured on the real fixture
// (real_weight/q4k_packed.bin, blk.11.ffn_down.weight): d spans 1.585e-05 to 9.484e-04 while fp16's smallest normal
// is 6.104e-05, so 3914 of 4864 superblock d values -- 80% -- are SUBNORMAL. dmin is not (1.1e-04 to 1.1e-02, zero
// subnormal), and the PRODUCTS d*sc are not (0 of 38912). That last fact is why the fp16-plane baseline never meets
// this: the offline forms d*sc in fp32 and stores only the normal product, so no subnormal fp16 ever reaches an
// instruction. The packed decode is the first thing to multiply BY d on the device.
//
// This ISA has an explicit `.noftz` qualifier on at least one f16x2 op (cutlass/functional.h:830,
// ppu.atom.gpu.global.add.noftz.f16x2), which only makes sense if the DEFAULT flushes. If ppu.fma.rtte.f16x2 flushes
// its subnormal input, the scale lane becomes 0 for 80% of superblocks while the zero lane (normal dmin) survives --
// which is exactly the observed failure: rowC wrong, errors dominated by scale, partial rather than total because each
// output sums over 19 superblocks and only the subnormal ones are lost.
//
// PPU_F16X2_NOFTZ=1 emits the qualified form. It is a PROBE first and a fix second: one build says whether the
// assembler accepts the mnemonic at all, and test_ppu_f16x2_probe says whether it changes the answer.
#if defined(PPU_F16X2_NOFTZ) && (PPU_F16X2_NOFTZ != 0)
#  define CUTLASS_PPU_F16X2_SUB "ppu.sub.noftz.f16x2 %0, %1, %2;\n"
#  define CUTLASS_PPU_F16X2_FMA "ppu.fma.rtte.noftz.f16x2 %0, %1, %2, %3;\n"
#else
#  define CUTLASS_PPU_F16X2_SUB "ppu.sub.f16x2 %0, %1, %2;\n"
#  define CUTLASS_PPU_F16X2_FMA "ppu.fma.rtte.f16x2 %0, %1, %2, %3;\n"
#endif

CUTLASS_HOST_DEVICE uint32_t pack_h2(half_t lo, half_t hi) {
  return uint32_t(lo.raw()) | (uint32_t(hi.raw()) << 16);
}
CUTLASS_HOST_DEVICE half_t lo_h2(uint32_t a) { return half_t::bitcast(uint16_t(a & 0xFFFFu)); }
CUTLASS_HOST_DEVICE half_t hi_h2(uint32_t a) { return half_t::bitcast(uint16_t(a >> 16)); }

// PPU_F16X2_EARLYCLOBBER=0 PUTS "=r" BACK, and it exists to turn a coincidence into a cause. rowC went from
// bad=724/4096 to MATCH across four commits, of which "=&r" is only the most plausible; one build with this at 0 says
// whether that line was the fix or whether the failure merely went away. Note what the constraint does NOT explain:
// test_ppu_f16x2_probe section (4) forces dest == each input in turn and the hardware returns the SAME answer in 5 of
// 5 forms, so the instruction tolerates overlap. Whatever "=r" allowed happens in register allocation under the
// mainloop's pressure -- where m2 is live across eight unrolled groups -- and is invisible with three live values.
// Pinning it further needs the emitted assembly, which this session did not have.
#if defined(PPU_F16X2_EARLYCLOBBER) && (PPU_F16X2_EARLYCLOBBER == 0)
#  define CUTLASS_PPU_F16X2_OUT "=r"
#else
#  define CUTLASS_PPU_F16X2_OUT "=&r"
#endif

CUTLASS_HOST_DEVICE uint32_t sub_f16x2(uint32_t a, uint32_t b) {
#if CUTLASS_GGUF_PACKED_F16X2_ASM
  uint32_t d;
  // "=&r" (EARLYCLOBBER), not "=r". With "=r" the compiler may allocate the destination to the same register as an
  // input, which is legal and is what the reference uses in fast_numeric_conversion_for_mix_gemm.h -- but there the
  // output IS an input by construction. Here the operands are distinct, so aliasing would depend on register
  // allocation, which differs per unrolled group, which is exactly the shape of a PARTIAL failure.
  asm volatile(CUTLASS_PPU_F16X2_SUB : CUTLASS_PPU_F16X2_OUT(d) : "r"(a), "r"(b));
  return d;
#else
  return pack_h2(lo_h2(a) - lo_h2(b), hi_h2(a) - hi_h2(b));
#endif
}

CUTLASS_HOST_DEVICE uint32_t fma_f16x2(uint32_t a, uint32_t b, uint32_t c) {
#if CUTLASS_GGUF_PACKED_F16X2_ASM
  uint32_t d;
  asm volatile(CUTLASS_PPU_F16X2_FMA : CUTLASS_PPU_F16X2_OUT(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  return pack_h2(lo_h2(a) * lo_h2(b) + lo_h2(c), hi_h2(a) * hi_h2(b) + hi_h2(c));
#endif
}

// half2(1152, 1152) -- and, by (1) above, also the bias/mask/magic-OR constant. One word, two jobs.
static constexpr uint32_t kMagic1152x2 = 0x64806480u;
// half2(-0, -0) -- the fma addend that turns the fma into an exact multiply.
static constexpr uint32_t kNegZeroX2   = 0x80008000u;

// The two 6-bit fields of one group, landed at bits 0 and 16 of one word. The shift direction for the high field is
// a compile-time choice, so this is shift, mask, shift, mask, or.
template <int BitLo, int BitHi, int NWords>
CUTLASS_HOST_DEVICE uint32_t code_pair_from_words(uint32_t const (&u)[NWords]) {
  constexpr int wl = BitLo >> 5, ol = BitLo & 31;
  constexpr int wh = BitHi >> 5, oh = BitHi & 31;
  static_assert(wl < NWords && wh < NWords, "a field lies outside the unit");
  uint32_t lo = u[wl] >> ol;
  if constexpr (ol > 26 && wl + 1 < NWords) lo |= u[wl + 1] << (32 - ol);
  uint32_t hi = u[wh] >> oh;
  if constexpr (oh > 26 && wh + 1 < NWords) hi |= u[wh + 1] << (32 - oh);
  // Written as a 6-bit insert at position 16 because that is one v.bfi.i, the opcode this converter already issues
  // 270k times. Whether the compiler takes it is a code-generation question, not a correctness one.
  return (lo & 0x3Fu) | ((hi & 0x3Fu) << 16);
}

// half2(d, -dmin) for the whole column: one xor. See (3).
template <int NWords>
CUTLASS_HOST_DEVICE uint32_t mul2_of_words(uint32_t const (&u)[NWords]) { return u[0] ^ 0x80000000u; }

// One group, both fields, from registers. `m2` is mul2_of_words' result, hoisted out of the group loop.
// Restricted to the Q4_K shape (unsigned codes, no centre, has a min) because that is where identity (1) holds;
// every other format keeps group_of_words, which is unchanged.
// ScaleBias IS A PARAMETER HERE ONLY SO IT CANNOT BE FORGOTTEN. This function has no per-lane place to subtract a
// centre from just the scale field -- the whole point is that ONE add serves both fields -- so a non-zero centre must
// come in through the constant, which is a change to the identity in (1) and not a change to this call. Rather than
// leave that as a comment, the assert makes a format that needs a centre fail to compile instead of silently losing
// it. The caller's kPackedPairFast gate happens to exclude that case today; nothing tied the two together, which is
// exactly the shape of defect this file keeps recording.
template <int G, int ZMul = 0, int ScaleBias = 0, int NWords>
CUTLASS_HOST_DEVICE GroupScale group_pair_of_words(uint32_t const (&u)[NWords], uint32_t const m2) {
  static_assert(ScaleBias == 0,
                "group_pair_of_words folds the bias into kMagic1152x2; a non-zero centre needs its own constant "
                "(0x6480 - ScaleBias in the scale lane), not this path");
  uint32_t const c2 = code_pair_from_words<bit_of(G, 0), bit_of(G, 1)>(u);
  uint32_t const x2 = sub_f16x2(c2 + kMagic1152x2, kMagic1152x2);   // half2(sc, mn), exactly
  uint32_t const y2 = fma_f16x2(x2, m2, kNegZeroX2);                // half2(d*sc, -dmin*mn)
  GroupScale out;
  out.scale = lo_h2(y2);
  out.zero  = hi_h2(y2);
  if constexpr (ZMul != 0) out.zero = out.zero + half_t(float(ZMul)) * out.scale;
  return out;
}

// The per-group part only: the two codes and the two products, with d/dmin handed in.
//
// F DEFAULTS TO Q4K so every existing call is unchanged, and every bit position and field width now comes from
// Unit<F> rather than from the hand-written Q4_K bit_of. Passing G >= 8 to the old form computed a Q4_K position for
// a group that format does not have, which is what the 16-group formats hit as "the field lies outside the unit".
template <int G, int ScaleBias = 0, bool HasMin = true, int ZMul = 0, Fmt F = Fmt::Q4K, int NWords>
CUTLASS_HOST_DEVICE GroupScale group_of_words(uint32_t const (&u)[NWords], UnitHead const h) {
  GroupScale out;
  int sc = code_from_words<Unit<F>::bit_of(G, 0), NWords, Unit<F>::kScaleBits>(u);
  if constexpr (Unit<F>::kSigned) {
    constexpr int kSign = 1 << (Unit<F>::kScaleBits - 1);
    if (sc & kSign) sc -= 1 << Unit<F>::kScaleBits;
  }
  out.scale = h.d * int_to_half_small(sc - ScaleBias);
  if constexpr (HasMin)
    out.zero = -(h.dmin * int_to_half_small(code_from_words<Unit<F>::bit_of(G, 1), NWords, Unit<F>::kMinBits>(u)));
  else                  out.zero = half_t(0.f);
  if constexpr (ZMul != 0) out.zero = out.zero + half_t(float(ZMul)) * out.scale;
  return out;
}

// The same dequant as group_of, from REGISTERS. G is a template parameter because that is what makes every bit position
// a constant; d and dmin are the unit's first four bytes, i.e. the low and high halves of word 0 (little endian).
template <int G, int ScaleBias = 0, bool HasMin = true, int ZMul = 0, Fmt F = Fmt::Q4K, int NWords>
CUTLASS_HOST_DEVICE GroupScale group_of_words(uint32_t const (&u)[NWords]) {
  GroupScale out;
  half_t const d = half_t::bitcast(uint16_t(u[0] & 0xFFFFu));
  int sc = code_from_words<Unit<F>::bit_of(G, 0), NWords, Unit<F>::kScaleBits>(u);
  if constexpr (Unit<F>::kSigned) {
    constexpr int kSign = 1 << (Unit<F>::kScaleBits - 1);
    if (sc & kSign) sc -= 1 << Unit<F>::kScaleBits;
  }
  out.scale = d * int_to_half_small(sc - ScaleBias);
  if constexpr (HasMin) {
    half_t const dmin = half_t::bitcast(uint16_t(u[0] >> 16));
    out.zero = -(dmin * int_to_half_small(code_from_words<Unit<F>::bit_of(G, 1), NWords, Unit<F>::kMinBits>(u)));
  } else {
    out.zero = half_t(0.f);
  }
  if constexpr (ZMul != 0) out.zero = out.zero + half_t(float(ZMul)) * out.scale;
  return out;
}

}  // namespace gguf_packed
}  // namespace cutlass
