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
template <int ScaleBias = 0, bool HasMin = true>
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
  return out;
}

}  // namespace gguf_packed
}  // namespace cutlass
