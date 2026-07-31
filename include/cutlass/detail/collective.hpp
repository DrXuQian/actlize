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

#include "cute/container/tuple.hpp"
#include "cute/layout.hpp" // cute::size(shape)
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <size_t I, class Tuple>
struct deduce_mixed_width_dtype {
// Index 3 added for the PPU B BIT-PLANE CONCAT (Q3=int2+int1, Q5=int4+int1, Q6=int4+int2): the 4th member of the
// B element tuple carries the SECOND B plane's element type, which is how the 2-plane mainloop
// (ppu_mma_aiu_mixed_input_2plane.hpp) gets it -- CollectiveMma's template parameter list is fixed by its primary
// template, so an extra parameter is impossible. The body below already returns void for any out-of-range index,
// so this only relaxes the guard; indices 0/1/2 behave exactly as before.
static_assert(I >= 0u && I <= 3u,
  "Valid indices are 0, 1, 2 (Operand, Scale, Bias) and 3 (second B bit plane, PPU bit-plane concat).");

private:
  using underlying_tuple = cute::conditional_t<cute::is_tuple<Tuple>::value, Tuple, cute::tuple<Tuple>>;
  static constexpr size_t valid_index = cute::min(I, cute::tuple_size_v<underlying_tuple> - 1);

public:
  using type = cute::conditional_t<(I < cute::tuple_size_v<underlying_tuple>), 
                                    cute::tuple_element_t<valid_index, underlying_tuple>,
                                    void>;
};

template <size_t I, class Tuple>
using deduce_mixed_width_dtype_t = typename deduce_mixed_width_dtype<I, Tuple>::type;

// AN EXPLICITLY EMPTY ZERO SLOT, for the 2-plane B tuple only.
//
// Everywhere else "no zero" is expressed by tuple LENGTH: <B, Scale> has none, <B, Scale, Zero> has one. The 2-plane
// path cannot use that convention, because its second B plane is deduced POSITIONALLY at index 3 -- so dropping the
// zero would leave <B, Scale, Plane>, where index 2 holds a plane and index 3 is void. Telling those two encodings
// apart from the types alone is impossible in general (a zero is fp16, a plane is a narrow int, and keying on that is
// exactly the kind of implicit rule that breaks silently later). cute::tuple cannot hold `void`, so the empty slot
// needs a real type: <B, Scale, NoZero, Plane> keeps the plane at 3 and says "no zero" out loud.
//
// This exists because QuantMode::FinegrainedScaleOnly ran as ScaleZero on the 2-plane path for as long as that path
// existed: the driver built the 4-tuple unconditionally, so every "ScaleOnly" 2-plane row -- including the ones in
// test_q3_bconcat_bench that were labelled as such -- was really an affine run whose zero cancelled the converter's
// bias. Measured, not guessed: with the bias moved to the converter the outputs came out equal to the UNBIASED golden,
// and the ratio (32-B)/(16-B) across the Q6 and Q5 rungs pinned B = 0.
struct NoZero {};

template <class T>
using strip_no_zero_t = cute::conditional_t<cute::is_same_v<T, NoZero>, void, T>;

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective
