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

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective
