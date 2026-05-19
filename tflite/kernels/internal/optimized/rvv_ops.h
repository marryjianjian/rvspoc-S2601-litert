/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_OPS_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"

namespace tflite {
namespace rvv_ops {

#if defined(USE_RVV) && !TFLITE_SINGLE_ROUNDING

inline vint32m4_t RoundingDivideByPOT(vint32m4_t x, int exponent, size_t vl) {
  const int32_t mask = (1ll << exponent) - 1;
  const int32_t threshold_base = mask >> 1;

  const vint32m4_t remainder = __riscv_vand_vx_i32m4(x, mask, vl);
  const vbool8_t negative_mask = __riscv_vmslt_vx_i32m4_b8(x, 0, vl);
  const vint32m4_t threshold_non_negative =
      __riscv_vmv_v_x_i32m4(threshold_base, vl);
  const vint32m4_t threshold_negative =
      __riscv_vmv_v_x_i32m4(threshold_base + 1, vl);
  const vint32m4_t threshold = __riscv_vmerge_vvm_i32m4(
      threshold_non_negative, threshold_negative, negative_mask, vl);

  const vint32m4_t shifted = __riscv_vsra_vx_i32m4(x, exponent, vl);
  const vint32m4_t rounded_up = __riscv_vadd_vx_i32m4(shifted, 1, vl);
  const vbool8_t round_up_mask =
      __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
  return __riscv_vmerge_vvm_i32m4(shifted, rounded_up, round_up_mask, vl);
}

inline vint32m4_t
SaturatingRoundingDoublingHighMul(vint32m4_t x, int32_t multiplier, size_t vl) {
  vint64m8_t product = __riscv_vwmul_vx_i64m8(x, multiplier, vl);
  const vbool8_t negative_product_mask =
      __riscv_vmslt_vx_i64m8_b8(product, 0, vl);
  const vint64m8_t positive_nudge = __riscv_vmv_v_x_i64m8(1ll << 30, vl);
  const vint64m8_t negative_nudge = __riscv_vmv_v_x_i64m8(1 - (1ll << 30), vl);
  const vint64m8_t nudge = __riscv_vmerge_vvm_i64m8(
      positive_nudge, negative_nudge, negative_product_mask, vl);
  product = __riscv_vadd_vv_i64m8(product, nudge, vl);

  vint64m8_t quotient = __riscv_vsra_vx_i64m8(product, 31, vl);
  const vint64m8_t remainder =
      __riscv_vand_vx_i64m8(product, (1ll << 31) - 1, vl);
  const vbool8_t negative_quotient_mask =
      __riscv_vmslt_vx_i64m8_b8(product, 0, vl);
  const vbool8_t non_zero_remainder_mask =
      __riscv_vmsne_vx_i64m8_b8(remainder, 0, vl);
  const vbool8_t correction_mask =
      __riscv_vmand_mm_b8(negative_quotient_mask, non_zero_remainder_mask, vl);
  const vint64m8_t corrected_quotient = __riscv_vadd_vx_i64m8(quotient, 1, vl);
  quotient = __riscv_vmerge_vvm_i64m8(quotient, corrected_quotient,
                                      correction_mask, vl);

  vint32m4_t result = __riscv_vnsra_wx_i32m4(quotient, 0, vl);
  if (multiplier == std::numeric_limits<int32_t>::min()) {
    const vbool8_t overflow_mask =
        __riscv_vmseq_vx_i32m4_b8(x, std::numeric_limits<int32_t>::min(), vl);
    const vint32m4_t saturated =
        __riscv_vmv_v_x_i32m4(std::numeric_limits<int32_t>::max(), vl);
    result = __riscv_vmerge_vvm_i32m4(result, saturated, overflow_mask, vl);
  }
  return result;
}

inline vint32m4_t
MultiplyByQuantizedMultiplierSmallerThanOneExp(vint32m4_t x, int32_t multiplier,
                                               int shift, size_t vl) {
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, multiplier, vl), -shift, vl);
}

inline vint32m4_t MultiplyByQuantizedMultiplier(vint32m4_t x,
                                                int32_t multiplier, int shift,
                                                size_t vl) {
  const int left_shift = std::max(shift, 0);
  const int right_shift = std::max(-shift, 0);
  x = __riscv_vsll_vx_i32m4(x, left_shift, vl);
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, multiplier,
                                                               vl),
                             right_shift, vl);
}

#endif  // defined(USE_RVV) && !TFLITE_SINGLE_ROUNDING

}  // namespace rvv_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_OPS_H_
