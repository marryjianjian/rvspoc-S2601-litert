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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

#ifdef USE_RVV

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
inline bool RvvGemmLayoutIsSupported(
    const MatrixParams<LhsScalar>& lhs_params,
    const MatrixParams<RhsScalar>& rhs_params,
    const MatrixParams<DstScalar>& dst_params) {
  return lhs_params.order == Order::kRowMajor &&
         rhs_params.order == Order::kColMajor &&
         dst_params.order == Order::kColMajor &&
         lhs_params.cols == rhs_params.rows &&
         lhs_params.rows == dst_params.rows &&
         rhs_params.cols == dst_params.cols && lhs_params.cols > 0;
}

inline float RvvReduceSum(vfloat32m4_t values, size_t vl) {
  vfloat32m1_t reduced = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  reduced = __riscv_vfredusum_vs_f32m4_f32m1(values, reduced, vl);
  return __riscv_vfmv_f_s_f32m1_f32(reduced);
}

inline int32_t RvvReduceSum(vint32m4_t values, size_t vl) {
  vint32m1_t reduced = __riscv_vmv_v_x_i32m1(0, 1);
  reduced = __riscv_vredsum_vs_i32m4_i32m1(values, reduced, vl);
  return __riscv_vmv_x_s_i32m1_i32(reduced);
}

template <typename Scalar>
inline vint16m2_t RvvLoad8AndSubtractZeroPoint(const Scalar* src,
                                               Scalar zero_point, size_t vl) {
  if constexpr (std::is_same<Scalar, std::int8_t>::value) {
    const vint8m1_t values = __riscv_vle8_v_i8m1(src, vl);
    vint16m2_t widened = __riscv_vsext_vf2_i16m2(values, vl);
    widened =
        __riscv_vsub_vx_i16m2(widened, static_cast<int16_t>(zero_point), vl);
    return widened;
  } else {
    static_assert(std::is_same<Scalar, std::uint8_t>::value, "");
    const vuint8m1_t values = __riscv_vle8_v_u8m1(src, vl);
    const vuint16m2_t widened_u16 = __riscv_vzext_vf2_u16m2(values, vl);
    vint16m2_t widened = __riscv_vreinterpret_v_u16m2_i16m2(widened_u16);
    widened =
        __riscv_vsub_vx_i16m2(widened, static_cast<int16_t>(zero_point), vl);
    return widened;
  }
}

template <typename DstScalar, QuantizationFlavor quantization_flavor>
inline DstScalar RvvFinalizeQuantizedAccumulator(
    int32_t acc, int row, const MatrixParams<DstScalar>& dst_params,
    const GemmParams<int32_t, DstScalar, quantization_flavor>& params) {
  if (params.bias != nullptr) {
    acc += params.bias[row];
  }
  int32_t multiplier;
  int shift;
  if constexpr (quantization_flavor ==
                QuantizationFlavor::kIntegerWithPerRowMultiplier) {
    multiplier = params.multiplier_fixedpoint_perchannel[row];
    shift = params.multiplier_exponent_perchannel[row];
  } else {
    multiplier = params.multiplier_fixedpoint;
    shift = params.multiplier_exponent;
  }
  acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
  acc += static_cast<int32_t>(dst_params.zero_point);
  acc = std::max(acc, static_cast<int32_t>(params.clamp_min));
  acc = std::min(acc, static_cast<int32_t>(params.clamp_max));
  return static_cast<DstScalar>(acc);
}

#endif  // USE_RVV

inline bool RvvFloatGemm(const MatrixParams<float>& lhs_params,
                         const float* lhs_data,
                         const MatrixParams<float>& rhs_params,
                         const float* rhs_data,
                         const MatrixParams<float>& dst_params, float* dst_data,
                         const GemmParams<float, float>& params) {
#ifdef USE_RVV
  if (lhs_params.zero_point != 0.0f || rhs_params.zero_point != 0.0f ||
      dst_params.zero_point != 0.0f) {
    return false;
  }
  if (!RvvGemmLayoutIsSupported(lhs_params, rhs_params, dst_params)) {
    return false;
  }

  const int rows = dst_params.rows;
  const int cols = dst_params.cols;
  const int depth = lhs_params.cols;
  const size_t acc_vl = __riscv_vsetvl_e32m4(depth);
  constexpr int kRowsPerBlock = 4;
  for (int col = 0; col < cols; ++col) {
    const float* rhs_col = rhs_data + col * depth;
    float* dst_col = dst_data + col * rows;
    int row = 0;
    for (; row <= rows - kRowsPerBlock; row += kRowsPerBlock) {
      const float* lhs_row0 = lhs_data + row * depth;
      const float* lhs_row1 = lhs_row0 + depth;
      const float* lhs_row2 = lhs_row1 + depth;
      const float* lhs_row3 = lhs_row2 + depth;
      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, acc_vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, acc_vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, acc_vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, acc_vl);
      for (int d = 0; d < depth;) {
        const size_t vl = __riscv_vsetvl_e32m4(depth - d);
        const vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_col + d, vl);
        acc0 = __riscv_vfmacc_vv_f32m4_tu(
            acc0, __riscv_vle32_v_f32m4(lhs_row0 + d, vl), rhs, vl);
        acc1 = __riscv_vfmacc_vv_f32m4_tu(
            acc1, __riscv_vle32_v_f32m4(lhs_row1 + d, vl), rhs, vl);
        acc2 = __riscv_vfmacc_vv_f32m4_tu(
            acc2, __riscv_vle32_v_f32m4(lhs_row2 + d, vl), rhs, vl);
        acc3 = __riscv_vfmacc_vv_f32m4_tu(
            acc3, __riscv_vle32_v_f32m4(lhs_row3 + d, vl), rhs, vl);
        d += vl;
      }
      float reduced[kRowsPerBlock] = {
          RvvReduceSum(acc0, acc_vl),
          RvvReduceSum(acc1, acc_vl),
          RvvReduceSum(acc2, acc_vl),
          RvvReduceSum(acc3, acc_vl),
      };
      for (int i = 0; i < kRowsPerBlock; ++i) {
        if (params.bias != nullptr) {
          reduced[i] += params.bias[row + i];
        }
        reduced[i] =
            std::max(params.clamp_min, std::min(params.clamp_max, reduced[i]));
        dst_col[row + i] = reduced[i];
      }
    }
    for (; row < rows; ++row) {
      const float* lhs_row = lhs_data + row * depth;
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, acc_vl);
      for (int d = 0; d < depth;) {
        const size_t vl = __riscv_vsetvl_e32m4(depth - d);
        const vfloat32m4_t lhs = __riscv_vle32_v_f32m4(lhs_row + d, vl);
        const vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_col + d, vl);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, lhs, rhs, vl);
        d += vl;
      }
      float reduced = RvvReduceSum(acc, acc_vl);
      if (params.bias != nullptr) {
        reduced += params.bias[row];
      }
      reduced = std::max(params.clamp_min, std::min(params.clamp_max, reduced));
      dst_col[row] = reduced;
    }
  }
  return true;
#else
  (void)lhs_params;
  (void)lhs_data;
  (void)rhs_params;
  (void)rhs_data;
  (void)dst_params;
  (void)dst_data;
  (void)params;
  return false;
#endif  // USE_RVV
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
inline bool RvvQuantizedGemm(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<int32_t, DstScalar, quantization_flavor>& params) {
#ifdef USE_RVV
  static_assert(std::is_same<LhsScalar, std::int8_t>::value ||
                    std::is_same<LhsScalar, std::uint8_t>::value,
                "");
  static_assert(std::is_same<RhsScalar, std::int8_t>::value ||
                    std::is_same<RhsScalar, std::uint8_t>::value,
                "");
  static_assert(std::is_same<DstScalar, std::int8_t>::value ||
                    std::is_same<DstScalar, std::uint8_t>::value,
                "");
  static_assert(quantization_flavor ==
                        QuantizationFlavor::kIntegerWithUniformMultiplier ||
                    quantization_flavor ==
                        QuantizationFlavor::kIntegerWithPerRowMultiplier,
                "");
  if (!RvvGemmLayoutIsSupported(lhs_params, rhs_params, dst_params)) {
    return false;
  }
  if constexpr (quantization_flavor ==
                QuantizationFlavor::kIntegerWithPerRowMultiplier) {
    if (params.multiplier_fixedpoint_perchannel == nullptr ||
        params.multiplier_exponent_perchannel == nullptr) {
      return false;
    }
  }

  const int rows = dst_params.rows;
  const int cols = dst_params.cols;
  const int depth = lhs_params.cols;
  const size_t acc_vl = __riscv_vsetvl_e8m1(depth);
  constexpr int kRowsPerBlock = 4;
  for (int col = 0; col < cols; ++col) {
    const RhsScalar* rhs_col = rhs_data + col * depth;
    DstScalar* dst_col = dst_data + col * rows;
    int row = 0;
    for (; row <= rows - kRowsPerBlock; row += kRowsPerBlock) {
      const LhsScalar* lhs_row0 = lhs_data + row * depth;
      const LhsScalar* lhs_row1 = lhs_row0 + depth;
      const LhsScalar* lhs_row2 = lhs_row1 + depth;
      const LhsScalar* lhs_row3 = lhs_row2 + depth;
      vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, acc_vl);
      vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, acc_vl);
      vint32m4_t acc2 = __riscv_vmv_v_x_i32m4(0, acc_vl);
      vint32m4_t acc3 = __riscv_vmv_v_x_i32m4(0, acc_vl);
      for (int d = 0; d < depth;) {
        const size_t vl = __riscv_vsetvl_e8m1(depth - d);
        const vint16m2_t rhs = RvvLoad8AndSubtractZeroPoint(
            rhs_col + d, rhs_params.zero_point, vl);
        const vint16m2_t lhs0 = RvvLoad8AndSubtractZeroPoint(
            lhs_row0 + d, lhs_params.zero_point, vl);
        const vint16m2_t lhs1 = RvvLoad8AndSubtractZeroPoint(
            lhs_row1 + d, lhs_params.zero_point, vl);
        const vint16m2_t lhs2 = RvvLoad8AndSubtractZeroPoint(
            lhs_row2 + d, lhs_params.zero_point, vl);
        const vint16m2_t lhs3 = RvvLoad8AndSubtractZeroPoint(
            lhs_row3 + d, lhs_params.zero_point, vl);
        acc0 = __riscv_vwmacc_vv_i32m4_tu(acc0, lhs0, rhs, vl);
        acc1 = __riscv_vwmacc_vv_i32m4_tu(acc1, lhs1, rhs, vl);
        acc2 = __riscv_vwmacc_vv_i32m4_tu(acc2, lhs2, rhs, vl);
        acc3 = __riscv_vwmacc_vv_i32m4_tu(acc3, lhs3, rhs, vl);
        d += vl;
      }
      const int32_t reduced[kRowsPerBlock] = {
          RvvReduceSum(acc0, acc_vl),
          RvvReduceSum(acc1, acc_vl),
          RvvReduceSum(acc2, acc_vl),
          RvvReduceSum(acc3, acc_vl),
      };
      for (int i = 0; i < kRowsPerBlock; ++i) {
        dst_col[row + i] = RvvFinalizeQuantizedAccumulator(reduced[i], row + i,
                                                           dst_params, params);
      }
    }
    for (; row < rows; ++row) {
      const LhsScalar* lhs_row = lhs_data + row * depth;
      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, acc_vl);
      for (int d = 0; d < depth;) {
        const size_t vl = __riscv_vsetvl_e8m1(depth - d);
        const vint16m2_t lhs = RvvLoad8AndSubtractZeroPoint(
            lhs_row + d, lhs_params.zero_point, vl);
        const vint16m2_t rhs = RvvLoad8AndSubtractZeroPoint(
            rhs_col + d, rhs_params.zero_point, vl);
        acc = __riscv_vwmacc_vv_i32m4_tu(acc, lhs, rhs, vl);
        d += vl;
      }
      dst_col[row] = RvvFinalizeQuantizedAccumulator(RvvReduceSum(acc, acc_vl),
                                                     row, dst_params, params);
    }
  }
  return true;
#else
  (void)lhs_params;
  (void)lhs_data;
  (void)rhs_params;
  (void)rhs_data;
  (void)dst_params;
  (void)dst_data;
  (void)params;
  return false;
#endif  // USE_RVV
}

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
