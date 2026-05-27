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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_SCALAR_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_SCALAR_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/internal/common.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

template <typename Scalar>
inline int ScalarMatrixIndex(const MatrixParams<Scalar>& params, int row,
                             int col) {
  return params.order == Order::kColMajor ? col * params.rows + row
                                          : row * params.cols + col;
}

template <typename Scalar>
inline Scalar ScalarMatrixValue(const MatrixParams<Scalar>& params,
                                const Scalar* data, int row, int col) {
  return data[ScalarMatrixIndex(params, row, col)];
}

template <typename Scalar>
inline void ScalarMatrixStore(const MatrixParams<Scalar>& params, Scalar* data,
                              int row, int col, Scalar value) {
  data[ScalarMatrixIndex(params, row, col)] = value;
}

template <typename Scalar>
inline int32_t ScalarValueMinusZeroPoint(Scalar value, Scalar zero_point) {
  return static_cast<int32_t>(value) - static_cast<int32_t>(zero_point);
}

template <typename DstScalar, QuantizationFlavor quantization_flavor>
inline DstScalar ScalarFinalizeQuantizedAccumulator(
    int32_t acc, int row, const MatrixParams<DstScalar>& dst_params,
    const GemmParams<int32_t, DstScalar, quantization_flavor>& params) {
  if (params.bias != nullptr) {
    acc += params.bias[row];
  }

  int32_t multiplier = params.multiplier_fixedpoint;
  int shift = params.multiplier_exponent;
  if constexpr (quantization_flavor ==
                QuantizationFlavor::kIntegerWithPerRowMultiplier) {
    if (params.multiplier_fixedpoint_perchannel == nullptr ||
        params.multiplier_exponent_perchannel == nullptr) {
      return DstScalar();
    }
    multiplier = params.multiplier_fixedpoint_perchannel[row];
    shift = params.multiplier_exponent_perchannel[row];
  }

  acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
  acc += static_cast<int32_t>(dst_params.zero_point);
  acc = std::max(acc, static_cast<int32_t>(params.clamp_min));
  acc = std::min(acc, static_cast<int32_t>(params.clamp_max));
  acc = std::max(acc, static_cast<int32_t>(std::numeric_limits<DstScalar>::lowest()));
  acc = std::min(acc, static_cast<int32_t>(std::numeric_limits<DstScalar>::max()));
  return static_cast<DstScalar>(acc);
}

inline bool ScalarFloatGemm(const MatrixParams<float>& lhs_params,
                            const float* lhs_data,
                            const MatrixParams<float>& rhs_params,
                            const float* rhs_data,
                            const MatrixParams<float>& dst_params,
                            float* dst_data,
                            const GemmParams<float, float>& params) {
  if (lhs_params.zero_point != 0.0f || rhs_params.zero_point != 0.0f ||
      dst_params.zero_point != 0.0f) {
    return false;
  }

  const int rows = dst_params.rows;
  const int cols = dst_params.cols;
  const int depth = lhs_params.cols;
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      float acc = 0.0f;
      for (int d = 0; d < depth; ++d) {
        acc += ScalarMatrixValue(lhs_params, lhs_data, row, d) *
               ScalarMatrixValue(rhs_params, rhs_data, d, col);
      }
      if (params.bias != nullptr) {
        acc += params.bias[row];
      }
      acc = std::max(params.clamp_min, std::min(params.clamp_max, acc));
      ScalarMatrixStore(dst_params, dst_data, row, col, acc);
    }
  }
  return true;
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
inline bool ScalarQuantizedGemm(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<int32_t, DstScalar, quantization_flavor>& params) {
  static_assert(std::is_integral<LhsScalar>::value, "");
  static_assert(std::is_integral<RhsScalar>::value, "");
  static_assert(std::is_integral<DstScalar>::value, "");
  static_assert(quantization_flavor ==
                        QuantizationFlavor::kIntegerWithUniformMultiplier ||
                    quantization_flavor ==
                        QuantizationFlavor::kIntegerWithPerRowMultiplier,
                "");
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
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      int32_t acc = 0;
      for (int d = 0; d < depth; ++d) {
        const int32_t lhs = ScalarValueMinusZeroPoint(
            ScalarMatrixValue(lhs_params, lhs_data, row, d),
            lhs_params.zero_point);
        const int32_t rhs = ScalarValueMinusZeroPoint(
            ScalarMatrixValue(rhs_params, rhs_data, d, col),
            rhs_params.zero_point);
        acc += lhs * rhs;
      }
      ScalarMatrixStore(dst_params, dst_data, row, col,
                        ScalarFinalizeQuantizedAccumulator(
                            acc, row, dst_params, params));
    }
  }
  return true;
}

template <typename LhsScalar, typename RhsScalar,
          QuantizationFlavor quantization_flavor>
inline bool ScalarRawAccumulatorGemm(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<int32_t>& dst_params, int32_t* dst_data,
    const GemmParams<int32_t, int32_t, quantization_flavor>& params) {
  static_assert(std::is_integral<LhsScalar>::value, "");
  static_assert(std::is_integral<RhsScalar>::value, "");

  const int rows = dst_params.rows;
  const int cols = dst_params.cols;
  const int depth = lhs_params.cols;
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      int32_t acc = 0;
      for (int d = 0; d < depth; ++d) {
        const int32_t lhs = ScalarValueMinusZeroPoint(
            ScalarMatrixValue(lhs_params, lhs_data, row, d),
            lhs_params.zero_point);
        const int32_t rhs = ScalarValueMinusZeroPoint(
            ScalarMatrixValue(rhs_params, rhs_data, d, col),
            rhs_params.zero_point);
        acc += lhs * rhs;
      }
      if (params.bias != nullptr) {
        acc += params.bias[row];
      }
      ScalarMatrixStore(dst_params, dst_data, row, col, acc);
    }
  }
  return true;
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
inline bool ScalarGemm(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params) {
  if constexpr (std::is_same<LhsScalar, float>::value &&
                std::is_same<RhsScalar, float>::value &&
                std::is_same<AccumScalar, float>::value &&
                std::is_same<DstScalar, float>::value &&
                quantization_flavor == QuantizationFlavor::kFloatingPoint) {
    return ScalarFloatGemm(lhs_params, lhs_data, rhs_params, rhs_data,
                           dst_params, dst_data, params);
  } else if constexpr (std::is_integral<LhsScalar>::value &&
                       std::is_integral<RhsScalar>::value &&
                       std::is_same<AccumScalar, int32_t>::value &&
                       std::is_integral<DstScalar>::value &&
                       !std::is_same<DstScalar, int32_t>::value &&
                       (quantization_flavor ==
                            QuantizationFlavor::kIntegerWithUniformMultiplier ||
                        quantization_flavor ==
                            QuantizationFlavor::kIntegerWithPerRowMultiplier)) {
    return ScalarQuantizedGemm(lhs_params, lhs_data, rhs_params, rhs_data,
                               dst_params, dst_data, params);
  } else if constexpr (std::is_integral<LhsScalar>::value &&
                       std::is_integral<RhsScalar>::value &&
                       std::is_same<AccumScalar, int32_t>::value &&
                       std::is_same<DstScalar, int32_t>::value) {
    return ScalarRawAccumulatorGemm(lhs_params, lhs_data, rhs_params, rhs_data,
                                    dst_params, dst_data, params);
  }
  return false;
}

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_SCALAR_H_
