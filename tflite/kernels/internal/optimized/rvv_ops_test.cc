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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "tflite/kernels/cpu_backend_gemm.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/conv.h"
#include "tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h"
#include "tflite/kernels/internal/optimized/integer_ops/fully_connected.h"
#include "tflite/kernels/internal/optimized/integer_ops/leaky_relu.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/reduce.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/internal/reference/add.h"
#include "tflite/kernels/internal/reference/conv.h"
#include "tflite/kernels/internal/reference/div.h"
#include "tflite/kernels/internal/reference/fully_connected.h"
#include "tflite/kernels/internal/reference/gelu.h"
#include "tflite/kernels/internal/reference/integer_ops/add.h"
#include "tflite/kernels/internal/reference/integer_ops/conv.h"
#include "tflite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tflite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tflite/kernels/internal/reference/integer_ops/mul.h"
#include "tflite/kernels/internal/reference/integer_ops/pooling.h"
#include "tflite/kernels/internal/reference/mul.h"
#include "tflite/kernels/internal/reference/portable_tensor_utils.h"
#include "tflite/kernels/internal/reference/quantize.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/reference/requantize.h"
#include "tflite/kernels/internal/reference/sub.h"
#include "tflite/kernels/internal/reference/svdf.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/stablehlo_elementwise.h"
#ifdef RVV_OPS_TEST_WITH_SINGLE_OP_MODEL
#include "tflite/kernels/test_util.h"
#endif
#include "tflite/schema/schema_generated.h"

namespace tflite {

namespace ops {
namespace builtin {
TfLiteRegistration* Register_STABLEHLO_ADD();
TfLiteRegistration* Register_STABLEHLO_MULTIPLY();
TfLiteRegistration* Register_STABLEHLO_MAXIMUM();
TfLiteRegistration* Register_STABLEHLO_MINIMUM();
TfLiteRegistration* Register_STABLEHLO_AND();
}  // namespace builtin
}  // namespace ops

namespace {

#ifdef USE_RVV

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

struct OneDimTfLiteIntArray {
  int size;
  int data[1];
};

struct TwoDimTfLiteIntArray {
  int size;
  int data[2];
};

float Clamp(float value, float min, float max) {
  return std::min(max, std::max(min, value));
}

std::vector<float> MakeInput(int size, float offset) {
  std::vector<float> values(size);
  for (int i = 0; i < size; ++i) {
    const float centered = static_cast<float>((i % 13) - 6);
    values[i] = centered * 0.75f + offset;
  }
  return values;
}

std::vector<float> MakeSpecialFloatInput(int size, float offset) {
  std::vector<float> values = MakeInput(size, offset);
  const float specials[] = {
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
  };
  const int count =
      std::min(size, static_cast<int>(sizeof(specials) / sizeof(specials[0])));
  for (int i = 0; i < count; ++i) {
    values[i] = specials[i];
  }
  return values;
}

void MakeStablehloSpecialFloatInputs(int size, std::vector<float>* input1,
                                     std::vector<float>* input2) {
  *input1 = MakeInput(size, 0.5f);
  *input2 = MakeInput(size, -1.25f);
  const float lhs[] = {
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
  };
  const float rhs[] = {
      -0.0f,
      0.0f,
      1.0f,
      2.0f,
      -1.0f,
      1.0f,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::denorm_min(),
  };
  const int count =
      std::min(size, static_cast<int>(sizeof(lhs) / sizeof(lhs[0])));
  for (int i = 0; i < count; ++i) {
    (*input1)[i] = lhs[i];
    (*input2)[i] = rhs[i];
  }
}

void ExpectFloatNearOrSpecial(const std::vector<float>& actual,
                              const std::vector<float>& expected,
                              float tolerance = 1e-6f) {
  ASSERT_EQ(actual.size(), expected.size());
  for (int i = 0; i < expected.size(); ++i) {
    if (std::isnan(expected[i])) {
      EXPECT_TRUE(std::isnan(actual[i])) << "i=" << i;
    } else if (std::isinf(expected[i])) {
      EXPECT_EQ(actual[i], expected[i]) << "i=" << i;
    } else {
      EXPECT_THAT(actual[i], FloatNear(expected[i], tolerance)) << "i=" << i;
    }
  }
}

float ReferenceLogisticFloat(float value) {
  constexpr float kCutoffUpper = 16.619047164916992188f;
  constexpr float kCutoffLower = -9.0f;
  if (value > kCutoffUpper) {
    return 1.0f;
  }
  if (value < kCutoffLower) {
    return std::exp(value);
  }
  return 1.0f / (1.0f + std::exp(-value));
}

std::vector<float> MakeActivationInput(int size) {
  std::vector<float> input = MakeInput(size, -0.1875f);
  const float special_values[] = {
      -20.0f,
      -16.619047164916992188f,
      -9.0f,
      -3.0f,
      -1.0f,
      -0.0f,
      0.0f,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      1.0f,
      3.0f,
      9.0f,
      16.619047164916992188f,
      20.0f,
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
  };
  const int count = std::min(size, static_cast<int>(sizeof(special_values) /
                                                    sizeof(special_values[0])));
  for (int i = 0; i < count; ++i) {
    input[i] = special_values[i];
  }
  return input;
}

std::vector<int8_t> MakeInt8Input(int size, int offset) {
  std::vector<int8_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<int8_t>(((i * 37 + offset) % 255) - 127);
  }
  return values;
}

void MakeSpecialInt8Inputs(int size, std::vector<int8_t>* input1,
                           std::vector<int8_t>* input2) {
  *input1 = MakeInt8Input(size, 17);
  *input2 = MakeInt8Input(size, 91);
  const int8_t lhs[] = {
      std::numeric_limits<int8_t>::min(),
      std::numeric_limits<int8_t>::max(),
      -1,
      0,
      1,
      64,
      -64,
      127,
  };
  const int8_t rhs[] = {1, -1, 1, 0, 1, 1, 1, 0};
  const int count =
      std::min(size, static_cast<int>(sizeof(lhs) / sizeof(lhs[0])));
  for (int i = 0; i < count; ++i) {
    (*input1)[i] = lhs[i];
    (*input2)[i] = rhs[i];
  }
}

std::vector<uint8_t> MakeUint8Input(int size, int offset) {
  std::vector<uint8_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<uint8_t>((i * 53 + offset) % 256);
  }
  return values;
}

std::vector<int32_t> MakeInt32Input(int size, int offset) {
  std::vector<int32_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = ((i * 7919 + offset) % 2001) - 1000;
  }
  return values;
}

std::vector<int32_t> MakeNonZeroInt32Input(int size, int offset) {
  std::vector<int32_t> values(size);
  for (int i = 0; i < size; ++i) {
    int32_t value = ((i * 1543 + offset) % 199) - 99;
    if (value == 0) {
      value = 17;
    }
    values[i] = value;
  }
  return values;
}

std::vector<int16_t> MakeInt16Input(int size, int offset) {
  std::vector<int16_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<int16_t>(((i * 2053 + offset) % 65535) - 32767);
  }
  return values;
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          cpu_backend_gemm::QuantizationFlavor quantization_flavor>
void ReferenceQuantizedCpuBackendGemm(
    const cpu_backend_gemm::MatrixParams<LhsScalar>& lhs_params,
    const LhsScalar* lhs_data,
    const cpu_backend_gemm::MatrixParams<RhsScalar>& rhs_params,
    const RhsScalar* rhs_data,
    const cpu_backend_gemm::MatrixParams<DstScalar>& dst_params,
    DstScalar* dst_data,
    const cpu_backend_gemm::GemmParams<int32_t, DstScalar, quantization_flavor>&
        params) {
  for (int col = 0; col < dst_params.cols; ++col) {
    for (int row = 0; row < dst_params.rows; ++row) {
      int32_t acc = 0;
      for (int depth = 0; depth < lhs_params.cols; ++depth) {
        const int32_t lhs =
            static_cast<int32_t>(lhs_data[row * lhs_params.cols + depth]) -
            static_cast<int32_t>(lhs_params.zero_point);
        const int32_t rhs =
            static_cast<int32_t>(rhs_data[col * rhs_params.rows + depth]) -
            static_cast<int32_t>(rhs_params.zero_point);
        acc += lhs * rhs;
      }
      if (params.bias != nullptr) {
        acc += params.bias[row];
      }
      int32_t multiplier;
      int shift;
      if constexpr (quantization_flavor ==
                    cpu_backend_gemm::QuantizationFlavor::
                        kIntegerWithPerRowMultiplier) {
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
      dst_data[col * dst_params.rows + row] = static_cast<DstScalar>(acc);
    }
  }
}

ArithmeticParams MakeInt8Params() {
  ArithmeticParams params;
  params.left_shift = 20;
  params.input1_offset = 17;
  params.input2_offset = -23;
  params.input1_multiplier = 1073741824;
  params.input2_multiplier = 1610612736;
  params.output_multiplier = 1342177280;
  params.input1_shift = -2;
  params.input2_shift = -1;
  params.output_shift = -3;
  params.output_offset = 9;
  params.quantized_activation_min = -96;
  params.quantized_activation_max = 101;
  return params;
}

ArithmeticParams MakeUint8Params() {
  ArithmeticParams params;
  params.left_shift = 20;
  params.input1_offset = -131;
  params.input2_offset = -117;
  params.input1_multiplier = 1073741824;
  params.input2_multiplier = 1610612736;
  params.output_multiplier = 1342177280;
  params.input1_shift = -2;
  params.input2_shift = -1;
  params.output_shift = -3;
  params.output_offset = 121;
  params.quantized_activation_min = 11;
  params.quantized_activation_max = 233;
  return params;
}

ArithmeticParams MakeInt8MulParams(int output_shift) {
  ArithmeticParams params;
  params.input1_offset = 5;
  params.input2_offset = -7;
  params.output_multiplier = 1073741824;
  params.output_shift = output_shift;
  params.output_offset = -3;
  params.quantized_activation_min = -100;
  params.quantized_activation_max = 101;
  return params;
}

ArithmeticParams MakeUint8MulParams(int output_shift) {
  ArithmeticParams params;
  params.input1_offset = -128;
  params.input2_offset = -123;
  params.output_multiplier = 1073741824;
  params.output_shift = output_shift;
  params.output_offset = 127;
  params.quantized_activation_min = 7;
  params.quantized_activation_max = 241;
  return params;
}

ArithmeticParams MakeInt16Params() {
  ArithmeticParams params;
  params.left_shift = 15;
  params.input1_offset = 97;
  params.input2_offset = -121;
  params.input1_multiplier = 1073741824;
  params.input2_multiplier = 1610612736;
  params.output_multiplier = 1342177280;
  params.input1_shift = -2;
  params.input2_shift = -1;
  params.output_shift = -3;
  params.output_offset = 211;
  params.quantized_activation_min = -24000;
  params.quantized_activation_max = 23000;
  return params;
}

ReluParams MakeInt8ReluParams() {
  ReluParams params;
  params.input_offset = -3;
  params.output_offset = -3;
  params.output_multiplier = 1073741824;
  params.output_shift = 1;
  params.quantized_activation_min = -3;
  params.quantized_activation_max = 127;
  return params;
}

ReluParams MakeUint8ReluParams() {
  ReluParams params;
  params.input_offset = 128;
  params.output_offset = 128;
  params.output_multiplier = 1073741824;
  params.output_shift = 1;
  params.quantized_activation_min = 128;
  params.quantized_activation_max = 255;
  return params;
}

ReluParams MakeInt16ReluParams() {
  ReluParams params;
  params.input_offset = 0;
  params.output_offset = 0;
  params.output_multiplier = 1073741824;
  params.output_shift = 1;
  params.quantized_activation_min = 0;
  params.quantized_activation_max = 32767;
  return params;
}

LeakyReluParams MakeFloatLeakyReluParams() {
  LeakyReluParams params;
  params.alpha = 0.125f;
  return params;
}

LeakyReluParams MakeInt16LeakyReluParams() {
  LeakyReluParams params;
  params.input_offset = 0;
  params.output_offset = 0;
  params.output_multiplier_alpha = 1073741824;
  params.output_shift_alpha = -2;
  params.output_multiplier_identity = 1073741824;
  params.output_shift_identity = 1;
  return params;
}

FullyConnectedParams MakeInt8FullyConnectedParams() {
  FullyConnectedParams params;
  params.input_offset = 9;
  params.weights_offset = -5;
  params.output_offset = -7;
  params.output_multiplier = 1234567890;
  params.output_shift = -8;
  params.quantized_activation_min = -101;
  params.quantized_activation_max = 97;
  return params;
}

FullyConnectedParams MakeInt8FullyConnectedPerChannelParams() {
  FullyConnectedParams params;
  params.input_offset = -11;
  params.output_offset = 13;
  params.quantized_activation_min = -95;
  params.quantized_activation_max = 99;
  return params;
}

ConvParams MakeInt8ConvPerChannelParams() {
  ConvParams params;
  params.input_offset = 7;
  params.output_offset = -9;
  params.stride_height = 1;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.padding_values.height = 0;
  params.padding_values.width = 0;
  params.quantized_activation_min = -103;
  params.quantized_activation_max = 101;
  return params;
}

DepthwiseParams MakeInt8DepthwiseConvPerChannelParams() {
  DepthwiseParams params;
  params.padding_values.height = 1;
  params.padding_values.width = 1;
  params.stride_height = 1;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.depth_multiplier = 1;
  params.input_offset = -13;
  params.output_offset = 11;
  params.quantized_activation_min = -97;
  params.quantized_activation_max = 103;
  return params;
}

PoolParams MakeFloatPoolParams() {
  PoolParams params;
  params.padding_values.height = 1;
  params.padding_values.width = 1;
  params.stride_height = 1;
  params.stride_width = 2;
  params.filter_height = 2;
  params.filter_width = 3;
  params.float_activation_min = -1.75f;
  params.float_activation_max = 2.25f;
  return params;
}

PoolParams MakeUint8PoolParams() {
  PoolParams params;
  params.padding_values.height = 1;
  params.padding_values.width = 1;
  params.stride_height = 1;
  params.stride_width = 2;
  params.filter_height = 2;
  params.filter_width = 3;
  params.quantized_activation_min = 13;
  params.quantized_activation_max = 231;
  return params;
}

PoolParams MakeInt8PoolParams() {
  PoolParams params;
  params.padding_values.height = 1;
  params.padding_values.width = 1;
  params.stride_height = 1;
  params.stride_width = 2;
  params.filter_height = 2;
  params.filter_width = 3;
  params.quantized_activation_min = -80;
  params.quantized_activation_max = 91;
  return params;
}

// Reuse these lengths for future RVV tests so every vectorized kernel gets
// coverage around its real strip-mining boundary.
std::vector<int> VectorLengthsAroundVlmax(int vlmax) {
  std::vector<int> lengths = {
      0,
      1,
      2,
      vlmax - 1,
      vlmax,
      vlmax + 1,
      2 * vlmax - 1,
      2 * vlmax,
      2 * vlmax + 3,
  };
  lengths.erase(std::remove_if(lengths.begin(), lengths.end(),
                               [](int length) { return length < 0; }),
                lengths.end());
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  return lengths;
}

std::vector<int> Float32M4VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e32m4()));
}

std::vector<int> Int8M1VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e8m1()));
}

std::vector<int> Int8M4VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e8m4()));
}

std::vector<int> Int8M8VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e8m8()));
}

std::vector<int> UInt8M8VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e8m8()));
}

std::vector<int> Int32M4VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e32m4()));
}

std::vector<int> Int16M2VectorLengths() {
  return VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e16m2()));
}

template <typename T>
std::vector<int> ElementLengthsAroundByteVlmax() {
  std::vector<int> lengths;
  for (int bytes :
       VectorLengthsAroundVlmax(static_cast<int>(__riscv_vsetvlmax_e8m8()))) {
    if (bytes == 0) {
      lengths.push_back(0);
    } else {
      lengths.push_back((bytes + static_cast<int>(sizeof(T)) - 1) /
                        static_cast<int>(sizeof(T)));
    }
  }
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  return lengths;
}

#endif  // USE_RVV

TEST(RvvOpsTest, RequiresRvvBuildToExerciseVectorPaths) {
#ifndef USE_RVV
  GTEST_SKIP() << "RVV correctness tests require USE_RVV.";
#endif
}

#ifdef USE_RVV

TEST(RvvOpsTest, AffineQuantizeInt8MatchesReferenceAcrossVectorBoundaries) {
  QuantizationParams params;
  params.zero_point = -7;
  params.scale = 0.25f;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) {
      input[i] = static_cast<float>((i % 41) - 20) * 0.125f;
    }
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_ops::AffineQuantize(params, shape, input.data(), shape,
                                  actual.data());
    reference_ops::AffineQuantize(params, shape, input.data(), shape,
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, AffineQuantizeUint8MatchesReferenceAcrossVectorBoundaries) {
  QuantizationParams params;
  params.zero_point = 119;
  params.scale = 0.125f;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) {
      input[i] = static_cast<float>((i % 47) - 23) * 0.0625f;
    }
    std::vector<uint8_t> actual(size);
    std::vector<uint8_t> expected(size);

    optimized_ops::AffineQuantize(params, shape, input.data(), shape,
                                  actual.data());
    reference_ops::AffineQuantize(params, shape, input.data(), shape,
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, AffineQuantizeInt16MatchesReferenceAcrossVectorBoundaries) {
  QuantizationParams params;
  params.zero_point = 0;
  params.scale = 0.03125f;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) {
      input[i] = static_cast<float>((i % 53) - 26) * 0.015625f;
    }
    std::vector<int16_t> actual(size);
    std::vector<int16_t> expected(size);

    optimized_ops::AffineQuantize(params, shape, input.data(), shape,
                                  actual.data());
    reference_ops::AffineQuantize(params, shape, input.data(), shape,
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, DequantizeInt8MatchesReferenceAcrossVectorBoundaries) {
  DequantizationParams params;
  params.zero_point = -5;
  params.scale = 0.03125f;

  for (int size : Int8M1VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int8_t> input = MakeInt8Input(size, 17);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Dequantize(params, shape, input.data(), shape,
                              actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] =
          static_cast<float>(params.scale * (input[i] - params.zero_point));
    }

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "size=" << size;
  }
}

TEST(RvvOpsTest, DequantizeUint8MatchesReferenceAcrossVectorBoundaries) {
  DequantizationParams params;
  params.zero_point = 127;
  params.scale = 0.015625f;

  for (int size : Int8M1VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<uint8_t> input = MakeUint8Input(size, 31);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Dequantize(params, shape, input.data(), shape,
                              actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] =
          static_cast<float>(params.scale * (input[i] - params.zero_point));
    }

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "size=" << size;
  }
}

TEST(RvvOpsTest, DequantizeInt16MatchesReferenceAcrossVectorBoundaries) {
  DequantizationParams params;
  params.zero_point = 0;
  params.scale = 0.000244140625f;

  for (int size : Int16M2VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int16_t> input = MakeInt16Input(size, 211);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Dequantize(params, shape, input.data(), shape,
                              actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] =
          static_cast<float>(params.scale * (input[i] - params.zero_point));
    }

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "size=" << size;
  }
}

TEST(RvvOpsTest, RequantizeInt8ToUint8MatchesReferenceAcrossVectorBoundaries) {
  constexpr int32_t kMultiplier = 1234567890;
  constexpr int kShift = -3;
  constexpr int32_t kInputZeroPoint = -11;
  constexpr int32_t kOutputZeroPoint = 129;

  for (int size : Int8M1VectorLengths()) {
    const std::vector<int8_t> input = MakeInt8Input(size, 73);
    std::vector<uint8_t> actual(size);
    std::vector<uint8_t> expected(size);

    optimized_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint, actual.data());
    reference_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint,
                              expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, RequantizeUint8ToInt8MatchesReferenceAcrossVectorBoundaries) {
  constexpr int32_t kMultiplier = 1325400064;
  constexpr int kShift = -2;
  constexpr int32_t kInputZeroPoint = 121;
  constexpr int32_t kOutputZeroPoint = -9;

  for (int size : Int8M1VectorLengths()) {
    const std::vector<uint8_t> input = MakeUint8Input(size, 97);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint, actual.data());
    reference_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint,
                              expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, RequantizeInt8ToInt8MatchesReferenceAcrossVectorBoundaries) {
  constexpr int32_t kMultiplier = 1073741824;
  constexpr int kShift = 1;
  constexpr int32_t kInputZeroPoint = -3;
  constexpr int32_t kOutputZeroPoint = 5;

  for (int size : Int8M1VectorLengths()) {
    const std::vector<int8_t> input = MakeInt8Input(size, 151);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint, actual.data());
    reference_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint,
                              expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, RequantizeInt32ToInt16MatchesReferenceAcrossVectorBoundaries) {
  constexpr int32_t kMultiplier = 987654321;
  constexpr int kShift = -4;
  constexpr int32_t kInputZeroPoint = 17;
  constexpr int32_t kOutputZeroPoint = -23;

  for (int size : Float32M4VectorLengths()) {
    const std::vector<int32_t> input = MakeInt32Input(size, 503);
    std::vector<int16_t> actual(size);
    std::vector<int16_t> expected(size);

    optimized_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint, actual.data());
    reference_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint,
                              expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest,
     Int8CpuBackendGemmMatchesScalarReferenceAcrossVectorBoundaries) {
  constexpr int kRows = 7;
  constexpr int kCols = 3;
  cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
  lhs_params.rows = kRows;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -5;
  cpu_backend_gemm::MatrixParams<int8_t> rhs_params;
  rhs_params.cols = kCols;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = 7;
  cpu_backend_gemm::MatrixParams<int8_t> dst_params;
  dst_params.rows = kRows;
  dst_params.cols = kCols;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = -3;
  cpu_backend_gemm::GemmParams<int32_t, int8_t> params;
  params.multiplier_fixedpoint = 1234567890;
  params.multiplier_exponent = -5;
  params.clamp_min = -103;
  params.clamp_max = 101;
  const std::vector<int32_t> bias = MakeInt32Input(kRows, 389);
  params.bias = bias.data();

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    lhs_params.cols = depth;
    rhs_params.rows = depth;
    std::vector<int8_t> lhs = MakeInt8Input(kRows * depth, 17);
    std::vector<int8_t> rhs = MakeInt8Input(kCols * depth, 97);
    lhs[0] = std::numeric_limits<int8_t>::min();
    lhs[1 % lhs.size()] = std::numeric_limits<int8_t>::max();
    rhs[0] = std::numeric_limits<int8_t>::min();
    rhs[1 % rhs.size()] = std::numeric_limits<int8_t>::max();
    std::vector<int8_t> actual(kRows * kCols);
    std::vector<int8_t> expected(kRows * kCols);

    cpu_backend_gemm::Gemm(lhs_params, lhs.data(), rhs_params, rhs.data(),
                           dst_params, actual.data(), params, nullptr);
    ReferenceQuantizedCpuBackendGemm(lhs_params, lhs.data(), rhs_params,
                                     rhs.data(), dst_params, expected.data(),
                                     params);

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest,
     Int8CpuBackendGemmPerRowMatchesScalarReferenceAcrossVectorBoundaries) {
  constexpr int kRows = 5;
  constexpr int kCols = 4;
  cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
  lhs_params.rows = kRows;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = 3;
  cpu_backend_gemm::MatrixParams<int8_t> rhs_params;
  rhs_params.cols = kCols;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -11;
  cpu_backend_gemm::MatrixParams<int8_t> dst_params;
  dst_params.rows = kRows;
  dst_params.cols = kCols;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = 9;
  cpu_backend_gemm::GemmParams<
      int32_t, int8_t,
      cpu_backend_gemm::QuantizationFlavor::kIntegerWithPerRowMultiplier>
      params;
  const int32_t multiplier[kRows] = {1073741824, 1234567890, 1342177280,
                                     987654321, 1503238554};
  const int shift[kRows] = {-4, -3, -5, -2, -6};
  const std::vector<int32_t> bias = MakeInt32Input(kRows, 577);
  params.multiplier_fixedpoint_perchannel = multiplier;
  params.multiplier_exponent_perchannel = shift;
  params.bias = bias.data();
  params.clamp_min = -96;
  params.clamp_max = 111;

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    lhs_params.cols = depth;
    rhs_params.rows = depth;
    std::vector<int8_t> lhs = MakeInt8Input(kRows * depth, 31);
    std::vector<int8_t> rhs = MakeInt8Input(kCols * depth, 151);
    lhs[0] = std::numeric_limits<int8_t>::min();
    rhs[0] = std::numeric_limits<int8_t>::max();
    std::vector<int8_t> actual(kRows * kCols);
    std::vector<int8_t> expected(kRows * kCols);

    cpu_backend_gemm::Gemm(lhs_params, lhs.data(), rhs_params, rhs.data(),
                           dst_params, actual.data(), params, nullptr);
    ReferenceQuantizedCpuBackendGemm(lhs_params, lhs.data(), rhs_params,
                                     rhs.data(), dst_params, expected.data(),
                                     params);

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest,
     Uint8CpuBackendGemmMatchesScalarReferenceAcrossVectorBoundaries) {
  constexpr int kRows = 7;
  constexpr int kCols = 3;
  cpu_backend_gemm::MatrixParams<uint8_t> lhs_params;
  lhs_params.rows = kRows;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = 127;
  cpu_backend_gemm::MatrixParams<uint8_t> rhs_params;
  rhs_params.cols = kCols;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = 119;
  cpu_backend_gemm::MatrixParams<uint8_t> dst_params;
  dst_params.rows = kRows;
  dst_params.cols = kCols;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = 121;
  cpu_backend_gemm::GemmParams<int32_t, uint8_t> params;
  params.multiplier_fixedpoint = 1325400064;
  params.multiplier_exponent = -4;
  params.clamp_min = 13;
  params.clamp_max = 239;
  const std::vector<int32_t> bias = MakeInt32Input(kRows, 719);
  params.bias = bias.data();

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    lhs_params.cols = depth;
    rhs_params.rows = depth;
    std::vector<uint8_t> lhs = MakeUint8Input(kRows * depth, 41);
    std::vector<uint8_t> rhs = MakeUint8Input(kCols * depth, 211);
    lhs[0] = std::numeric_limits<uint8_t>::min();
    lhs[1 % lhs.size()] = std::numeric_limits<uint8_t>::max();
    rhs[0] = std::numeric_limits<uint8_t>::min();
    rhs[1 % rhs.size()] = std::numeric_limits<uint8_t>::max();
    std::vector<uint8_t> actual(kRows * kCols);
    std::vector<uint8_t> expected(kRows * kCols);

    cpu_backend_gemm::Gemm(lhs_params, lhs.data(), rhs_params, rhs.data(),
                           dst_params, actual.data(), params, nullptr);
    ReferenceQuantizedCpuBackendGemm(lhs_params, lhs.data(), rhs_params,
                                     rhs.data(), dst_params, expected.data(),
                                     params);

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, Int8FullyConnectedMatchesReferenceAcrossVectorBoundaries) {
  const FullyConnectedParams params = MakeInt8FullyConnectedParams();
  constexpr int kBatches = 3;
  constexpr int kOutputDepth = 7;
  const RuntimeShape bias_shape({kOutputDepth});
  const std::vector<int32_t> bias = MakeInt32Input(kOutputDepth, 41);

  for (int accum_depth : Int8M1VectorLengths()) {
    if (accum_depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({kBatches, accum_depth});
    const RuntimeShape filter_shape({kOutputDepth, accum_depth});
    const RuntimeShape output_shape({kBatches, kOutputDepth});
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 19);
    const std::vector<int8_t> filter =
        MakeInt8Input(filter_shape.FlatSize(), 83);
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize());

    optimized_integer_ops::FullyConnected(
        params, input_shape, input.data(), filter_shape, filter.data(),
        bias_shape, bias.data(), output_shape, actual.data(), nullptr);
    reference_integer_ops::FullyConnected(
        params, input_shape, input.data(), filter_shape, filter.data(),
        bias_shape, bias.data(), output_shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "accum_depth=" << accum_depth;
  }
}

TEST(RvvOpsTest,
     Int8FullyConnectedPerChannelMatchesReferenceAcrossVectorBoundaries) {
  const FullyConnectedParams params = MakeInt8FullyConnectedPerChannelParams();
  constexpr int kBatches = 2;
  constexpr int kOutputDepth = 5;
  const RuntimeShape bias_shape({kOutputDepth});
  const std::vector<int32_t> bias = MakeInt32Input(kOutputDepth, 157);
  const int32_t output_multiplier[kOutputDepth] = {
      1073741824, 1234567890, 1342177280, 987654321, 1503238554};
  const int output_shift[kOutputDepth] = {-4, -3, -5, -2, -6};

  for (int accum_depth : Int8M1VectorLengths()) {
    if (accum_depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({kBatches, accum_depth});
    const RuntimeShape filter_shape({kOutputDepth, accum_depth});
    const RuntimeShape output_shape({kBatches, kOutputDepth});
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 61);
    const std::vector<int8_t> filter =
        MakeInt8Input(filter_shape.FlatSize(), 131);
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize());

    optimized_integer_ops::FullyConnectedPerChannel(
        params, output_multiplier, output_shift, input_shape, input.data(),
        filter_shape, filter.data(), bias_shape, bias.data(), output_shape,
        actual.data(), nullptr);
    reference_integer_ops::FullyConnectedPerChannel(
        params, output_multiplier, output_shift, input_shape, input.data(),
        filter_shape, filter.data(), bias_shape, bias.data(), output_shape,
        expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "accum_depth=" << accum_depth;
  }
}

TEST(RvvOpsTest, Int8ConvPerChannelMatchesReferenceAcrossVectorBoundaries) {
  const ConvParams params = MakeInt8ConvPerChannelParams();
  constexpr int kBatches = 1;
  constexpr int kInputHeight = 3;
  constexpr int kInputWidth = 4;
  constexpr int kFilterHeight = 2;
  constexpr int kFilterWidth = 2;
  constexpr int kOutputHeight = 2;
  constexpr int kOutputWidth = 3;
  constexpr int kOutputDepth = 6;
  const RuntimeShape bias_shape({kOutputDepth});
  const std::vector<int32_t> bias = MakeInt32Input(kOutputDepth, 211);
  const int32_t output_multiplier[kOutputDepth] = {
      1073741824, 1234567890, 1342177280, 987654321, 1503238554, 1191182336};
  const int32_t output_shift[kOutputDepth] = {-4, -3, -5, -2, -6, -3};

  for (int input_depth : Int8M1VectorLengths()) {
    if (input_depth == 0) {
      continue;
    }
    const RuntimeShape input_shape(
        {kBatches, kInputHeight, kInputWidth, input_depth});
    const RuntimeShape filter_shape(
        {kOutputDepth, kFilterHeight, kFilterWidth, input_depth});
    const RuntimeShape output_shape(
        {kBatches, kOutputHeight, kOutputWidth, kOutputDepth});
    const RuntimeShape im2col_shape(
        {1, kOutputHeight, kOutputWidth,
         kFilterHeight * kFilterWidth * input_depth});
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 29);
    const std::vector<int8_t> filter =
        MakeInt8Input(filter_shape.FlatSize(), 179);
    std::vector<int8_t> im2col(im2col_shape.FlatSize());
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize());

    optimized_integer_ops::ConvPerChannel(
        params, output_multiplier, output_shift, input_shape, input.data(),
        filter_shape, filter.data(), bias_shape, bias.data(), output_shape,
        actual.data(), im2col_shape, im2col.data(), nullptr);
    reference_integer_ops::ConvPerChannel(
        params, output_multiplier, output_shift, input_shape, input.data(),
        filter_shape, filter.data(), bias_shape, bias.data(), output_shape,
        expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "input_depth=" << input_depth;
  }
}

TEST(RvvOpsTest,
     Int8DepthwiseConvPerChannelMatchesReferenceAcrossVectorBoundaries) {
  const DepthwiseParams params = MakeInt8DepthwiseConvPerChannelParams();
  constexpr int kBatches = 1;
  constexpr int kInputHeight = 4;
  constexpr int kInputWidth = 5;
  constexpr int kFilterHeight = 2;
  constexpr int kFilterWidth = 3;
  constexpr int kOutputHeight = 5;
  constexpr int kOutputWidth = 3;
  CpuBackendContext cpu_backend_context;
  cpu_backend_context.SetMaxNumThreads(1);

  for (int input_depth : Int8M1VectorLengths()) {
    if (input_depth == 0) {
      continue;
    }
    const int output_depth = input_depth * params.depth_multiplier;
    const RuntimeShape input_shape(
        {kBatches, kInputHeight, kInputWidth, input_depth});
    const RuntimeShape filter_shape(
        {1, kFilterHeight, kFilterWidth, output_depth});
    const RuntimeShape bias_shape({output_depth});
    const RuntimeShape output_shape(
        {kBatches, kOutputHeight, kOutputWidth, output_depth});
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 53);
    const std::vector<int8_t> filter =
        MakeInt8Input(filter_shape.FlatSize(), 197);
    const std::vector<int32_t> bias = MakeInt32Input(output_depth, 307);
    std::vector<int32_t> output_multiplier(output_depth);
    std::vector<int32_t> output_shift(output_depth);
    for (int i = 0; i < output_depth; ++i) {
      output_multiplier[i] = 1073741824 + (i % 5) * 67108864;
      output_shift[i] = -2 - (i % 4);
    }
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize());

    optimized_integer_ops::DepthwiseConvPerChannel(
        params, output_multiplier.data(), output_shift.data(), input_shape,
        input.data(), filter_shape, filter.data(), bias_shape, bias.data(),
        output_shape, actual.data(), &cpu_backend_context);
    reference_integer_ops::DepthwiseConvPerChannel(
        params, output_multiplier.data(), output_shift.data(), input_shape,
        input.data(), filter_shape, filter.data(), bias_shape, bias.data(),
        output_shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "input_depth=" << input_depth;
  }
}

TEST(RvvOpsTest, AddElementwiseMatchesScalarReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.float_activation_min = -2.5f;
  params.float_activation_max = 3.5f;

  for (int size : Float32M4VectorLengths()) {
    const std::vector<float> input1 = MakeInput(size, -1.25f);
    const std::vector<float> input2 = MakeInput(size, 0.5f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::AddElementwise(size, params, input1.data(), input2.data(),
                                  actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = Clamp(input1[i] + input2[i], params.float_activation_min,
                          params.float_activation_max);
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest,
     AddScalarBroadcastMatchesScalarReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.float_activation_min = -1.0f;
  params.float_activation_max = 2.25f;
  constexpr float kBroadcastValue = 0.875f;

  for (int size : Float32M4VectorLengths()) {
    const std::vector<float> input = MakeInput(size, -0.25f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::AddScalarBroadcast(size, params, kBroadcastValue,
                                      input.data(), actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] =
          Clamp(kBroadcastValue + input[i], params.float_activation_min,
                params.float_activation_max);
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, BiasAndClampMatchesScalarReferenceAcrossVectorBoundaries) {
  constexpr float kClampMin = -1.5f;
  constexpr float kClampMax = 2.0f;
  constexpr int kBatchCount = 3;

  for (int bias_size : Float32M4VectorLengths()) {
    if (bias_size == 0) {
      continue;
    }

    const std::vector<float> bias = MakeInput(bias_size, 0.375f);
    std::vector<float> actual = MakeInput(bias_size * kBatchCount, -0.625f);
    std::vector<float> expected = actual;

    BiasAndClamp(kClampMin, kClampMax, bias_size, bias.data(),
                 static_cast<int>(actual.size()), actual.data());
    for (int batch = 0; batch < kBatchCount; ++batch) {
      for (int i = 0; i < bias_size; ++i) {
        const int index = batch * bias_size + i;
        expected[index] =
            Clamp(expected[index] + bias[i], kClampMin, kClampMax);
      }
    }

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "bias_size=" << bias_size;
  }
}

TEST(RvvOpsTest,
     FloatFullyConnectedGemmMatchesReferenceAcrossVectorBoundaries) {
  FullyConnectedParams params;
  params.float_activation_min = -3.75f;
  params.float_activation_max = 4.5f;
  constexpr int kBatches = 3;
  constexpr int kOutputDepth = 7;
  const RuntimeShape bias_shape({kOutputDepth});
  std::vector<float> bias = MakeInput(kOutputDepth, 0.25f);
  for (float& value : bias) {
    value *= 0.125f;
  }

  for (int accum_depth : Float32M4VectorLengths()) {
    if (accum_depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({kBatches, accum_depth});
    const RuntimeShape filter_shape({kOutputDepth, accum_depth});
    const RuntimeShape output_shape({kBatches, kOutputDepth});
    std::vector<float> input = MakeInput(input_shape.FlatSize(), -0.375f);
    std::vector<float> filter = MakeInput(filter_shape.FlatSize(), 0.625f);
    for (float& value : input) {
      value *= 0.125f;
    }
    for (float& value : filter) {
      value *= 0.125f;
    }
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    optimized_ops::FullyConnected(
        params, input_shape, input.data(), filter_shape, filter.data(),
        bias_shape, bias.data(), output_shape, actual.data(), nullptr);
    reference_ops::FullyConnected(params, input_shape, input.data(),
                                  filter_shape, filter.data(), bias_shape,
                                  bias.data(), output_shape, expected.data());

    SCOPED_TRACE(::testing::Message() << "accum_depth=" << accum_depth);
    ExpectFloatNearOrSpecial(actual, expected, 1e-4f);
  }
}

TEST(RvvOpsTest, FloatConv1x1GemmMatchesReferenceAcrossVectorBoundaries) {
  ConvParams params;
  params.padding_type = PaddingType::kNone;
  params.padding_values.width = 0;
  params.padding_values.height = 0;
  params.stride_width = 1;
  params.stride_height = 1;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  params.float_activation_min = -2.5f;
  params.float_activation_max = 3.25f;
  constexpr int kBatches = 1;
  constexpr int kInputHeight = 3;
  constexpr int kInputWidth = 4;
  constexpr int kOutputDepth = 6;
  const RuntimeShape bias_shape({kOutputDepth});
  std::vector<float> bias = MakeInput(kOutputDepth, -0.125f);
  for (float& value : bias) {
    value *= 0.125f;
  }

  for (int input_depth : Float32M4VectorLengths()) {
    if (input_depth == 0) {
      continue;
    }
    const RuntimeShape input_shape(
        {kBatches, kInputHeight, kInputWidth, input_depth});
    const RuntimeShape filter_shape({kOutputDepth, 1, 1, input_depth});
    const RuntimeShape output_shape(
        {kBatches, kInputHeight, kInputWidth, kOutputDepth});
    const RuntimeShape im2col_shape({0});
    std::vector<float> input = MakeInput(input_shape.FlatSize(), 0.375f);
    std::vector<float> filter = MakeInput(filter_shape.FlatSize(), -0.625f);
    for (float& value : input) {
      value *= 0.125f;
    }
    for (float& value : filter) {
      value *= 0.125f;
    }
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    optimized_ops::Conv(params, input_shape, input.data(), filter_shape,
                        filter.data(), bias_shape, bias.data(), output_shape,
                        actual.data(), im2col_shape, nullptr, nullptr);
    reference_ops::Conv(params, input_shape, input.data(), filter_shape,
                        filter.data(), bias_shape, bias.data(), output_shape,
                        expected.data(), im2col_shape, nullptr);

    SCOPED_TRACE(::testing::Message() << "input_depth=" << input_depth);
    ExpectFloatNearOrSpecial(actual, expected, 1e-4f);
  }
}

TEST(RvvOpsTest, FloatMaxPoolMatchesReferenceAcrossVectorBoundaries) {
  const PoolParams params = MakeFloatPoolParams();

  for (int depth : Float32M4VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<float> input = MakeInput(input_shape.FlatSize(), -0.25f);
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    optimized_ops::MaxPool(params, input_shape, input.data(), output_shape,
                           actual.data());
    reference_ops::MaxPool(params, input_shape, input.data(), output_shape,
                           expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, Uint8MaxPoolMatchesReferenceAcrossVectorBoundaries) {
  const PoolParams params = MakeUint8PoolParams();

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<uint8_t> input =
        MakeUint8Input(input_shape.FlatSize(), 43);
    std::vector<uint8_t> actual(output_shape.FlatSize());
    std::vector<uint8_t> expected(output_shape.FlatSize());

    optimized_ops::MaxPool(params, input_shape, input.data(), output_shape,
                           actual.data());
    reference_ops::MaxPool(params, input_shape, input.data(), output_shape,
                           expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, Int8MaxPoolMatchesReferenceAcrossVectorBoundaries) {
  const PoolParams params = MakeInt8PoolParams();

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 37);
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize());

    optimized_integer_ops::MaxPool(params, input_shape, input.data(),
                                   output_shape, actual.data());
    reference_integer_ops::MaxPool(params, input_shape, input.data(),
                                   output_shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, FloatAveragePoolMatchesReferenceAcrossVectorBoundaries) {
  const PoolParams params = MakeFloatPoolParams();

  for (int depth : Float32M4VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<float> input = MakeInput(input_shape.FlatSize(), 0.125f);
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    EXPECT_TRUE(optimized_ops::AveragePool(params, input_shape, input.data(),
                                           output_shape, actual.data()));
    EXPECT_TRUE(reference_ops::AveragePool(params, input_shape, input.data(),
                                           output_shape, expected.data()));

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "depth=" << depth;
  }
}

TEST(RvvOpsTest, Uint8AveragePoolMatchesReferenceAcrossVectorBoundaries) {
  const PoolParams params = MakeUint8PoolParams();

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<uint8_t> input =
        MakeUint8Input(input_shape.FlatSize(), 89);
    std::vector<uint8_t> actual(output_shape.FlatSize());
    std::vector<uint8_t> expected(output_shape.FlatSize());

    EXPECT_TRUE(optimized_ops::AveragePool(params, input_shape, input.data(),
                                           output_shape, actual.data()));
    EXPECT_TRUE(reference_ops::AveragePool(params, input_shape, input.data(),
                                           output_shape, expected.data()));

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, Int8AveragePoolMatchesReferenceAcrossVectorBoundaries) {
  const PoolParams params = MakeInt8PoolParams();

  for (int depth : Int8M1VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<int8_t> input =
        MakeInt8Input(input_shape.FlatSize(), 181);
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize());

    EXPECT_TRUE(optimized_integer_ops::AveragePool(
        params, input_shape, input.data(), output_shape, actual.data()));
    EXPECT_TRUE(reference_integer_ops::AveragePool(
        params, input_shape, input.data(), output_shape, expected.data()));

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, FloatL2PoolMatchesReferenceAcrossVectorBoundaries) {
  PoolParams params = MakeFloatPoolParams();
  params.float_activation_min = 0.25f;
  params.float_activation_max = 3.0f;

  for (int depth : Float32M4VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape input_shape({1, 4, 5, depth});
    const RuntimeShape output_shape({1, 4, 3, depth});
    const std::vector<float> input = MakeInput(input_shape.FlatSize(), -0.5f);
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    optimized_ops::L2Pool(params, input_shape, input.data(), output_shape,
                          actual.data());
    reference_ops::L2Pool(params, input_shape, input.data(), output_shape,
                          expected.data());

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-5f), expected))
        << "depth=" << depth;
  }
}

TEST(RvvOpsTest, FloatReduceMaxLastAxisMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kOuterSize = 3;
  const int axis[] = {1};

  for (int axis_size : Float32M4VectorLengths()) {
    if (axis_size == 0) {
      continue;
    }
    const int input_dims[] = {kOuterSize, axis_size};
    const int output_dims[] = {kOuterSize};
    int resolved_axis[2];
    int normalized_dims[2];
    const std::vector<float> input = MakeInput(kOuterSize * axis_size, -0.125f);
    std::vector<float> actual(kOuterSize);
    std::vector<float> expected(kOuterSize,
                                std::numeric_limits<float>::lowest());

    EXPECT_TRUE(optimized_ops::ReduceGeneric(
        input.data(), input_dims, 2, actual.data(), output_dims, 1, axis, 1,
        resolved_axis, normalized_dims, ops::builtin::reduce::kMax));
    for (int outer = 0; outer < kOuterSize; ++outer) {
      for (int i = 0; i < axis_size; ++i) {
        expected[outer] =
            std::max(expected[outer], input[outer * axis_size + i]);
      }
    }

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "axis_size=" << axis_size;
  }
}

TEST(RvvOpsTest, FloatReduceMinLastAxisMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kOuterSize = 3;
  const int axis[] = {1};

  for (int axis_size : Float32M4VectorLengths()) {
    if (axis_size == 0) {
      continue;
    }
    const int input_dims[] = {kOuterSize, axis_size};
    const int output_dims[] = {kOuterSize};
    int resolved_axis[2];
    int normalized_dims[2];
    const std::vector<float> input = MakeInput(kOuterSize * axis_size, 0.375f);
    std::vector<float> actual(kOuterSize);
    std::vector<float> expected(kOuterSize, std::numeric_limits<float>::max());

    EXPECT_TRUE(optimized_ops::ReduceGeneric(
        input.data(), input_dims, 2, actual.data(), output_dims, 1, axis, 1,
        resolved_axis, normalized_dims, ops::builtin::reduce::kMin));
    for (int outer = 0; outer < kOuterSize; ++outer) {
      for (int i = 0; i < axis_size; ++i) {
        expected[outer] =
            std::min(expected[outer], input[outer * axis_size + i]);
      }
    }

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "axis_size=" << axis_size;
  }
}

TEST(RvvOpsTest, FloatReduceSumLastAxisMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kOuterSize = 3;
  const int axis[] = {1};

  for (int axis_size : Float32M4VectorLengths()) {
    if (axis_size == 0) {
      continue;
    }
    const int input_dims[] = {kOuterSize, axis_size};
    const int output_dims[] = {kOuterSize};
    int resolved_axis[2];
    int normalized_dims[2];
    const std::vector<float> input = MakeInput(kOuterSize * axis_size, -0.5f);
    std::vector<float> actual(kOuterSize);
    std::vector<float> expected(kOuterSize, 0.0f);

    EXPECT_TRUE(optimized_ops::ReduceGeneric(
        input.data(), input_dims, 2, actual.data(), output_dims, 1, axis, 1,
        resolved_axis, normalized_dims, ops::builtin::reduce::kSum));
    for (int outer = 0; outer < kOuterSize; ++outer) {
      for (int i = 0; i < axis_size; ++i) {
        expected[outer] += input[outer * axis_size + i];
      }
    }

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-5f), expected))
        << "axis_size=" << axis_size;
  }
}

TEST(RvvOpsTest, FloatMeanLastAxisMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kOuterSize = 3;
  const int axis[] = {1};

  for (int axis_size : Float32M4VectorLengths()) {
    if (axis_size == 0) {
      continue;
    }
    const int input_dims[] = {kOuterSize, axis_size};
    const int output_dims[] = {kOuterSize};
    int resolved_axis[2];
    int normalized_dims[2];
    std::vector<float> temp_sum(kOuterSize);
    const std::vector<float> input = MakeInput(kOuterSize * axis_size, 0.25f);
    std::vector<float> actual(kOuterSize);
    std::vector<float> expected(kOuterSize, 0.0f);

    EXPECT_TRUE((optimized_ops::Mean<float, float>(
        input.data(), input_dims, 2, actual.data(), output_dims, 1, axis, 1,
        false, normalized_dims, resolved_axis, temp_sum.data())));
    for (int outer = 0; outer < kOuterSize; ++outer) {
      for (int i = 0; i < axis_size; ++i) {
        expected[outer] += input[outer * axis_size + i];
      }
      expected[outer] /= static_cast<float>(axis_size);
    }

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "axis_size=" << axis_size;
  }
}

TEST(RvvOpsTest, FloatArgMaxLastAxisMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kOuterSize = 3;
  const int axis[] = {1};

  for (int axis_size : Float32M4VectorLengths()) {
    if (axis_size == 0) {
      continue;
    }
    const RuntimeShape input_shape({kOuterSize, axis_size});
    const RuntimeShape output_shape({kOuterSize});
    const std::vector<float> input = MakeInput(kOuterSize * axis_size, -0.75f);
    std::vector<int32_t> actual(kOuterSize);
    std::vector<int32_t> expected(kOuterSize);

    optimized_ops::ArgMax(input_shape, input.data(), axis, output_shape,
                          actual.data());
    reference_ops::ArgMax(input_shape, input.data(), axis, output_shape,
                          expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "axis_size=" << axis_size;
  }
}

TEST(RvvOpsTest, FloatArgMinLastAxisMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kOuterSize = 3;
  const int axis[] = {1};

  for (int axis_size : Float32M4VectorLengths()) {
    if (axis_size == 0) {
      continue;
    }
    const RuntimeShape input_shape({kOuterSize, axis_size});
    const RuntimeShape output_shape({kOuterSize});
    const std::vector<float> input = MakeInput(kOuterSize * axis_size, 0.75f);
    std::vector<int32_t> actual(kOuterSize);
    std::vector<int32_t> expected(kOuterSize);

    optimized_ops::ArgMinMax(input_shape, input.data(), axis, output_shape,
                             actual.data(), /*is_arg_max=*/false);
    reference_ops::ArgMinMax(input_shape, input.data(), axis, output_shape,
                             expected.data(), /*is_arg_max=*/false);

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "axis_size=" << axis_size;
  }
}

TEST(RvvOpsTest, FloatSubMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.float_activation_min = -3.25f;
  params.float_activation_max = 2.0f;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input1 = MakeInput(size, 1.375f);
    const std::vector<float> input2 = MakeInput(size, -0.625f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::SubWithActivation<float>(params, shape, input1.data(), shape,
                                            input2.data(), shape,
                                            actual.data());
    reference_ops::SubWithActivation(params, shape, input1.data(), shape,
                                     input2.data(), shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, FloatReluMatchesReferenceAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeInput(size, -0.125f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Relu(shape, input.data(), shape, actual.data());
    reference_ops::Relu(shape, input.data(), shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, FloatLeakyReluMatchesReferenceAcrossVectorBoundaries) {
  const LeakyReluParams params = MakeFloatLeakyReluParams();

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeInput(size, -0.5f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::LeakyRelu(params, shape, input.data(), shape, actual.data());
    reference_ops::LeakyRelu(params, shape, input.data(), shape,
                             expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest,
     FloatPReluScalarBroadcastMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  constexpr float kAlpha = 0.375f;

  for (int size : Float32M4VectorLengths()) {
    const std::vector<float> input = MakeInput(size, -0.625f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::PReluScalarBroadcast(size, params, kAlpha, input.data(),
                                        actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = input[i] >= 0.0f ? input[i] : input[i] * kAlpha;
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, FloatPReluElementWiseMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;

  for (int size : Float32M4VectorLengths()) {
    const std::vector<float> input = MakeInput(size, -0.75f);
    std::vector<float> alpha(size);
    for (int i = 0; i < size; ++i) {
      alpha[i] = static_cast<float>((i % 7) + 1) * 0.125f;
    }
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::PReluElementWise(size, params, alpha.data(), input.data(),
                                    actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = input[i] >= 0.0f ? input[i] : input[i] * alpha[i];
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, FloatHardSwishMatchesReferenceAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeInput(size, -0.875f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::HardSwish(shape, input.data(), shape, actual.data());
    reference_ops::HardSwish(shape, input.data(), shape, expected.data());

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "size=" << size;
  }
}

TEST(RvvOpsTest, FloatLogisticMatchesScalarAcrossVectorBoundaries) {
  LogisticParams params;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeActivationInput(size);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Logistic(params, shape, input.data(), shape, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = ReferenceLogisticFloat(input[i]);
    }

    SCOPED_TRACE(size);
    ExpectFloatNearOrSpecial(actual, expected);
  }
}

TEST(RvvOpsTest, FloatTanhMatchesScalarAcrossVectorBoundaries) {
  TanhParams params;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeActivationInput(size);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Tanh(params, shape, input.data(), shape, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = std::tanh(input[i]);
    }

    SCOPED_TRACE(size);
    ExpectFloatNearOrSpecial(actual, expected);
  }
}

TEST(RvvOpsTest, FloatEluMatchesScalarAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeActivationInput(size);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Elu(shape, input.data(), shape, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = input[i] < 0.0f ? std::expm1(input[i]) : input[i];
    }

    SCOPED_TRACE(size);
    ExpectFloatNearOrSpecial(actual, expected);
  }
}

TEST(RvvOpsTest, FloatGeluMatchesScalarAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeActivationInput(size);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    reference_ops::Gelu(shape, input.data(), /*approximate=*/false, shape,
                        actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = reference_ops::GeluTransform(input[i]);
    }

    SCOPED_TRACE(size);
    ExpectFloatNearOrSpecial(actual, expected, 1e-5f);
  }
}

TEST(RvvOpsTest, FloatGeluApproximateMatchesScalarAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeActivationInput(size);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    reference_ops::Gelu(shape, input.data(), /*approximate=*/true, shape,
                        actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = reference_ops::GeluTransformApproximate(input[i]);
    }

    SCOPED_TRACE(size);
    ExpectFloatNearOrSpecial(actual, expected, 1e-5f);
  }
}

TEST(RvvOpsTest, FloatSwishMatchesScalarAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input = MakeActivationInput(size);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Swish(shape, input.data(), shape, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = input[i] * ReferenceLogisticFloat(input[i]);
    }

    SCOPED_TRACE(size);
    ExpectFloatNearOrSpecial(actual, expected);
  }
}

TEST(RvvOpsTest, FloatSoftmaxMatchesReferenceAcrossVectorBoundaries) {
  SoftmaxParams params;
  params.beta = 0.75f;
  constexpr int kBatchSize = 3;

  for (int depth : Float32M4VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    const RuntimeShape shape({kBatchSize, depth});
    const std::vector<float> input = MakeInput(kBatchSize * depth, -0.375f);
    std::vector<float> actual(kBatchSize * depth);
    std::vector<float> expected(kBatchSize * depth);

    optimized_ops::Softmax(params, shape, input.data(), shape, actual.data());
    reference_ops::Softmax(params, shape, input.data(), shape, expected.data());

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "depth=" << depth;
  }
}

TEST(RvvOpsTest, FloatDivElementwiseMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.float_activation_min = -4.0f;
  params.float_activation_max = 5.0f;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input1 = MakeInput(size, 1.25f);
    std::vector<float> input2 = MakeInput(size, 2.5f);
    for (float& value : input2) {
      if (value == 0.0f) {
        value = 0.5f;
      }
    }
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Div(params, shape, input1.data(), shape, input2.data(),
                       shape, actual.data());
    reference_ops::Div(params, shape, input1.data(), shape, input2.data(),
                       shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, FloatMulElementwiseMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.float_activation_min = -3.0f;
  params.float_activation_max = 4.0f;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<float> input1 = MakeInput(size, -0.75f);
    const std::vector<float> input2 = MakeInput(size, 0.625f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::Mul(params, shape, input1.data(), shape, input2.data(),
                       shape, actual.data());
    reference_ops::Mul(params, shape, input1.data(), shape, input2.data(),
                       shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest,
     FloatMulSimpleBroadcastMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.float_activation_min = -2.75f;
  params.float_activation_max = 3.25f;
  constexpr float kBroadcastValue = -1.375f;

  for (int size : Float32M4VectorLengths()) {
    const std::vector<float> input = MakeInput(size, 0.375f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    optimized_ops::MulSimpleBroadcast(size, params, kBroadcastValue,
                                      input.data(), actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] =
          Clamp(kBroadcastValue * input[i], params.float_activation_min,
                params.float_activation_max);
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int32MulElementwiseMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.quantized_activation_min = -250000;
  params.quantized_activation_max = 225000;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int32_t> input1 = MakeInt32Input(size, 101);
    const std::vector<int32_t> input2 = MakeInt32Input(size, 1703);
    std::vector<int32_t> actual(size);
    std::vector<int32_t> expected(size);

    optimized_ops::Mul(params, shape, input1.data(), shape, input2.data(),
                       shape, actual.data());
    reference_ops::Mul(params, shape, input1.data(), shape, input2.data(),
                       shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int32DivElementwiseMatchesReferenceAcrossVectorBoundaries) {
  ArithmeticParams params;
  params.quantized_activation_min = -250;
  params.quantized_activation_max = 225;

  for (int size : Float32M4VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int32_t> input1 = MakeInt32Input(size, 1901);
    const std::vector<int32_t> input2 = MakeNonZeroInt32Input(size, 73);
    std::vector<int32_t> actual(size);
    std::vector<int32_t> expected(size);

    optimized_ops::Div(params, shape, input1.data(), shape, input2.data(),
                       shape, actual.data());
    reference_ops::Div(params, shape, input1.data(), shape, input2.data(),
                       shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int8AddElementwiseMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeInt8Params();

  for (int size : Int8M1VectorLengths()) {
    const std::vector<int8_t> input1 = MakeInt8Input(size, 19);
    const std::vector<int8_t> input2 = MakeInt8Input(size, 71);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_integer_ops::AddElementwiseInt8(size, params, input1.data(),
                                              input2.data(), actual.data());
    reference_integer_ops::AddElementwise(size, params, input1.data(),
                                          input2.data(), expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int8AddScalarBroadcastMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeInt8Params();
  constexpr int8_t kBroadcastValue = -37;

  for (int size : Int8M1VectorLengths()) {
    const std::vector<int8_t> input = MakeInt8Input(size, 103);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_integer_ops::AddScalarBroadcast(size, params, kBroadcastValue,
                                              input.data(), actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] =
          reference_integer_ops::AddFunc(kBroadcastValue, input[i], params);
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int8SubElementwiseMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeInt8Params();

  for (int size : Int8M1VectorLengths()) {
    const std::vector<int8_t> input1 = MakeInt8Input(size, 41);
    const std::vector<int8_t> input2 = MakeInt8Input(size, 113);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_integer_ops::SubElementwiseInt8(size, params, input1.data(),
                                              input2.data(), actual.data());
    reference_ops::SubElementwise(size, params, input1.data(), input2.data(),
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int8MulElementwiseMatchesReferenceAcrossVectorBoundaries) {
  for (const ArithmeticParams params :
       {MakeInt8MulParams(-2), MakeInt8MulParams(1)}) {
    for (int size : Int8M1VectorLengths()) {
      const std::vector<int8_t> input1 = MakeInt8Input(size, 59);
      const std::vector<int8_t> input2 = MakeInt8Input(size, 127);
      std::vector<int8_t> actual(size);
      std::vector<int8_t> expected(size);

      optimized_integer_ops::MulElementwise(size, params, input1.data(),
                                            input2.data(), actual.data());
      reference_integer_ops::MulElementwise(size, params, input1.data(),
                                            input2.data(), expected.data());

      EXPECT_THAT(actual, ElementsAreArray(expected))
          << "size=" << size << " output_shift=" << params.output_shift;
    }
  }
}

TEST(RvvOpsTest, Int8MulSimpleBroadcastMatchesReferenceAcrossVectorBoundaries) {
  constexpr int8_t kBroadcastValue = -29;

  for (const ArithmeticParams params :
       {MakeInt8MulParams(-2), MakeInt8MulParams(1)}) {
    for (int size : Int8M1VectorLengths()) {
      const std::vector<int8_t> input = MakeInt8Input(size, 211);
      const std::vector<int8_t> broadcast_values(size, kBroadcastValue);
      std::vector<int8_t> actual(size);
      std::vector<int8_t> expected(size);

      optimized_integer_ops::MulSimpleBroadcast(size, params, kBroadcastValue,
                                                input.data(), actual.data());
      reference_integer_ops::MulElementwise(
          size, params, broadcast_values.data(), input.data(), expected.data());

      EXPECT_THAT(actual, ElementsAreArray(expected))
          << "size=" << size << " output_shift=" << params.output_shift;
    }
  }
}

TEST(RvvOpsTest, Int8ReluXMatchesReferenceAcrossVectorBoundaries) {
  const ReluParams params = MakeInt8ReluParams();

  for (int size : Int8M1VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int8_t> input = MakeInt8Input(size, 97);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    optimized_ops::ReluX(params, shape, input.data(), shape, actual.data());
    reference_ops::ReluX(params, shape, input.data(), shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Uint8AddElementwiseMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeUint8Params();

  for (int size : Int8M1VectorLengths()) {
    const std::vector<uint8_t> input1 = MakeUint8Input(size, 17);
    const std::vector<uint8_t> input2 = MakeUint8Input(size, 83);
    std::vector<uint8_t> actual(size);
    std::vector<uint8_t> expected(size);

    optimized_ops::AddElementwise(size, params, input1.data(), input2.data(),
                                  actual.data());
    reference_ops::AddElementwise(size, params, input1.data(), input2.data(),
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Uint8SubElementwiseMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeUint8Params();

  for (int size : Int8M1VectorLengths()) {
    const std::vector<uint8_t> input1 = MakeUint8Input(size, 47);
    const std::vector<uint8_t> input2 = MakeUint8Input(size, 139);
    std::vector<uint8_t> actual(size);
    std::vector<uint8_t> expected(size);

    optimized_ops::SubElementwise(size, params, input1.data(), input2.data(),
                                  actual.data());
    reference_ops::SubElementwise(size, params, input1.data(), input2.data(),
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Uint8MulElementwiseMatchesReferenceAcrossVectorBoundaries) {
  for (const ArithmeticParams params :
       {MakeUint8MulParams(-2), MakeUint8MulParams(1)}) {
    for (int size : Int8M1VectorLengths()) {
      const std::vector<uint8_t> input1 = MakeUint8Input(size, 31);
      const std::vector<uint8_t> input2 = MakeUint8Input(size, 149);
      std::vector<uint8_t> actual(size);
      std::vector<uint8_t> expected(size);

      optimized_ops::MulElementwise(size, params, input1.data(), input2.data(),
                                    actual.data());
      reference_ops::MulElementwise(size, params, input1.data(), input2.data(),
                                    expected.data());

      EXPECT_THAT(actual, ElementsAreArray(expected))
          << "size=" << size << " output_shift=" << params.output_shift;
    }
  }
}

TEST(RvvOpsTest,
     Uint8AddScalarBroadcastMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeUint8Params();
  constexpr uint8_t kBroadcastValue = 173;

  for (int size : Int8M1VectorLengths()) {
    const std::vector<uint8_t> input = MakeUint8Input(size, 29);
    std::vector<uint8_t> actual(size);
    std::vector<uint8_t> expected(size);

    optimized_ops::AddScalarBroadcast(size, params, kBroadcastValue,
                                      input.data(), actual.data());
    reference_ops::AddScalarBroadcast(size, params, kBroadcastValue,
                                      input.data(), expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest,
     Uint8MulSimpleBroadcastMatchesReferenceAcrossVectorBoundaries) {
  constexpr uint8_t kBroadcastValue = 173;

  for (const ArithmeticParams params :
       {MakeUint8MulParams(-2), MakeUint8MulParams(1)}) {
    for (int size : Int8M1VectorLengths()) {
      const std::vector<uint8_t> input = MakeUint8Input(size, 67);
      const std::vector<uint8_t> broadcast_values(size, kBroadcastValue);
      std::vector<uint8_t> actual(size);
      std::vector<uint8_t> expected(size);

      optimized_ops::MulSimpleBroadcast(size, params, kBroadcastValue,
                                        input.data(), actual.data());
      reference_ops::MulElementwise(size, params, broadcast_values.data(),
                                    input.data(), expected.data());

      EXPECT_THAT(actual, ElementsAreArray(expected))
          << "size=" << size << " output_shift=" << params.output_shift;
    }
  }
}

TEST(RvvOpsTest, Uint8ReluXMatchesReferenceAcrossVectorBoundaries) {
  const ReluParams params = MakeUint8ReluParams();

  for (int size : Int8M1VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<uint8_t> input = MakeUint8Input(size, 197);
    std::vector<uint8_t> actual(size);
    std::vector<uint8_t> expected(size);

    optimized_ops::ReluX(params, shape, input.data(), shape, actual.data());
    reference_ops::ReluX(params, shape, input.data(), shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int16AddElementwiseMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeInt16Params();

  for (int size : Int16M2VectorLengths()) {
    const std::vector<int16_t> input1 = MakeInt16Input(size, 701);
    const std::vector<int16_t> input2 = MakeInt16Input(size, 1703);
    std::vector<int16_t> actual(size);
    std::vector<int16_t> expected(size);

    optimized_integer_ops::AddElementwiseInt16(size, params, input1.data(),
                                               input2.data(), actual.data());
    reference_ops::AddElementwise(size, params, input1.data(), input2.data(),
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int16LeakyReluMatchesReferenceAcrossVectorBoundaries) {
  const LeakyReluParams params = MakeInt16LeakyReluParams();

  for (int size : Int16M2VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int16_t> input = MakeInt16Input(size, 3907);
    std::vector<int16_t> actual(size);
    std::vector<int16_t> expected(size);

    optimized_integer_ops::QuantizeLeakyRelu(params, shape, input.data(), shape,
                                             actual.data());
    reference_ops::QuantizeLeakyRelu(params, shape, input.data(), shape,
                                     expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int16ReluXMatchesReferenceAcrossVectorBoundaries) {
  const ReluParams params = MakeInt16ReluParams();

  for (int size : Int16M2VectorLengths()) {
    const RuntimeShape shape({size});
    const std::vector<int16_t> input = MakeInt16Input(size, 1103);
    std::vector<int16_t> actual(size);
    std::vector<int16_t> expected(size);

    optimized_ops::ReluX(params, shape, input.data(), shape, actual.data());
    reference_ops::ReluX(params, shape, input.data(), shape, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, Int16SubElementwiseMatchesReferenceAcrossVectorBoundaries) {
  const ArithmeticParams params = MakeInt16Params();

  for (int size : Int16M2VectorLengths()) {
    const std::vector<int16_t> input1 = MakeInt16Input(size, 301);
    const std::vector<int16_t> input2 = MakeInt16Input(size, 2701);
    std::vector<int16_t> actual(size);
    std::vector<int16_t> expected(size);

    optimized_integer_ops::SubElementwiseInt16(size, params, input1.data(),
                                               input2.data(), actual.data());
    reference_ops::SubElementwise(size, params, input1.data(), input2.data(),
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

void ReferenceLstmCwiseMulInt16(const int16_t* input1, const int16_t* input2,
                                int n_batch, int n_input, int shift,
                                int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int32_t value = static_cast<int32_t>(input1[index]) * input2[index];
      output[index] =
          static_cast<int16_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void ReferenceLstmCwiseMulInt16ToInt8(const int16_t* input1,
                                      const int16_t* input2, int32_t multiplier,
                                      int32_t shift, int32_t n_batch,
                                      int32_t n_input, int32_t output_zp,
                                      int8_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t value = static_cast<int32_t>(input1[index]) * input2[index];
      value = MultiplyByQuantizedMultiplier(value, multiplier, shift);
      value += output_zp;
      value = std::min(std::max(value, -128), 127);
      output[index] = static_cast<int8_t>(value);
    }
  }
}

void ReferenceLstmCwiseAddInt16(const int16_t* input1, const int16_t* input2,
                                int n_batch, int n_input, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t sum = input1[index] + input2[index];
      sum = std::min(std::max(sum, static_cast<int32_t>(-32768)),
                     static_cast<int32_t>(32767));
      output[index] = static_cast<int16_t>(sum);
    }
  }
}

void ReferenceLstmVectorBatchVectorCwiseProductAccumulate(
    const int16_t* vector, int v_size, const int16_t* batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t* result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < v_size; ++i) {
      const int index = batch * v_size + i;
      int32_t product = vector[i] * batch_vector[index];
      product = MultiplyByQuantizedMultiplier(product, multiplier, shift);
      int32_t output = product + result[index];
      output = std::min(std::max(output, static_cast<int32_t>(-32768)),
                        static_cast<int32_t>(32767));
      result[index] = static_cast<int16_t>(output);
    }
  }
}

void ReferenceRnnMatrixBatchVectorMultiplyAccumulateFloat(
    const float* matrix, int m_rows, int m_cols, const float* vector,
    int n_batch, float* result) {
  float* result_in_batch = result;
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < m_rows; ++row) {
      float dot_product = 0.0f;
      for (int col = 0; col < m_cols; ++col) {
        dot_product +=
            matrix[row * m_cols + col] * vector[batch * m_cols + col];
      }
      *result_in_batch += dot_product;
      ++result_in_batch;
    }
  }
}

void ReferenceRnnMatrixBatchVectorMultiplyAccumulateInt8(
    const int8_t* matrix, int m_rows, int m_cols, const int8_t* vectors,
    const float* scaling_factors, int n_batch, const float* per_channel_scale,
    const int32_t* input_offset, const int32_t* row_sums, float* result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < m_rows; ++row) {
      int32_t dot_product = 0;
      for (int col = 0; col < m_cols; ++col) {
        dot_product +=
            matrix[row * m_cols + col] * vectors[batch * m_cols + col];
      }
      float scale = scaling_factors[batch];
      if (per_channel_scale != nullptr) {
        scale *= per_channel_scale[row];
      }
      if (input_offset != nullptr) {
        dot_product -= row_sums[row] * input_offset[batch];
      }
      *result += dot_product * scale;
      ++result;
    }
  }
}

std::vector<int32_t> ReferenceRnnRowSums(const int8_t* matrix, int m_rows,
                                         int m_cols) {
  std::vector<int32_t> row_sums(m_rows);
  for (int row = 0; row < m_rows; ++row) {
    int32_t sum = 0;
    for (int col = 0; col < m_cols; ++col) {
      sum += matrix[row * m_cols + col];
    }
    row_sums[row] = sum;
  }
  return row_sums;
}

TEST(
    RvvOpsTest,
    RnnFloatMatrixBatchVectorMultiplyAccumulateMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kRows = 7;
  constexpr int kBatch = 3;

  for (int cols : Float32M4VectorLengths()) {
    const std::vector<float> matrix = MakeInput(kRows * cols, 0.125f);
    const std::vector<float> vector = MakeInput(kBatch * cols, -0.375f);
    std::vector<float> actual = MakeInput(kRows * kBatch, 0.25f);
    std::vector<float> expected = actual;

    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        matrix.data(), kRows, cols, vector.data(), kBatch, actual.data());
    ReferenceRnnMatrixBatchVectorMultiplyAccumulateFloat(
        matrix.data(), kRows, cols, vector.data(), kBatch, expected.data());

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-4f), expected))
        << "cols=" << cols;
  }
}

TEST(
    RvvOpsTest,
    RnnHybridInt8MatrixBatchVectorMultiplyAccumulateMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kRows = 6;
  constexpr int kBatch = 3;
  const std::vector<float> scaling_factors = {0.03125f, 0.046875f, 0.0625f};

  for (int cols : Int8M1VectorLengths()) {
    const std::vector<int8_t> matrix = MakeInt8Input(kRows * cols, 19);
    const std::vector<int8_t> vectors = MakeInt8Input(kBatch * cols, 113);
    std::vector<float> actual = MakeInput(kRows * kBatch, -0.125f);
    std::vector<float> expected = actual;

    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        matrix.data(), kRows, cols, vectors.data(), scaling_factors.data(),
        kBatch, actual.data());
    ReferenceRnnMatrixBatchVectorMultiplyAccumulateInt8(
        matrix.data(), kRows, cols, vectors.data(), scaling_factors.data(),
        kBatch, /*per_channel_scale=*/nullptr, /*input_offset=*/nullptr,
        /*row_sums=*/nullptr, expected.data());

    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "cols=" << cols;
  }
}

TEST(
    RvvOpsTest,
    RnnHybridInt8MatrixBatchVectorMultiplyAccumulateWithOffsetsMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kRows = 5;
  constexpr int kBatch = 3;
  const std::vector<float> scaling_factors = {0.03125f, 0.046875f, 0.0625f};
  const std::vector<float> per_channel_scale = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f};
  const std::vector<int32_t> input_offsets = {-3, 5, -7};

  for (int cols : Int8M1VectorLengths()) {
    const std::vector<int8_t> matrix = MakeInt8Input(kRows * cols, 37);
    const std::vector<int8_t> vectors = MakeInt8Input(kBatch * cols, 211);
    std::vector<float> actual = MakeInput(kRows * kBatch, 0.375f);
    std::vector<float> expected = actual;
    std::vector<int32_t> actual_row_sums(kRows);
    std::vector<int32_t> scratch(kRows * kBatch);
    bool compute_row_sums = true;
    const std::vector<int32_t> expected_row_sums =
        ReferenceRnnRowSums(matrix.data(), kRows, cols);

    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        matrix.data(), kRows, cols, vectors.data(), scaling_factors.data(),
        kBatch, actual.data(), per_channel_scale.data(), input_offsets.data(),
        scratch.data(), actual_row_sums.data(), &compute_row_sums,
        /*context=*/nullptr);
    ReferenceRnnMatrixBatchVectorMultiplyAccumulateInt8(
        matrix.data(), kRows, cols, vectors.data(), scaling_factors.data(),
        kBatch, per_channel_scale.data(), input_offsets.data(),
        expected_row_sums.data(), expected.data());

    EXPECT_THAT(actual_row_sums, ElementsAreArray(expected_row_sums))
        << "cols=" << cols;
    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "cols=" << cols;
    EXPECT_FALSE(compute_row_sums) << "cols=" << cols;
  }
}

TEST(RvvOpsTest, RnnFloatActivationsMatchReferenceAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    const std::vector<float> input = MakeInput(size, -0.25f);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    tensor_utils::ApplyReluToVector(input.data(), size, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = std::max(0.0f, input[i]);
    }
    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "Relu size=" << size;

    tensor_utils::ApplyRelu1ToVector(input.data(), size, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = std::max(-1.0f, std::min(input[i], 1.0f));
    }
    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "Relu1 size=" << size;

    tensor_utils::ApplyRelu6ToVector(input.data(), size, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = std::max(0.0f, std::min(input[i], 6.0f));
    }
    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "Relu6 size=" << size;

    tensor_utils::ApplySignbitToVector(input.data(), size, actual.data());
    for (int i = 0; i < size; ++i) {
      expected[i] = std::signbit(input[i]);
    }
    EXPECT_THAT(actual, Pointwise(FloatNear(1e-6f), expected))
        << "Signbit size=" << size;

    if (size >= 2) {
      std::vector<float> zero_input(size, 0.0f);
      zero_input[0] = -0.0f;
      tensor_utils::ApplySignbitToVector(zero_input.data(), size,
                                         actual.data());
      EXPECT_EQ(actual[0], 1.0f) << "negative zero size=" << size;
      EXPECT_EQ(actual[1], 0.0f) << "positive zero size=" << size;
    }
  }
}

void ReferenceSvdfFloat(
    const TfLiteSVDFParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_feature_shape,
    const float* weights_feature_data, const RuntimeShape& weights_time_shape,
    const float* weights_time_data, const float* bias_data, float* scratch_data,
    float* state_data, float* output_data) {
  const int rank = params.rank;
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.Dims(1);
  const int num_filters = weights_feature_shape.Dims(0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time_shape.Dims(1);

  std::copy(state_data + 1, state_data + batch_size * memory_size * num_filters,
            state_data);
  std::fill_n(scratch_data, batch_size * num_filters, 0.0f);

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int row = 0; row < num_filters; ++row) {
      float dot_product = 0.0f;
      for (int col = 0; col < input_size; ++col) {
        dot_product += weights_feature_data[row * input_size + col] *
                       input_data[batch * input_size + col];
      }
      scratch_data[batch * num_filters + row] += dot_product;
    }
  }

  for (int i = 0; i < batch_size * num_filters; ++i) {
    state_data[i * memory_size + memory_size - 1] = scratch_data[i];
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int filter = 0; filter < num_filters; ++filter) {
      float dot_product = 0.0f;
      for (int memory = 0; memory < memory_size; ++memory) {
        dot_product += weights_time_data[filter * memory_size + memory] *
                       state_data[batch * memory_size * num_filters +
                                  filter * memory_size + memory];
      }
      scratch_data[batch * num_filters + filter] = dot_product;
    }
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int unit = 0; unit < num_units; ++unit) {
      float output = 0.0f;
      for (int r = 0; r < rank; ++r) {
        output += scratch_data[batch * num_filters + unit * rank + r];
      }
      if (bias_data != nullptr) {
        output += bias_data[unit];
      }
      if (params.activation == kTfLiteActRelu) {
        output = std::max(0.0f, output);
      }
      output_data[batch * num_units + unit] = output;
    }
  }
}

void ReferenceSvdfInteger(
    const TfLiteSVDFParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& weights_feature_shape,
    const int8_t* weights_feature_data, const RuntimeShape& weights_time_shape,
    const int16_t* weights_time_data, const int32_t* bias_data,
    int16_t* state_data, int8_t* output_data, int32_t* scratch_data,
    int32_t* output_temp_data, int32_t scale_1_a, int scale_1_b,
    int32_t scale_2_a, int scale_2_b, int32_t input_zp, int32_t output_zp) {
  const int rank = params.rank;
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.Dims(1);
  const int num_filters = weights_feature_shape.Dims(0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time_shape.Dims(1);

  std::copy(state_data + 1, state_data + batch_size * memory_size * num_filters,
            state_data);

  int16_t* result_in_batch = state_data + (memory_size - 1);
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int row = 0; row < num_filters; ++row) {
      int32_t dot_product = 0;
      for (int col = 0; col < input_size; ++col) {
        dot_product += weights_feature_data[row * input_size + col] *
                       (input_data[batch * input_size + col] - input_zp);
      }
      dot_product =
          MultiplyByQuantizedMultiplier(dot_product, scale_1_a, scale_1_b);
      dot_product = std::min(std::max(dot_product, -32768), 32767);
      *result_in_batch = static_cast<int16_t>(dot_product);
      result_in_batch += memory_size;
    }
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int filter = 0; filter < num_filters; ++filter) {
      int32_t dot_product = 0;
      for (int memory = 0; memory < memory_size; ++memory) {
        dot_product += weights_time_data[filter * memory_size + memory] *
                       state_data[batch * memory_size * num_filters +
                                  filter * memory_size + memory];
      }
      scratch_data[batch * num_filters + filter] = dot_product;
    }
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int unit = 0; unit < num_units; ++unit) {
      int32_t output = 0;
      for (int r = 0; r < rank; ++r) {
        output += scratch_data[batch * num_filters + unit * rank + r];
      }
      if (bias_data != nullptr) {
        output += bias_data[unit];
      }
      output = MultiplyByQuantizedMultiplier(output, scale_2_a, scale_2_b);
      output += output_zp;
      output = std::min(std::max(output, -128), 127);
      output_temp_data[batch * num_units + unit] = output;
      output_data[batch * num_units + unit] = static_cast<int8_t>(output);
    }
  }
}

std::vector<int8_t> MakeSmallInt8Input(int size, int offset) {
  std::vector<int8_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<int8_t>(((i * 7 + offset) % 15) - 7);
  }
  return values;
}

std::vector<int16_t> MakeSmallInt16Input(int size, int offset) {
  std::vector<int16_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<int16_t>(((i * 17 + offset) % 129) - 64);
  }
  return values;
}

TEST(RvvOpsTest, SvdfFloatMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kBatch = 2;
  constexpr int kRank = 2;
  constexpr int kUnits = 3;
  constexpr int kFilters = kRank * kUnits;

  TfLiteSVDFParams params = {};
  params.rank = kRank;
  params.activation = kTfLiteActRelu;

  for (int size : Float32M4VectorLengths()) {
    if (size == 0) {
      continue;
    }
    const int input_size = size;
    const int memory_size = size;
    const RuntimeShape input_shape({kBatch, input_size});
    const RuntimeShape weights_feature_shape({kFilters, input_size});
    const RuntimeShape weights_time_shape({kFilters, memory_size});
    const RuntimeShape bias_shape({kUnits});
    const RuntimeShape output_shape({kBatch, kUnits});

    const std::vector<float> input = MakeInput(kBatch * input_size, 0.125f);
    const std::vector<float> weights_feature =
        MakeInput(kFilters * input_size, -0.25f);
    const std::vector<float> weights_time =
        MakeInput(kFilters * memory_size, 0.375f);
    const std::vector<float> bias = MakeInput(kUnits, -0.125f);
    std::vector<float> actual_state =
        MakeInput(kBatch * memory_size * kFilters, 0.25f);
    std::vector<float> expected_state = actual_state;
    std::vector<float> actual_scratch(kBatch * kFilters);
    std::vector<float> expected_scratch(kBatch * kFilters);
    std::vector<float> actual_output(kBatch * kUnits);
    std::vector<float> expected_output(kBatch * kUnits);

    reference_ops::EvalFloatSVDF(
        &params, input_shape, input.data(), weights_feature_shape,
        weights_feature.data(), weights_time_shape, weights_time.data(),
        bias_shape, bias.data(), actual_scratch.data(), actual_state.data(),
        output_shape, actual_output.data());
    ReferenceSvdfFloat(params, input_shape, input.data(), weights_feature_shape,
                       weights_feature.data(), weights_time_shape,
                       weights_time.data(), bias.data(),
                       expected_scratch.data(), expected_state.data(),
                       expected_output.data());

    EXPECT_THAT(actual_state, Pointwise(FloatNear(1e-4f), expected_state))
        << "size=" << size;
    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-4f), expected_output))
        << "size=" << size;
  }
}

TEST(RvvOpsTest, SvdfIntegerMatchesReferenceAcrossVectorBoundaries) {
  constexpr int kBatch = 2;
  constexpr int kRank = 2;
  constexpr int kUnits = 3;
  constexpr int kFilters = kRank * kUnits;
  constexpr int32_t kScale1A = 1073741824;
  constexpr int kScale1B = -3;
  constexpr int32_t kScale2A = 1073741824;
  constexpr int kScale2B = -5;
  constexpr int32_t kInputZeroPoint = -2;
  constexpr int32_t kOutputZeroPoint = 3;

  TfLiteSVDFParams params = {};
  params.rank = kRank;
  params.activation = kTfLiteActRelu;

  for (int size : Int8M1VectorLengths()) {
    if (size == 0) {
      continue;
    }
    const int input_size = size;
    const int memory_size = size;
    const RuntimeShape input_shape({kBatch, input_size});
    const RuntimeShape weights_feature_shape({kFilters, input_size});
    const RuntimeShape weights_time_shape({kFilters, memory_size});
    const RuntimeShape bias_shape({kUnits});
    const RuntimeShape output_shape({kBatch, kUnits});

    const std::vector<int8_t> input =
        MakeSmallInt8Input(kBatch * input_size, 3);
    const std::vector<int8_t> weights_feature =
        MakeSmallInt8Input(kFilters * input_size, 11);
    const std::vector<int16_t> weights_time =
        MakeSmallInt16Input(kFilters * memory_size, 29);
    const std::vector<int32_t> bias = {17, -23, 31};
    std::vector<int16_t> actual_state =
        MakeSmallInt16Input(kBatch * memory_size * kFilters, 41);
    std::vector<int16_t> expected_state = actual_state;
    std::vector<int32_t> actual_scratch(kBatch * kFilters);
    std::vector<int32_t> expected_scratch(kBatch * kFilters);
    std::vector<int32_t> actual_output_temp(kBatch * kUnits);
    std::vector<int32_t> expected_output_temp(kBatch * kUnits);
    std::vector<int8_t> actual_output(kBatch * kUnits);
    std::vector<int8_t> expected_output(kBatch * kUnits);

    reference_ops::EvalIntegerSVDF(
        &params, input_shape, input.data(), weights_feature_shape,
        weights_feature.data(), weights_time_shape, weights_time.data(),
        bias_shape, bias.data(), actual_state.data(), output_shape,
        actual_output.data(), actual_scratch.data(), actual_output_temp.data(),
        kScale1A, kScale1B, kScale2A, kScale2B, kInputZeroPoint,
        kOutputZeroPoint);
    ReferenceSvdfInteger(
        params, input_shape, input.data(), weights_feature_shape,
        weights_feature.data(), weights_time_shape, weights_time.data(),
        bias.data(), expected_state.data(), expected_output.data(),
        expected_scratch.data(), expected_output_temp.data(), kScale1A,
        kScale1B, kScale2A, kScale2B, kInputZeroPoint, kOutputZeroPoint);

    EXPECT_THAT(actual_state, ElementsAreArray(expected_state))
        << "size=" << size;
    EXPECT_THAT(actual_output, ElementsAreArray(expected_output))
        << "size=" << size;
  }
}

TEST(RvvOpsTest, LstmCwiseMulInt16MatchesPortableAcrossVectorBoundaries) {
  for (int size : Int16M2VectorLengths()) {
    constexpr int kBatch = 3;
    const int n_input = size;
    const std::vector<int16_t> input1 = MakeInt16Input(kBatch * n_input, 211);
    const std::vector<int16_t> input2 = MakeInt16Input(kBatch * n_input, 907);
    std::vector<int16_t> actual(kBatch * n_input);
    std::vector<int16_t> expected(kBatch * n_input);

    tensor_utils::CwiseMul(input1.data(), input2.data(), kBatch, n_input, 15,
                           actual.data());
    ReferenceLstmCwiseMulInt16(input1.data(), input2.data(), kBatch, n_input,
                               15, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, LstmCwiseAddInt16MatchesPortableAcrossVectorBoundaries) {
  for (int size : Int16M2VectorLengths()) {
    constexpr int kBatch = 2;
    const int n_input = size;
    std::vector<int16_t> input1 = MakeInt16Input(kBatch * n_input, 1231);
    std::vector<int16_t> input2 = MakeInt16Input(kBatch * n_input, 4567);
    if (kBatch * n_input >= 4) {
      input1[0] = std::numeric_limits<int16_t>::max();
      input2[0] = 1;
      input1[1] = std::numeric_limits<int16_t>::min();
      input2[1] = -1;
    }
    std::vector<int16_t> actual(kBatch * n_input);
    std::vector<int16_t> expected(kBatch * n_input);

    tensor_utils::CwiseAdd(input1.data(), input2.data(), kBatch, n_input,
                           actual.data());
    ReferenceLstmCwiseAddInt16(input1.data(), input2.data(), kBatch, n_input,
                               expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(RvvOpsTest, LstmCwiseClippingMatchesPortableAcrossVectorBoundaries) {
  for (int size : Int16M2VectorLengths()) {
    std::vector<float> actual_float = MakeInput(size, 0.25f);
    std::vector<float> expected_float = actual_float;
    tensor_utils::CwiseClipping(actual_float.data(), size, 2.5f);
    for (float& value : expected_float) {
      value = std::max(std::min(2.5f, value), -2.5f);
    }
    EXPECT_THAT(actual_float, Pointwise(FloatNear(1e-6f), expected_float))
        << "float size=" << size;

    std::vector<int16_t> actual_i16 = MakeInt16Input(size, 3109);
    std::vector<int16_t> expected_i16 = actual_i16;
    tensor_utils::CwiseClipping(actual_i16.data(), size,
                                static_cast<int16_t>(4096));
    for (int16_t& value : expected_i16) {
      value = std::max(std::min(static_cast<int16_t>(4096), value),
                       static_cast<int16_t>(-4096));
    }
    EXPECT_THAT(actual_i16, ElementsAreArray(expected_i16))
        << "int16 size=" << size;

    std::vector<int8_t> actual_i8 = MakeInt8Input(size, 71);
    std::vector<int8_t> expected_i8 = actual_i8;
    tensor_utils::CwiseClipping(actual_i8.data(), size,
                                static_cast<int8_t>(64));
    for (int8_t& value : expected_i8) {
      value = std::max(std::min(static_cast<int8_t>(64), value),
                       static_cast<int8_t>(-64));
    }
    EXPECT_THAT(actual_i8, ElementsAreArray(expected_i8))
        << "int8 size=" << size;
  }
}

TEST(RvvOpsTest, LstmSub1VectorMatchesPortableAcrossVectorBoundaries) {
  for (int size : Int16M2VectorLengths()) {
    const std::vector<float> input_float = MakeInput(size, -0.5f);
    std::vector<float> actual_float(size);
    std::vector<float> expected_float(size);
    tensor_utils::Sub1Vector(input_float.data(), size, actual_float.data());
    for (int i = 0; i < size; ++i) {
      expected_float[i] = 1.0f - input_float[i];
    }
    EXPECT_THAT(actual_float, Pointwise(FloatNear(1e-6f), expected_float))
        << "float size=" << size;

    const std::vector<int16_t> input_i16 = MakeInt16Input(size, 1777);
    std::vector<int16_t> actual_i16(size);
    std::vector<int16_t> expected_i16(size);
    tensor_utils::Sub1Vector(input_i16.data(), size, actual_i16.data());
    for (int i = 0; i < size; ++i) {
      expected_i16[i] = static_cast<int16_t>(32767 - input_i16[i]);
    }
    EXPECT_THAT(actual_i16, ElementsAreArray(expected_i16))
        << "int16 size=" << size;
  }
}

TEST(RvvOpsTest,
     ResizeNearestNeighborUint8MatchesScalarAcrossVectorBoundaries) {
  ResizeNearestNeighborParams params = {};
  params.align_corners = false;
  params.half_pixel_centers = false;

  for (int depth : UInt8M8VectorLengths()) {
    if (depth == 0) {
      continue;
    }
    constexpr int kBatch = 2;
    constexpr int kInputHeight = 3;
    constexpr int kInputWidth = 4;
    constexpr int kOutputHeight = 5;
    constexpr int kOutputWidth = 7;
    const RuntimeShape input_shape({kBatch, kInputHeight, kInputWidth, depth});
    const RuntimeShape output_size_shape({2});
    const int32_t output_size_data[2] = {kOutputHeight, kOutputWidth};
    const RuntimeShape output_shape(
        {kBatch, kOutputHeight, kOutputWidth, depth});
    const std::vector<uint8_t> input =
        MakeUint8Input(input_shape.FlatSize(), 19);
    std::vector<uint8_t> actual(output_shape.FlatSize());
    std::vector<uint8_t> expected(output_shape.FlatSize());

    optimized_ops::ResizeNearestNeighbor(params, input_shape, input.data(),
                                         output_size_shape, output_size_data,
                                         output_shape, actual.data());
    int out = 0;
    for (int b = 0; b < kBatch; ++b) {
      for (int y = 0; y < kOutputHeight; ++y) {
        const int in_y = std::min(
            static_cast<int>(std::floor(
                y * (static_cast<float>(kInputHeight) / kOutputHeight))),
            kInputHeight - 1);
        for (int x = 0; x < kOutputWidth; ++x) {
          const int in_x = std::min(
              static_cast<int>(std::floor(
                  x * (static_cast<float>(kInputWidth) / kOutputWidth))),
              kInputWidth - 1);
          const int in =
              ((b * kInputHeight + in_y) * kInputWidth + in_x) * depth;
          for (int c = 0; c < depth; ++c) {
            expected[out++] = input[in + c];
          }
        }
      }
    }

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "depth=" << depth;
  }
}

TEST(RvvOpsTest, ConcatenationFloatMatchesScalarAcrossVectorBoundaries) {
  for (int inner_size : ElementLengthsAroundByteVlmax<float>()) {
    if (inner_size == 0) {
      continue;
    }
    constexpr int kOuter = 3;
    const RuntimeShape input0_shape({kOuter, 2, inner_size});
    const RuntimeShape input1_shape({kOuter, 3, inner_size});
    const RuntimeShape output_shape({kOuter, 5, inner_size});
    const std::vector<float> input0 =
        MakeSpecialFloatInput(input0_shape.FlatSize(), 0.25f);
    const std::vector<float> input1 =
        MakeSpecialFloatInput(input1_shape.FlatSize(), -1.0f);
    const RuntimeShape* input_shapes[2] = {&input0_shape, &input1_shape};
    const float* input_data[2] = {input0.data(), input1.data()};
    ConcatenationParams params = {};
    params.axis = 1;
    params.inputs_count = 2;
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    reference_ops::Concatenation(params, input_shapes, input_data, output_shape,
                                 actual.data());
    int out = 0;
    for (int outer = 0; outer < kOuter; ++outer) {
      const int input0_base = outer * 2 * inner_size;
      for (int i = 0; i < 2 * inner_size; ++i) {
        expected[out++] = input0[input0_base + i];
      }
      const int input1_base = outer * 3 * inner_size;
      for (int i = 0; i < 3 * inner_size; ++i) {
        expected[out++] = input1[input1_base + i];
      }
    }

    ExpectFloatNearOrSpecial(actual, expected);
  }
}

TEST(RvvOpsTest, PadInt8MatchesScalarAcrossVectorBoundaries) {
  for (int input_depth : Int8M8VectorLengths()) {
    if (input_depth == 0) {
      continue;
    }
    PadParams params = {};
    const int left_padding[5] = {1, 0, 1, 2, 3};
    const int right_padding[5] = {0, 1, 2, 1, 4};
    params.left_padding_count = 5;
    params.right_padding_count = 5;
    std::copy(left_padding, left_padding + 5, params.left_padding);
    std::copy(right_padding, right_padding + 5, params.right_padding);
    const RuntimeShape input_shape({2, 2, 3, 4, input_depth});
    const RuntimeShape output_shape(
        {3, 3, 6, 7, input_depth + left_padding[4] + right_padding[4]});
    const int8_t pad_value = -9;
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 31);
    std::vector<int8_t> actual(output_shape.FlatSize());
    std::vector<int8_t> expected(output_shape.FlatSize(), pad_value);

    reference_ops::Pad(params, input_shape, input.data(), &pad_value,
                       output_shape, actual.data());
    for (int b = 0; b < input_shape.Dims(0); ++b) {
      for (int p = 0; p < input_shape.Dims(1); ++p) {
        for (int h = 0; h < input_shape.Dims(2); ++h) {
          for (int w = 0; w < input_shape.Dims(3); ++w) {
            for (int d = 0; d < input_shape.Dims(4); ++d) {
              expected[Offset(output_shape, b + left_padding[0],
                              p + left_padding[1], h + left_padding[2],
                              w + left_padding[3], d + left_padding[4])] =
                  input[Offset(input_shape, b, p, h, w, d)];
            }
          }
        }
      }
    }

    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "input_depth=" << input_depth;
  }
}

TEST(RvvOpsTest, GatherFloatMatchesScalarAcrossVectorBoundaries) {
  GatherParams params = {};
  params.axis = 1;
  params.batch_dims = 0;

  for (int inner_size : ElementLengthsAroundByteVlmax<float>()) {
    if (inner_size == 0) {
      continue;
    }
    constexpr int kOuter = 2;
    constexpr int kAxis = 5;
    const RuntimeShape input_shape({kOuter, kAxis, inner_size});
    const RuntimeShape coords_shape({6});
    const RuntimeShape output_shape(
        {kOuter, coords_shape.FlatSize(), inner_size});
    const std::vector<float> input =
        MakeSpecialFloatInput(input_shape.FlatSize(), 0.75f);
    const std::vector<int32_t> coords = {4, 0, 3, 1, 1, 2};
    std::vector<float> actual(output_shape.FlatSize());
    std::vector<float> expected(output_shape.FlatSize());

    ASSERT_EQ(
        reference_ops::Gather(params, input_shape, input.data(), coords_shape,
                              coords.data(), output_shape, actual.data()),
        kTfLiteOk);
    int out = 0;
    for (int outer = 0; outer < kOuter; ++outer) {
      for (int coord : coords) {
        const int in = (outer * kAxis + coord) * inner_size;
        for (int i = 0; i < inner_size; ++i) {
          expected[out++] = input[in + i];
        }
      }
    }

    ExpectFloatNearOrSpecial(actual, expected);
  }
}

#ifdef RVV_OPS_TEST_WITH_SINGLE_OP_MODEL
class StablehloElementwiseOpModel : public SingleOpModel {
 public:
  StablehloElementwiseOpModel(BuiltinOperator op, TensorType type,
                              const std::vector<int>& shape) {
    input1_ = AddInput({type, shape});
    input2_ = AddInput({type, shape});
    output_ = AddOutput({type, {}});
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() const { return input1_; }
  int input2() const { return input2_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input1_;
  int input2_;
  int output_;
};
#endif

TEST(RvvOpsTest, StablehloFloatElementwiseMatchesScalarAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    std::vector<float> input1;
    std::vector<float> input2;
    MakeStablehloSpecialFloatInputs(size, &input1, &input2);
    std::vector<float> actual(size);
    std::vector<float> expected(size);

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kAdd>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = input1[i] + input2[i];
    ExpectFloatNearOrSpecial(actual, expected);

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kMul>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = input1[i] * input2[i];
    ExpectFloatNearOrSpecial(actual, expected);

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kMax>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = std::max(input1[i], input2[i]);
    ExpectFloatNearOrSpecial(actual, expected);

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kMin>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = std::min(input1[i], input2[i]);
    ExpectFloatNearOrSpecial(actual, expected);
  }
}

#ifdef RVV_OPS_TEST_WITH_SINGLE_OP_MODEL
TEST(RvvOpsTest, StablehloFloatKernelEntryMatchesScalarAcrossVectorBoundaries) {
  for (int size : Float32M4VectorLengths()) {
    if (size == 0) {
      continue;
    }
    std::vector<float> input1;
    std::vector<float> input2;
    MakeStablehloSpecialFloatInputs(size, &input1, &input2);

    const struct {
      BuiltinOperator op;
      ops::builtin::ComputationType computation;
    } test_cases[] = {
        {BuiltinOperator_STABLEHLO_ADD, ops::builtin::ComputationType::kAdd},
        {BuiltinOperator_STABLEHLO_MULTIPLY,
         ops::builtin::ComputationType::kMul},
        {BuiltinOperator_STABLEHLO_MAXIMUM,
         ops::builtin::ComputationType::kMax},
        {BuiltinOperator_STABLEHLO_MINIMUM,
         ops::builtin::ComputationType::kMin},
    };

    for (const auto& test_case : test_cases) {
      StablehloElementwiseOpModel model(test_case.op, TensorType_FLOAT32,
                                        {size});
      model.PopulateTensor<float>(model.input1(), input1);
      model.PopulateTensor<float>(model.input2(), input2);
      ASSERT_EQ(model.Invoke(), kTfLiteOk);
      std::vector<float> expected(size);
      for (int i = 0; i < size; ++i) {
        switch (test_case.computation) {
          case ops::builtin::ComputationType::kAdd:
            expected[i] = input1[i] + input2[i];
            break;
          case ops::builtin::ComputationType::kMul:
            expected[i] = input1[i] * input2[i];
            break;
          case ops::builtin::ComputationType::kMax:
            expected[i] = std::max(input1[i], input2[i]);
            break;
          case ops::builtin::ComputationType::kMin:
            expected[i] = std::min(input1[i], input2[i]);
            break;
          default:
            TFL_UNREACHABLE();
        }
      }
      ExpectFloatNearOrSpecial(model.GetOutput<float>(), expected);
    }
  }
}
#endif

TEST(RvvOpsTest, StablehloInt8ElementwiseMatchesScalarAcrossVectorBoundaries) {
  for (int size : Int8M4VectorLengths()) {
    std::vector<int8_t> input1;
    std::vector<int8_t> input2;
    MakeSpecialInt8Inputs(size, &input1, &input2);
    std::vector<int8_t> actual(size);
    std::vector<int8_t> expected(size);

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kAdd>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) {
      expected[i] = static_cast<int8_t>(input1[i] + input2[i]);
    }
    EXPECT_THAT(actual, ElementsAreArray(expected)) << "add size=" << size;

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kMul>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) {
      expected[i] = static_cast<int8_t>(input1[i] * input2[i]);
    }
    EXPECT_THAT(actual, ElementsAreArray(expected)) << "mul size=" << size;

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kAnd>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = input1[i] & input2[i];
    EXPECT_THAT(actual, ElementsAreArray(expected)) << "and size=" << size;
  }
}

#ifdef RVV_OPS_TEST_WITH_SINGLE_OP_MODEL
TEST(RvvOpsTest, StablehloInt8KernelEntryMatchesScalarAcrossVectorBoundaries) {
  for (int size : Int8M4VectorLengths()) {
    if (size == 0) {
      continue;
    }
    std::vector<int8_t> input1;
    std::vector<int8_t> input2;
    MakeSpecialInt8Inputs(size, &input1, &input2);

    const struct {
      BuiltinOperator op;
      ops::builtin::ComputationType computation;
    } test_cases[] = {
        {BuiltinOperator_STABLEHLO_ADD, ops::builtin::ComputationType::kAdd},
        {BuiltinOperator_STABLEHLO_MULTIPLY,
         ops::builtin::ComputationType::kMul},
        {BuiltinOperator_STABLEHLO_AND, ops::builtin::ComputationType::kAnd},
    };

    for (const auto& test_case : test_cases) {
      StablehloElementwiseOpModel model(test_case.op, TensorType_INT8, {size});
      model.PopulateTensor<int8_t>(model.input1(), input1);
      model.PopulateTensor<int8_t>(model.input2(), input2);
      ASSERT_EQ(model.Invoke(), kTfLiteOk);
      std::vector<int8_t> expected(size);
      for (int i = 0; i < size; ++i) {
        switch (test_case.computation) {
          case ops::builtin::ComputationType::kAdd:
            expected[i] = static_cast<int8_t>(input1[i] + input2[i]);
            break;
          case ops::builtin::ComputationType::kMul:
            expected[i] = static_cast<int8_t>(input1[i] * input2[i]);
            break;
          case ops::builtin::ComputationType::kAnd:
            expected[i] = input1[i] & input2[i];
            break;
          default:
            TFL_UNREACHABLE();
        }
      }
      EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray(expected));
    }
  }
}
#endif

TEST(RvvOpsTest, StablehloInt32MinMaxMatchesScalarAcrossVectorBoundaries) {
  for (int size : Int32M4VectorLengths()) {
    std::vector<int32_t> input1 = MakeInt32Input(size, 19);
    std::vector<int32_t> input2 = MakeInt32Input(size, 701);
    if (size >= 4) {
      input1[0] = std::numeric_limits<int32_t>::min();
      input2[0] = std::numeric_limits<int32_t>::max();
      input1[1] = std::numeric_limits<int32_t>::max();
      input2[1] = std::numeric_limits<int32_t>::min();
      input1[2] = 0;
      input2[2] = -1;
      input1[3] = -1;
      input2[3] = 0;
    }
    std::vector<int32_t> actual(size);
    std::vector<int32_t> expected(size);

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kMax>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = std::max(input1[i], input2[i]);
    EXPECT_THAT(actual, ElementsAreArray(expected)) << "max size=" << size;

    ASSERT_TRUE(ops::builtin::RvvStablehloElementwiseFlat<
                ops::builtin::ComputationType::kMin>(
        input1.data(), input2.data(), actual.data(), size));
    for (int i = 0; i < size; ++i) expected[i] = std::min(input1[i], input2[i]);
    EXPECT_THAT(actual, ElementsAreArray(expected)) << "min size=" << size;
  }
}

#if !TFLITE_SINGLE_ROUNDING
TEST(RvvOpsTest, LstmCwiseMulInt16ToInt8MatchesPortableAcrossVectorBoundaries) {
  for (int size : Int16M2VectorLengths()) {
    constexpr int kBatch = 3;
    const int n_input = size;
    const std::vector<int16_t> input1 = MakeInt16Input(kBatch * n_input, 1201);
    const std::vector<int16_t> input2 = MakeInt16Input(kBatch * n_input, 431);
    std::vector<int8_t> actual(kBatch * n_input);
    std::vector<int8_t> expected(kBatch * n_input);

    tensor_utils::CwiseMul(input1.data(), input2.data(), 1073741824, -7, kBatch,
                           n_input, -5, actual.data());
    ReferenceLstmCwiseMulInt16ToInt8(input1.data(), input2.data(), 1073741824,
                                     -7, kBatch, n_input, -5, expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

TEST(
    RvvOpsTest,
    LstmVectorBatchVectorCwiseProductAccumulateMatchesPortableAcrossVectorBoundaries) {
  for (int size : Int16M2VectorLengths()) {
    constexpr int kBatch = 3;
    const std::vector<int16_t> vector = MakeInt16Input(size, 701);
    const std::vector<int16_t> batch_vector = MakeInt16Input(kBatch * size, 53);
    std::vector<int16_t> actual = MakeInt16Input(kBatch * size, 2909);
    std::vector<int16_t> expected = actual;

    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        vector.data(), size, batch_vector.data(), kBatch, 1073741824, -8,
        actual.data());
    ReferenceLstmVectorBatchVectorCwiseProductAccumulate(
        vector.data(), size, batch_vector.data(), kBatch, 1073741824, -8,
        expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}
#endif  // !TFLITE_SINGLE_ROUNDING

#endif  // USE_RVV

}  // namespace
}  // namespace tflite
