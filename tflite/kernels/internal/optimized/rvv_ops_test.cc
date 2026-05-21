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

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/conv.h"
#include "tflite/kernels/internal/optimized/integer_ops/fully_connected.h"
#include "tflite/kernels/internal/optimized/integer_ops/leaky_relu.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/reduce.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/reference/add.h"
#include "tflite/kernels/internal/reference/div.h"
#include "tflite/kernels/internal/reference/integer_ops/add.h"
#include "tflite/kernels/internal/reference/integer_ops/conv.h"
#include "tflite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tflite/kernels/internal/reference/integer_ops/mul.h"
#include "tflite/kernels/internal/reference/integer_ops/pooling.h"
#include "tflite/kernels/internal/reference/mul.h"
#include "tflite/kernels/internal/reference/quantize.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/reference/requantize.h"
#include "tflite/kernels/internal/reference/sub.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace {

#ifdef USE_RVV

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

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

std::vector<int8_t> MakeInt8Input(int size, int offset) {
  std::vector<int8_t> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<int8_t>(((i * 37 + offset) % 255) - 127);
  }
  return values;
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
      0, 1, 2, vlmax - 1, vlmax, vlmax + 1,
      2 * vlmax - 1, 2 * vlmax, 2 * vlmax + 3,
  };
  lengths.erase(std::remove_if(lengths.begin(), lengths.end(),
                               [](int length) { return length < 0; }),
                lengths.end());
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  return lengths;
}

std::vector<int> Float32M4VectorLengths() {
  return VectorLengthsAroundVlmax(
      static_cast<int>(__riscv_vsetvlmax_e32m4()));
}

std::vector<int> Int8M1VectorLengths() {
  return VectorLengthsAroundVlmax(
      static_cast<int>(__riscv_vsetvlmax_e8m1()));
}

std::vector<int> Int16M2VectorLengths() {
  return VectorLengthsAroundVlmax(
      static_cast<int>(__riscv_vsetvlmax_e16m2()));
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
      expected[i] = static_cast<float>(
          params.scale * (input[i] - params.zero_point));
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
      expected[i] = static_cast<float>(
          params.scale * (input[i] - params.zero_point));
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
      expected[i] = static_cast<float>(
          params.scale * (input[i] - params.zero_point));
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
                              kInputZeroPoint, kOutputZeroPoint,
                              actual.data());
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
                              kInputZeroPoint, kOutputZeroPoint,
                              actual.data());
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
                              kInputZeroPoint, kOutputZeroPoint,
                              actual.data());
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
                              kInputZeroPoint, kOutputZeroPoint,
                              actual.data());
    reference_ops::Requantize(input.data(), size, kMultiplier, kShift,
                              kInputZeroPoint, kOutputZeroPoint,
                              expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
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
    const std::vector<int8_t> input =
        MakeInt8Input(input_shape.FlatSize(), 19);
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
    const std::vector<int8_t> input =
        MakeInt8Input(input_shape.FlatSize(), 61);
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
    const std::vector<int8_t> input =
        MakeInt8Input(input_shape.FlatSize(), 29);
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
      expected[i] = Clamp(kBroadcastValue + input[i],
                          params.float_activation_min,
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

    EXPECT_TRUE(optimized_ops::AveragePool(
        params, input_shape, input.data(), output_shape, actual.data()));
    EXPECT_TRUE(reference_ops::AveragePool(
        params, input_shape, input.data(), output_shape, expected.data()));

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

    EXPECT_TRUE(optimized_ops::AveragePool(
        params, input_shape, input.data(), output_shape, actual.data()));
    EXPECT_TRUE(reference_ops::AveragePool(
        params, input_shape, input.data(), output_shape, expected.data()));

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
    const std::vector<int8_t> input = MakeInt8Input(input_shape.FlatSize(), 181);
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
    const std::vector<float> input =
        MakeInput(kOuterSize * axis_size, -0.125f);
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
    const std::vector<float> input =
        MakeInput(kOuterSize * axis_size, 0.375f);
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
    const std::vector<float> input =
        MakeInput(kOuterSize * axis_size, -0.5f);
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
    const std::vector<float> input =
        MakeInput(kOuterSize * axis_size, 0.25f);
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
    const std::vector<float> input =
        MakeInput(kOuterSize * axis_size, -0.75f);
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
    const std::vector<float> input =
        MakeInput(kOuterSize * axis_size, 0.75f);
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

    optimized_ops::SubWithActivation<float>(
        params, shape, input1.data(), shape, input2.data(), shape,
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

    optimized_ops::LeakyRelu(params, shape, input.data(), shape,
                             actual.data());
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

TEST(RvvOpsTest,
     FloatPReluElementWiseMatchesReferenceAcrossVectorBoundaries) {
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
    reference_ops::Softmax(params, shape, input.data(), shape,
                           expected.data());

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

TEST(RvvOpsTest, FloatMulSimpleBroadcastMatchesReferenceAcrossVectorBoundaries) {
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
      expected[i] = Clamp(kBroadcastValue * input[i],
                          params.float_activation_min,
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

    optimized_integer_ops::AddElementwiseInt8(
        size, params, input1.data(), input2.data(), actual.data());
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

    optimized_integer_ops::AddScalarBroadcast(
        size, params, kBroadcastValue, input.data(), actual.data());
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

    optimized_integer_ops::SubElementwiseInt8(
        size, params, input1.data(), input2.data(), actual.data());
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

      optimized_integer_ops::MulElementwise(
          size, params, input1.data(), input2.data(), actual.data());
      reference_integer_ops::MulElementwise(
          size, params, input1.data(), input2.data(), expected.data());

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

      optimized_integer_ops::MulSimpleBroadcast(
          size, params, kBroadcastValue, input.data(), actual.data());
      reference_integer_ops::MulElementwise(
          size, params, broadcast_values.data(), input.data(),
          expected.data());

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

      optimized_ops::MulElementwise(size, params, input1.data(),
                                    input2.data(), actual.data());
      reference_ops::MulElementwise(size, params, input1.data(),
                                    input2.data(), expected.data());

      EXPECT_THAT(actual, ElementsAreArray(expected))
          << "size=" << size << " output_shift=" << params.output_shift;
    }
  }
}

TEST(RvvOpsTest, Uint8AddScalarBroadcastMatchesReferenceAcrossVectorBoundaries) {
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

    optimized_integer_ops::AddElementwiseInt16(
        size, params, input1.data(), input2.data(), actual.data());
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

    optimized_integer_ops::QuantizeLeakyRelu(
        params, shape, input.data(), shape, actual.data());
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

    optimized_integer_ops::SubElementwiseInt16(
        size, params, input1.data(), input2.data(), actual.data());
    reference_ops::SubElementwise(size, params, input1.data(), input2.data(),
                                  expected.data());

    EXPECT_THAT(actual, ElementsAreArray(expected)) << "size=" << size;
  }
}

#endif  // USE_RVV

}  // namespace
}  // namespace tflite
