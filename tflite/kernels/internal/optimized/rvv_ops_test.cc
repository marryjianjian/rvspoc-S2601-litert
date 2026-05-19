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
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/reference/add.h"
#include "tflite/kernels/internal/reference/div.h"
#include "tflite/kernels/internal/reference/integer_ops/add.h"
#include "tflite/kernels/internal/reference/integer_ops/mul.h"
#include "tflite/kernels/internal/reference/mul.h"
#include "tflite/kernels/internal/reference/sub.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace {

#ifdef USE_RVV

using ::testing::ElementsAreArray;

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
