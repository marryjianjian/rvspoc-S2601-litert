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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
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

#endif  // USE_RVV

}  // namespace
}  // namespace tflite
