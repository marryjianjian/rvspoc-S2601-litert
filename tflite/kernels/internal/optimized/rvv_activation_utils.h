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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_ACTIVATION_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_ACTIVATION_UTILS_H_

#include <cmath>
#include <cstddef>

#include "tflite/kernels/internal/optimized/rvv_check.h"

namespace tflite {
namespace rvv_ops {

inline float LogisticFloatValue(float value) {
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

#ifdef USE_RVV

template <typename UnaryFn>
inline void UnaryFloatTransform(int size, const float* input_data,
                                float* output_data, UnaryFn fn) {
  int i = 0;
  while (i < size) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t input = __riscv_vle32_v_f32m4(input_data + i, vl);
    __riscv_vse32_v_f32m4(output_data + i, input, vl);
    for (size_t lane = 0; lane < vl; ++lane) {
      output_data[i + lane] = fn(output_data[i + lane]);
    }
    i += vl;
  }
}

inline void LogisticFloat(int size, const float* input_data,
                          float* output_data) {
  UnaryFloatTransform(size, input_data, output_data, LogisticFloatValue);
}

inline void TanhFloat(int size, const float* input_data, float* output_data) {
  UnaryFloatTransform(size, input_data, output_data,
                      [](float value) { return std::tanh(value); });
}

inline void SwishFloat(int size, const float* input_data, float* output_data) {
  UnaryFloatTransform(size, input_data, output_data, [](float value) {
    return value * LogisticFloatValue(value);
  });
}

#endif  // USE_RVV

}  // namespace rvv_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_ACTIVATION_UTILS_H_
