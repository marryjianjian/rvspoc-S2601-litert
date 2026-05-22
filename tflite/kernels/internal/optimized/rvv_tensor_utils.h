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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "tflite/kernels/internal/optimized/rvv_check.h"

namespace tflite {
namespace rvv_ops {

#ifdef USE_RVV

inline void CopyBytes(const void* src, void* dst, size_t bytes) {
  const uint8_t* input = static_cast<const uint8_t*>(src);
  uint8_t* output = static_cast<uint8_t*>(dst);
  size_t offset = 0;
  while (offset < bytes) {
    const size_t vl = __riscv_vsetvl_e8m8(bytes - offset);
    const vuint8m8_t values = __riscv_vle8_v_u8m8(input + offset, vl);
    __riscv_vse8_v_u8m8(output + offset, values, vl);
    offset += vl;
  }
}

template <typename T>
inline void CopyVector(const T* input, T* output, int size) {
  CopyBytes(input, output, static_cast<size_t>(size) * sizeof(T));
}

template <typename T>
inline void FillVector(T* output, int size, T value) {
  int i = 0;
  if constexpr (std::is_same<T, float>::value) {
    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e32m4(size - i);
      const vfloat32m4_t values = __riscv_vfmv_v_f_f32m4(value, vl);
      __riscv_vse32_v_f32m4(output + i, values, vl);
      i += vl;
    }
  } else if constexpr (std::is_same<T, int32_t>::value) {
    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e32m4(size - i);
      const vint32m4_t values = __riscv_vmv_v_x_i32m4(value, vl);
      __riscv_vse32_v_i32m4(output + i, values, vl);
      i += vl;
    }
  } else if constexpr (std::is_same<T, int16_t>::value) {
    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e16m4(size - i);
      const vint16m4_t values = __riscv_vmv_v_x_i16m4(value, vl);
      __riscv_vse16_v_i16m4(output + i, values, vl);
      i += vl;
    }
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e8m8(size - i);
      const vuint8m8_t values = __riscv_vmv_v_x_u8m8(value, vl);
      __riscv_vse8_v_u8m8(output + i, values, vl);
      i += vl;
    }
  } else if constexpr (std::is_same<T, int8_t>::value) {
    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e8m8(size - i);
      const vint8m8_t values = __riscv_vmv_v_x_i8m8(value, vl);
      __riscv_vse8_v_i8m8(output + i, values, vl);
      i += vl;
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e8m8(size - i);
      const vuint8m8_t values =
          __riscv_vmv_v_x_u8m8(static_cast<uint8_t>(value), vl);
      __riscv_vse8_v_u8m8(reinterpret_cast<uint8_t*>(output + i), values, vl);
      i += vl;
    }
  }

  for (; i < size; ++i) {
    output[i] = value;
  }
}

#else

inline void CopyBytes(const void* src, void* dst, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

template <typename T>
inline void CopyVector(const T* input, T* output, int size) {
  std::memcpy(output, input, static_cast<size_t>(size) * sizeof(T));
}

template <typename T>
inline void FillVector(T* output, int size, T value) {
  std::fill_n(output, size, value);
}

#endif  // USE_RVV

}  // namespace rvv_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_H_
