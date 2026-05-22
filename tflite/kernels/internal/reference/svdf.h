/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/tensor_utils.h"
#include "tflite/kernels/internal/types.h"

// SVDF op that compresses a fully connected op via low-rank matrix
// factorization. See https://research.google.com/pubs/archive/43813.pdf for
// details.

namespace tflite {
namespace reference_ops {

#if defined(USE_RVV)
inline float RvvSvdfDotProductFloat(const float* lhs, const float* rhs,
                                    int size) {
  float result = 0.0f;
  for (int i = 0; i < size;) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t lhs_values = __riscv_vle32_v_f32m4(lhs + i, vl);
    const vfloat32m4_t rhs_values = __riscv_vle32_v_f32m4(rhs + i, vl);
    const vfloat32m4_t product =
        __riscv_vfmul_vv_f32m4(lhs_values, rhs_values, vl);
    vfloat32m1_t scalar = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    scalar = __riscv_vfredusum_vs_f32m4_f32m1(product, scalar, vl);
    result += __riscv_vfmv_f_s_f32m1_f32(scalar);
    i += vl;
  }
  return result;
}

inline int32_t RvvSvdfDotProductInt16(const int16_t* lhs, const int16_t* rhs,
                                      int size) {
  vint32m1_t reduced = __riscv_vmv_v_x_i32m1(0, 1);
  for (int i = 0; i < size;) {
    const size_t vl = __riscv_vsetvl_e16m2(size - i);
    const vint16m2_t lhs_values = __riscv_vle16_v_i16m2(lhs + i, vl);
    const vint16m2_t rhs_values = __riscv_vle16_v_i16m2(rhs + i, vl);
    const vint32m4_t product =
        __riscv_vwmul_vv_i32m4(lhs_values, rhs_values, vl);
    reduced = __riscv_vredsum_vs_i32m4_i32m1(product, reduced, vl);
    i += vl;
  }
  return __riscv_vmv_x_s_i32m1_i32(reduced);
}

inline int32_t RvvSvdfDotProductInt8WithInputZeroPoint(
    const int8_t* weights, const int8_t* input, int input_zp, int size) {
  vint32m1_t reduced = __riscv_vmv_v_x_i32m1(0, 1);
  for (int i = 0; i < size;) {
    const size_t vl = __riscv_vsetvl_e8m1(size - i);
    const vint8m1_t weights_i8 = __riscv_vle8_v_i8m1(weights + i, vl);
    const vint8m1_t input_i8 = __riscv_vle8_v_i8m1(input + i, vl);
    const vint16m2_t weights_i16 = __riscv_vsext_vf2_i16m2(weights_i8, vl);
    const vint16m2_t input_i16 = __riscv_vsext_vf2_i16m2(input_i8, vl);
    const vint32m4_t weights_i32 = __riscv_vsext_vf2_i32m4(weights_i16, vl);
    vint32m4_t input_i32 = __riscv_vsext_vf2_i32m4(input_i16, vl);
    input_i32 = __riscv_vsub_vx_i32m4(input_i32, input_zp, vl);
    const vint32m4_t product =
        __riscv_vmul_vv_i32m4(weights_i32, input_i32, vl);
    reduced = __riscv_vredsum_vs_i32m4_i32m1(product, reduced, vl);
    i += vl;
  }
  return __riscv_vmv_x_s_i32m1_i32(reduced);
}

inline void RvvSvdfBatchVectorBatchVectorDotProductFloat(
    const float* weights_time_data, const float* state_ptr_batch,
    int memory_size, int num_filters, float* scratch_ptr_batch) {
  for (int filter = 0; filter < num_filters; ++filter) {
    scratch_ptr_batch[filter] = RvvSvdfDotProductFloat(
        weights_time_data + filter * memory_size,
        state_ptr_batch + filter * memory_size, memory_size);
  }
}

inline void RvvSvdfBatchVectorBatchVectorDotProductInt16(
    const int16_t* weights_time_data, const int16_t* state_ptr_batch,
    int memory_size, int num_filters, int32_t* scratch_ptr_batch) {
  for (int filter = 0; filter < num_filters; ++filter) {
    scratch_ptr_batch[filter] = RvvSvdfDotProductInt16(
        weights_time_data + filter * memory_size,
        state_ptr_batch + filter * memory_size, memory_size);
  }
}

inline void RvvSvdfCopyLatestActivationToState(const float* scratch_data,
                                               int size, int memory_size,
                                               float* state_data) {
  const ptrdiff_t stride = memory_size * static_cast<ptrdiff_t>(sizeof(float));
  float* latest_state = state_data + memory_size - 1;
  for (int i = 0; i < size;) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t scratch = __riscv_vle32_v_f32m4(scratch_data + i, vl);
    __riscv_vsse32_v_f32m4(latest_state + i * memory_size, stride, scratch,
                           vl);
    i += vl;
  }
}
#endif  // defined(USE_RVV)

static inline void ApplyTimeWeightsBiasAndActivation(
    int batch_size, int memory_size, int num_filters, int num_units, int rank,
    const float* const __restrict__ weights_time_data,
    const float* const __restrict__ bias_ptr, TfLiteFusedActivation activation,
    float* const __restrict__ state_ptr, float* const __restrict__ scratch_ptr,
    float* const __restrict__ output_ptr) {
  // Compute matmul(state, weights_time).
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch = state_ptr + b * memory_size * num_filters;
    float* scratch_ptr_batch = scratch_ptr + b * num_filters;
#if defined(USE_RVV)
    RvvSvdfBatchVectorBatchVectorDotProductFloat(
        weights_time_data, state_ptr_batch, memory_size, num_filters,
        scratch_ptr_batch);
#else
    tensor_utils::BatchVectorBatchVectorDotProduct(
        weights_time_data, state_ptr_batch, memory_size, num_filters,
        scratch_ptr_batch);
#endif
  }

  // Reduction sum.
  tensor_utils::ReductionSumVector(scratch_ptr, output_ptr,
                                   batch_size * num_units, rank);
  // Add bias if provided.
  if (bias_ptr) {
    tensor_utils::VectorBatchVectorAdd(bias_ptr, num_units, batch_size,
                                       output_ptr);
  }

  // Apply activation.
  tensor_utils::ApplyActivationToVector(output_ptr, batch_size * num_units,
                                        activation, output_ptr);
}

inline void EvalIntegerSVDF(
    const TfLiteSVDFParams* params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& weights_feature_shape,
    const int8_t* weights_feature_data, const RuntimeShape& weights_time_shape,
    const int16_t* weights_time_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, int16_t* state_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    int32_t* scratch_data, int32_t* output_temp_data, int32_t scale_1_a,
    int scale_1_b, int32_t scale_2_a, int scale_2_b, int32_t input_zp,
    int32_t output_zp) {
  const int n_rank = params->rank;
  const int n_batch = input_shape.Dims(0);
  const int n_input = input_shape.Dims(1);
  const int n_filter = weights_feature_shape.Dims(0);
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_shape.Dims(1);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_data + 1, state_data + n_batch * n_memory * n_filter,
            state_data);

  // Feature matmul.
  // Note: no need to clear the latest activation, matmul is not accumulative.
  {
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state_data + (n_memory - 1);
    for (int b = 0; b < n_batch; b++) {
      const int8_t* matrix_data = weights_feature_data;
      for (int r = 0; r < n_filter; r++) {
#if defined(USE_RVV)
        int32_t dot_prod = RvvSvdfDotProductInt8WithInputZeroPoint(
            matrix_data, input_data + b * n_input, input_zp, n_input);
        matrix_data += n_input;
#else
        int32_t dot_prod = 0;
        const int8_t* vector_in_batch = input_data + b * n_input;
        for (int c = 0; c < n_input; c++) {
          dot_prod += *matrix_data++ * (*vector_in_batch++ - input_zp);
        }
#endif
        dot_prod =
            MultiplyByQuantizedMultiplier(dot_prod, scale_1_a, scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod.
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      const int16_t* state_data_batch = state_data + b * n_memory * n_filter;
      int32_t* scratch_data_batch = scratch_data + b * n_filter;
#if defined(USE_RVV)
      RvvSvdfBatchVectorBatchVectorDotProductInt16(
          weights_time_data, state_data_batch, n_memory, n_filter,
          scratch_data_batch);
#else
      tensor_utils::BatchVectorBatchVectorDotProduct(
          weights_time_data, state_data_batch, n_memory, n_filter,
          scratch_data_batch);
#endif
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Reduce.
    tensor_utils::ReductionSumVector(scratch_data, output_temp_data,
                                     n_batch * n_unit, n_rank);
    // Add bias.
    if (bias_data) {
      tensor_utils::VectorBatchVectorAdd(bias_data, n_unit, n_batch,
                                         output_temp_data);
    }
    // Rescale.
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i) {
      int32_t x1 = output_temp_data[i];
      int32_t x2 = MultiplyByQuantizedMultiplier(x1, scale_2_a, scale_2_b);
      int32_t x3 = x2 + output_zp;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      output_data[i] = static_cast<int8_t>(x4);
    }
  }
}

inline void EvalFloatSVDF(
    const TfLiteSVDFParams* params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_feature_shape,
    const float* weights_feature_data, const RuntimeShape& weights_time_shape,
    const float* weights_time_data, const RuntimeShape& bias_shape,
    const float* bias_data, float* scratch_data, float* state_data,
    const RuntimeShape& output_shape, float* output_data) {
  const int rank = params->rank;
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.Dims(1);
  const int num_filters = weights_feature_shape.Dims(0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time_shape.Dims(1);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_data + 1, state_data + batch_size * memory_size * num_filters,
            state_data);

  // Clear scratch (the matmul is accumulative).
  std::fill_n(scratch_data, batch_size * num_filters, 0.0f);

  // Compute conv1d(inputs, weights_feature).
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      weights_feature_data, num_filters, input_size, input_data, batch_size,
      scratch_data);

  // Copy the latest activation from scratch into activation_state:
  // The last, i.e. (memory_size-1)th entry for each batch, and filter.
#if defined(USE_RVV)
  RvvSvdfCopyLatestActivationToState(scratch_data, batch_size * num_filters,
                                     memory_size, state_data);
#else
  for (int i = 0; i < batch_size * num_filters; ++i) {
    state_data[i * memory_size + memory_size - 1] = scratch_data[i];
  }
#endif

  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time_data,
      bias_data, params->activation, state_data, scratch_data, output_data);
}

inline void EvalHybridSVDF(
    const TfLiteSVDFParams* params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_feature_shape,
    const int8_t* weights_feature_data, const float weights_feature_scale,
    const RuntimeShape& weights_time_shape, const float* weights_time_data,
    const RuntimeShape& bias_shape, const float* bias_data, float* scratch,
    float* scaling_factors, int8_t* quantized_input, float* state,
    const RuntimeShape& output_shape, float* output_data, int32_t* zero_points,
    int32_t* row_sums, bool* compute_row_sums) {
  const int rank = params->rank;
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.Dims(1);
  const int num_filters = weights_feature_shape.Dims(0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time_shape.Dims(1);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state + 1, state + batch_size * memory_size * num_filters, state);

  // Clear scratch (the matmul is accumulative).
  std::fill_n(scratch, batch_size * num_filters, 0.0f);

  if (!tensor_utils::IsZeroVector(input_data, batch_size * input_size)) {
    // Quantize input from float to int8_t.
    tensor_utils::BatchQuantizeFloats(
        input_data, batch_size, input_size, quantized_input, scaling_factors,
        zero_points, params->asymmetric_quantize_inputs);
    for (int b = 0; b < batch_size; ++b) {
      scaling_factors[b] *= weights_feature_scale;
    }

    // Compute conv1d(inputs, weights_feature).
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weights_feature_data, num_filters, input_size, quantized_input,
        scaling_factors, batch_size, scratch,
        /*per_channel_scale=*/nullptr, zero_points,
        reinterpret_cast<int32_t*>(scratch), row_sums, compute_row_sums,
        /*context=*/nullptr);
  }
  // Copy the latest activation from scratch into activation_state:
  // The last, i.e. (memory_size-1)th entry for each batch, and filter.
#if defined(USE_RVV)
  RvvSvdfCopyLatestActivationToState(scratch, batch_size * num_filters,
                                     memory_size, state);
#else
  for (int i = 0; i < batch_size * num_filters; ++i) {
    state[i * memory_size + memory_size - 1] = scratch[i];
  }
#endif

  // TODO(b/174275776): can optimize hybrid case ~5% by unrolling loop in
  // applying time weights so that the inner loop multiplies eight elements at
  // a time.
  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time_data,
      bias_data, params->activation, state, scratch, output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
