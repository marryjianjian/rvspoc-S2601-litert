/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <type_traits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/optimized/rvv_ops.h"
#include "tflite/kernels/internal/quantization_util.h"
#include "tflite/kernels/internal/reference/binary_function.h"
#include "tflite/kernels/internal/reference/integer_ops/add.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/tensor.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace squared_difference {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;
  ArithmeticParams arithmetic_params;
};

template <typename T>
T SquaredDifference(T input1, T input2) {
  const T difference = input1 - input2;
  return difference * difference;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = input2->type;

  // Ensure the quantization parameters are equivalent.
  if (input1->type == kTfLiteInt8) {
    const auto& input1_quantization_params = input1->params;
    const auto& input2_quantization_params = input2->params;
    const auto& output_quantization_params = output->params;
    const int32_t integer_type_min = std::numeric_limits<int8_t>::min();
    const int32_t integer_type_max = std::numeric_limits<int8_t>::max();
    TF_LITE_ENSURE(context,
                   input1_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   input1_quantization_params.zero_point <= integer_type_max);
    TF_LITE_ENSURE(context,
                   input2_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   input2_quantization_params.zero_point <= integer_type_max);
    TF_LITE_ENSURE(context,
                   output_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   output_quantization_params.zero_point <= integer_type_max);
    data->arithmetic_params.input1_offset =
        -input1_quantization_params.zero_point;
    data->arithmetic_params.input2_offset =
        -input2_quantization_params.zero_point;
    data->arithmetic_params.output_offset =
        output_quantization_params.zero_point;

    // shift to make integer for scales.
    data->arithmetic_params.left_shift = 7;
    const double twice_max_input_scale =
        2 * std::max(input1_quantization_params.scale,
                     input2_quantization_params.scale);
    const double real_input1_multiplier =
        input1_quantization_params.scale / twice_max_input_scale;
    double real_input2_multiplier =
        input2_quantization_params.scale / twice_max_input_scale;
    const double real_output_multiplier =
        (twice_max_input_scale * twice_max_input_scale) /
        ((1 << data->arithmetic_params.left_shift * 2) *
         output_quantization_params.scale);
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &data->arithmetic_params.input1_multiplier,
        &data->arithmetic_params.input1_shift);
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &data->arithmetic_params.input2_multiplier,
        &data->arithmetic_params.input2_shift);
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->arithmetic_params.output_multiplier,
        &data->arithmetic_params.output_shift);
    data->arithmetic_params.quantized_activation_min =
        std::numeric_limits<int8_t>::min();
    data->arithmetic_params.quantized_activation_max =
        std::numeric_limits<int8_t>::max();
  }

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

inline int8_t SquaredDifference(int8_t x, int8_t y,
                                const ArithmeticParams& params) {
  const int32_t input1_val = params.input1_offset + x;
  const int32_t input2_val = params.input2_offset + y;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
  const int32_t scaled_input1_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input1_val, params.input1_multiplier, params.input1_shift);
  const int32_t scaled_input2_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input2_val, params.input2_multiplier, params.input2_shift);
  const int32_t raw_diff = scaled_input1_val - scaled_input2_val;

  // Max of this is 255^2 * (1 << 14), so won't overflow 32 bits.
  const int32_t squared_raw_diff = raw_diff * raw_diff;
  const int32_t raw_output =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          squared_raw_diff, params.output_multiplier, params.output_shift) +
      params.output_offset;
  const int32_t clamped_output =
      std::min(params.quantized_activation_max,
               std::max(params.quantized_activation_min, raw_output));
  return static_cast<int8_t>(clamped_output);
}

void RiscvScalarSquaredDifferenceFloatFlat(int flat_size,
                                           const float* input1_data,
                                           const float* input2_data,
                                           float* output_data) {
  ruy::profiler::ScopeLabel label("SquaredDifference/Float/RiscvScalar");
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = SquaredDifference(input1_data[i], input2_data[i]);
  }
}

void RiscvScalarSquaredDifferenceInt32Flat(int flat_size,
                                           const int32_t* input1_data,
                                           const int32_t* input2_data,
                                           int32_t* output_data) {
  ruy::profiler::ScopeLabel label("SquaredDifference/Int32/RiscvScalar");
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = SquaredDifference(input1_data[i], input2_data[i]);
  }
}

void RiscvScalarSquaredDifferenceInt8Flat(
    int flat_size, const ArithmeticParams& params, const int8_t* input1_data,
    const int8_t* input2_data, int8_t* output_data) {
  ruy::profiler::ScopeLabel label("SquaredDifference/Int8/RiscvScalar");
  reference_integer_ops::CheckArithmeticParams(params);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = SquaredDifference(input1_data[i], input2_data[i], params);
  }
}

#ifdef USE_RVV
void RvvSquaredDifferenceFloatFlat(int flat_size, const float* input1_data,
                                   const float* input2_data,
                                   float* output_data) {
  ruy::profiler::ScopeLabel label("SquaredDifference/Float/RVV");
  int i = 0;
  while (i < flat_size) {
    const size_t vl = __riscv_vsetvl_e32m4(flat_size - i);
    const vfloat32m4_t input1 = __riscv_vle32_v_f32m4(input1_data + i, vl);
    const vfloat32m4_t input2 = __riscv_vle32_v_f32m4(input2_data + i, vl);
    const vfloat32m4_t diff = __riscv_vfsub_vv_f32m4(input1, input2, vl);
    const vfloat32m4_t squared = __riscv_vfmul_vv_f32m4(diff, diff, vl);
    __riscv_vse32_v_f32m4(output_data + i, squared, vl);
    i += vl;
  }
}

void RvvSquaredDifferenceInt32Flat(int flat_size, const int32_t* input1_data,
                                   const int32_t* input2_data,
                                   int32_t* output_data) {
  ruy::profiler::ScopeLabel label("SquaredDifference/Int32/RVV");
  int i = 0;
  while (i < flat_size) {
    const size_t vl = __riscv_vsetvl_e32m4(flat_size - i);
    const vint32m4_t input1 = __riscv_vle32_v_i32m4(input1_data + i, vl);
    const vint32m4_t input2 = __riscv_vle32_v_i32m4(input2_data + i, vl);
    const vint32m4_t diff = __riscv_vsub_vv_i32m4(input1, input2, vl);
    const vint32m4_t squared = __riscv_vmul_vv_i32m4(diff, diff, vl);
    __riscv_vse32_v_i32m4(output_data + i, squared, vl);
    i += vl;
  }
}

#if !TFLITE_SINGLE_ROUNDING
void RvvSquaredDifferenceInt8Flat(int flat_size,
                                  const ArithmeticParams& params,
                                  const int8_t* input1_data,
                                  const int8_t* input2_data,
                                  int8_t* output_data) {
  ruy::profiler::ScopeLabel label("SquaredDifference/Int8/RVV");
  reference_integer_ops::CheckArithmeticParams(params);
  int i = 0;
  while (i < flat_size) {
    const size_t vl = __riscv_vsetvl_e8m1(flat_size - i);
    const vint8m1_t input1_i8 = __riscv_vle8_v_i8m1(input1_data + i, vl);
    const vint8m1_t input2_i8 = __riscv_vle8_v_i8m1(input2_data + i, vl);
    const vint16m2_t input1_i16 = __riscv_vsext_vf2_i16m2(input1_i8, vl);
    const vint16m2_t input2_i16 = __riscv_vsext_vf2_i16m2(input2_i8, vl);
    const vint32m4_t input1 = __riscv_vadd_vx_i32m4(
        __riscv_vsext_vf2_i32m4(input1_i16, vl), params.input1_offset, vl);
    const vint32m4_t input2 = __riscv_vadd_vx_i32m4(
        __riscv_vsext_vf2_i32m4(input2_i16, vl), params.input2_offset, vl);
    const vint32m4_t shifted_input1 =
        __riscv_vsll_vx_i32m4(input1, params.left_shift, vl);
    const vint32m4_t shifted_input2 =
        __riscv_vsll_vx_i32m4(input2, params.left_shift, vl);
    const vint32m4_t scaled_input1 =
        rvv_ops::MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1, params.input1_multiplier, params.input1_shift, vl);
    const vint32m4_t scaled_input2 =
        rvv_ops::MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2, params.input2_multiplier, params.input2_shift, vl);
    const vint32m4_t raw_diff =
        __riscv_vsub_vv_i32m4(scaled_input1, scaled_input2, vl);
    const vint32m4_t squared_raw_diff =
        __riscv_vmul_vv_i32m4(raw_diff, raw_diff, vl);
    vint32m4_t raw_output =
        rvv_ops::MultiplyByQuantizedMultiplierSmallerThanOneExp(
            squared_raw_diff, params.output_multiplier, params.output_shift,
            vl);
    raw_output = __riscv_vadd_vx_i32m4(raw_output, params.output_offset, vl);
    raw_output = __riscv_vmax_vx_i32m4(raw_output,
                                       params.quantized_activation_min, vl);
    raw_output = __riscv_vmin_vx_i32m4(raw_output,
                                       params.quantized_activation_max, vl);

    const vint16m2_t narrowed_i16 = __riscv_vnsra_wx_i16m2(raw_output, 0, vl);
    const vint8m1_t narrowed_i8 = __riscv_vnsra_wx_i8m1(narrowed_i16, 0, vl);
    __riscv_vse8_v_i8m1(output_data + i, narrowed_i8, vl);
    i += vl;
  }
}
#endif  // !TFLITE_SINGLE_ROUNDING
#endif  // USE_RVV

template <typename T>
void EvalQuantizedSquaredDifference(TfLiteContext* context, TfLiteNode* node,
                                    const OpData* data,
                                    const TfLiteTensor* input1,
                                    const TfLiteTensor* input2,
                                    TfLiteTensor* output) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  if (data->requires_broadcast) {
    reference_integer_ops::BroadcastBinaryFunction4DSlow(
        op_data->arithmetic_params, GetTensorShape(input1),
        GetTensorData<T>(input1), GetTensorShape(input2),
        GetTensorData<T>(input2), GetTensorShape(output),
        GetTensorData<T>(output), reference_integer_ops::CheckArithmeticParams,
        SquaredDifference);
  } else {
    const int flat_size = GetTensorShape(input1).FlatSize();
#if defined(USE_RVV) && !TFLITE_SINGLE_ROUNDING
    RvvSquaredDifferenceInt8Flat(
        flat_size, op_data->arithmetic_params, GetTensorData<int8_t>(input1),
        GetTensorData<int8_t>(input2), GetTensorData<int8_t>(output));
#elif defined(__riscv)
    RiscvScalarSquaredDifferenceInt8Flat(
        flat_size, op_data->arithmetic_params, GetTensorData<int8_t>(input1),
        GetTensorData<int8_t>(input2), GetTensorData<int8_t>(output));
#else
    reference_integer_ops::ElementWise(
        flat_size, op_data->arithmetic_params, GetTensorData<int8_t>(input1),
        GetTensorData<int8_t>(input2), GetTensorData<int8_t>(output),
        reference_integer_ops::CheckArithmeticParams, SquaredDifference);
#endif
  }
}

template <typename T>
void EvalSquaredDifference(TfLiteContext* context, TfLiteNode* node,
                           const OpData* data, const TfLiteTensor* input1,
                           const TfLiteTensor* input2, TfLiteTensor* output) {
  if (data->requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output), SquaredDifference<T>);
  } else {
#ifdef USE_RVV
    const int flat_size =
        MatchingFlatSize(GetTensorShape(input1), GetTensorShape(input2),
                         GetTensorShape(output));
    if constexpr (std::is_same<T, float>::value) {
      RvvSquaredDifferenceFloatFlat(flat_size, GetTensorData<float>(input1),
                                    GetTensorData<float>(input2),
                                    GetTensorData<float>(output));
    } else if constexpr (std::is_same<T, int32_t>::value) {
      RvvSquaredDifferenceInt32Flat(flat_size, GetTensorData<int32_t>(input1),
                                    GetTensorData<int32_t>(input2),
                                    GetTensorData<int32_t>(output));
    } else {
      reference_ops::BinaryFunction<T, T, T>(
          GetTensorShape(input1), GetTensorData<T>(input1),
          GetTensorShape(input2), GetTensorData<T>(input2),
          GetTensorShape(output), GetTensorData<T>(output),
          SquaredDifference<T>);
    }
#elif defined(__riscv)
    const int flat_size =
        MatchingFlatSize(GetTensorShape(input1), GetTensorShape(input2),
                         GetTensorShape(output));
    if constexpr (std::is_same<T, float>::value) {
      RiscvScalarSquaredDifferenceFloatFlat(
          flat_size, GetTensorData<float>(input1), GetTensorData<float>(input2),
          GetTensorData<float>(output));
    } else if constexpr (std::is_same<T, int32_t>::value) {
      RiscvScalarSquaredDifferenceInt32Flat(
          flat_size, GetTensorData<int32_t>(input1),
          GetTensorData<int32_t>(input2), GetTensorData<int32_t>(output));
    } else {
      reference_ops::BinaryFunction<T, T, T>(
          GetTensorShape(input1), GetTensorData<T>(input1),
          GetTensorShape(input2), GetTensorData<T>(input2),
          GetTensorShape(output), GetTensorData<T>(output),
          SquaredDifference<T>);
    }
#else
    reference_ops::BinaryFunction<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output), SquaredDifference<T>);
#endif
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  ruy::profiler::ScopeLabel label("SquaredDifference");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (output->type == kTfLiteFloat32) {
    EvalSquaredDifference<float>(context, node, data, input1, input2, output);
  } else if (output->type == kTfLiteInt32) {
    EvalSquaredDifference<int32_t>(context, node, data, input1, input2, output);
  } else if (output->type == kTfLiteInt8) {
    EvalQuantizedSquaredDifference<int8_t>(context, node, data, input1, input2,
                                           output);
  } else {
    TF_LITE_KERNEL_LOG(
        context,
        "SquaredDifference only supports FLOAT32 and INT32 now, got %d.",
        output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace squared_difference

TfLiteRegistration* Register_SQUARED_DIFFERENCE() {
  static TfLiteRegistration r = {
      squared_difference::Init, squared_difference::Free,
      squared_difference::Prepare, squared_difference::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
