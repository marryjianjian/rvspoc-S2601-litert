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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_

#include <string.h>

#include <algorithm>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/cppmath.h"
#include "tflite/kernels/internal/optimized/cpu_check.h"
#include "tflite/kernels/internal/optimized/im2col_utils.h"
#include "tflite/kernels/internal/optimized/neon_check.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/quantization_util.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/strided_slice_logic.h"
#include "tflite/kernels/internal/tensor_utils.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8_t* input_data, const RuntimeShape& output_shape,
                    int8_t* output_data) {
  ruy::profiler::ScopeLabel label("MaxPool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int8_t acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          memset(acc, params.quantized_activation_min,
                 tranche_depth * sizeof(acc[0]));
          const int8_t* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8_t* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8_t* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                int8x16_t acc_reg = vld1q_s8(acc + channel);
                int8x16_t input_reg = vld1q_s8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg = vmaxq_s8(acc_reg, input_reg);
                vst1q_s8(acc + channel, acc_reg);
              }

              for (; channel <= tranche_depth - 8; channel += 8) {
                int8x8_t acc_reg = vld1_s8(acc + channel);
                int8x8_t input_reg = vld1_s8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vmax_s8(acc_reg, input_reg);
                vst1_s8(acc + channel, acc_reg);
              }
#elif defined(USE_RVV)
              for (; channel < tranche_depth;) {
                const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
                const vint8m1_t acc_reg =
                    __riscv_vle8_v_i8m1(acc + channel, vl);
                const vint8m1_t input =
                    __riscv_vle8_v_i8m1(input_channel_ptr, vl);
                __riscv_vse8_v_i8m1(
                    acc + channel, __riscv_vmax_vv_i8m1(acc_reg, input, vl),
                    vl);
                input_channel_ptr += vl;
                channel += vl;
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] = std::max(acc[channel], *input_channel_ptr++);
              }
              input_row_ptr += depth;
            }
          }
          int8_t* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                    out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 16; channel += 16) {
            int8x16_t a = vld1q_s8(acc + channel);
            a = vminq_s8(a, vdupq_n_s8(params.quantized_activation_max));
            a = vmaxq_s8(a, vdupq_n_s8(params.quantized_activation_min));
            vst1q_s8(output_ptr + channel, a);
          }
          for (; channel <= tranche_depth - 8; channel += 8) {
            int8x8_t a = vld1_s8(acc + channel);
            a = vmin_s8(a, vdup_n_s8(params.quantized_activation_max));
            a = vmax_s8(a, vdup_n_s8(params.quantized_activation_min));
            vst1_s8(output_ptr + channel, a);
          }
#elif defined(USE_RVV)
          for (; channel < tranche_depth;) {
            const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
            vint8m1_t output = __riscv_vle8_v_i8m1(acc + channel, vl);
            output = __riscv_vmin_vx_i8m1(
                output, params.quantized_activation_max, vl);
            output = __riscv_vmax_vx_i8m1(
                output, params.quantized_activation_min, vl);
            __riscv_vse8_v_i8m1(output_ptr + channel, output, vl);
            channel += vl;
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            int8_t a = acc[channel];
            a = std::max<int8_t>(a, params.quantized_activation_min);
            a = std::min<int8_t>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<int8_t>(a);
          }
        }
      }
    }
  }
}

#if defined(__riscv)
inline bool RiscvScalarAveragePool(
    const PoolParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  ruy::profiler::ScopeLabel label(
      "AveragePool/8bitWith32bitAccumulator/RISCVScalar");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  if (stride_height == 0 || stride_width == 0) return false;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin =
            (out_x * stride_width) - params.padding_values.width;
        const int in_y_origin =
            (out_y * stride_height) - params.padding_values.height;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end =
            std::min(params.filter_width, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end =
            std::min(params.filter_height, input_height - in_y_origin);
        const int filter_count =
            (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
        if (filter_count == 0) return false;
        for (int channel = 0; channel < depth; ++channel) {
          int32_t acc = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              acc += input_data[Offset(input_shape, batch, in_y, in_x,
                                       channel)];
            }
          }
          int32_t average = acc > 0
                                ? (acc + filter_count / 2) / filter_count
                                : (acc - filter_count / 2) / filter_count;
          average = std::max<int32_t>(average, params.quantized_activation_min);
          average = std::min<int32_t>(average, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<int8_t>(average);
        }
      }
    }
  }
  return true;
}
#endif  // defined(__riscv)

#ifdef USE_RVV
inline bool RvvAveragePool(const PoolParams& params,
                           const RuntimeShape& input_shape,
                           const int8_t* input_data,
                           const RuntimeShape& output_shape,
                           int8_t* output_data) {
  ruy::profiler::ScopeLabel label(
      "AveragePool/8bitWith32bitAccumulator/RVV");
  static constexpr int kPoolingAccTrancheSize = 256;
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  if (stride_height == 0 || stride_width == 0) return false;

  int32_t acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          if (filter_count == 0) return false;
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const int8_t* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8_t* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8_t* input_channel_ptr = input_row_ptr;
              for (int channel = 0; channel < tranche_depth;) {
                const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
                const vint8m1_t input =
                    __riscv_vle8_v_i8m1(input_channel_ptr, vl);
                const vint32m4_t input_wide =
                    __riscv_vsext_vf4_i32m4(input, vl);
                const vint32m4_t acc_reg =
                    __riscv_vle32_v_i32m4(acc + channel, vl);
                __riscv_vse32_v_i32m4(
                    acc + channel,
                    __riscv_vadd_vv_i32m4(acc_reg, input_wide, vl), vl);
                input_channel_ptr += vl;
                channel += vl;
              }
              input_row_ptr += depth;
            }
          }
          int8_t* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                    out_x, depth_base);
          for (int channel = 0; channel < tranche_depth;) {
            const size_t vl = __riscv_vsetvl_e32m4(tranche_depth - channel);
            vint32m4_t output = __riscv_vle32_v_i32m4(acc + channel, vl);
            const vbool8_t positive_mask =
                __riscv_vmsgt_vx_i32m4_b8(output, 0, vl);
            const vint32m4_t negative_nudge =
                __riscv_vmv_v_x_i32m4(-(filter_count / 2), vl);
            const vint32m4_t positive_nudge =
                __riscv_vmv_v_x_i32m4(filter_count / 2, vl);
            const vint32m4_t nudge = __riscv_vmerge_vvm_i32m4(
                negative_nudge, positive_nudge, positive_mask, vl);
            output = __riscv_vadd_vv_i32m4(output, nudge, vl);
            output = __riscv_vdiv_vx_i32m4(output, filter_count, vl);
            output = __riscv_vmax_vx_i32m4(
                output, params.quantized_activation_min, vl);
            output = __riscv_vmin_vx_i32m4(
                output, params.quantized_activation_max, vl);
            const vint16m2_t output_i16 =
                __riscv_vnsra_wx_i16m2(output, 0, vl);
            const vint8m1_t output_i8 =
                __riscv_vnsra_wx_i8m1(output_i16, 0, vl);
            __riscv_vse8_v_i8m1(output_ptr + channel, output_i8, vl);
            channel += vl;
          }
        }
      }
    }
  }
  return true;
}
#endif  // USE_RVV

inline bool AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const RuntimeShape& output_shape, int8_t* output_data) {
  ruy::profiler::ScopeLabel label("AveragePool/8bitWith32bitAccumulator");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  if (stride_height == 0 || stride_width == 0) return false;

#ifdef USE_RVV
  return RvvAveragePool(params, input_shape, input_data, output_shape,
                        output_data);
#elif defined(__riscv)
  return RiscvScalarAveragePool(params, input_shape, input_data, output_shape,
                                output_data);
#endif

  int32_t acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          if (filter_count == 0) return false;
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const int8_t* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8_t* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8_t* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                int16x4_t acc_reg[4];
                int8x16_t input_reg = vld1q_s8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg[0] = vget_low_s16(vmovl_s8(vget_low_s8(input_reg)));
                acc_reg[1] = vget_high_s16(vmovl_s8(vget_low_s8(input_reg)));
                acc_reg[2] = vget_low_s16(vmovl_s8(vget_high_s8(input_reg)));
                acc_reg[3] = vget_high_s16(vmovl_s8(vget_high_s8(input_reg)));
                for (int i = 0; i < 4; i++) {
                  vst1q_s32(
                      acc + channel + 4 * i,
                      vaddw_s16(vld1q_s32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
              for (; channel <= tranche_depth - 8; channel += 8) {
                int16x4_t acc_reg[2];
                int16x8_t input_reg = vmovl_s8(vld1_s8(input_channel_ptr));
                input_channel_ptr += 8;
                acc_reg[0] = vget_low_s16(input_reg);
                acc_reg[1] = vget_high_s16(input_reg);
                for (int i = 0; i < 2; i++) {
                  vst1q_s32(
                      acc + channel + 4 * i,
                      vaddw_s16(vld1q_s32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
#elif defined(USE_RVV)
              for (; channel < tranche_depth;) {
                const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
                const vint8m1_t input =
                    __riscv_vle8_v_i8m1(input_channel_ptr, vl);
                const vint32m4_t input_wide =
                    __riscv_vsext_vf4_i32m4(input, vl);
                const vint32m4_t acc_reg =
                    __riscv_vle32_v_i32m4(acc + channel, vl);
                __riscv_vse32_v_i32m4(
                    acc + channel,
                    __riscv_vadd_vv_i32m4(acc_reg, input_wide, vl), vl);
                input_channel_ptr += vl;
                channel += vl;
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] += *input_channel_ptr++;
              }
              input_row_ptr += depth;
            }
          }
          int8_t* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                    out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 8; channel += 8) {
            int16_t buf[8];
            for (int i = 0; i < 8; i++) {
              buf[i] =
                  acc[channel + i] > 0
                      ? (acc[channel + i] + filter_count / 2) / filter_count
                      : (acc[channel + i] - filter_count / 2) / filter_count;
            }
            int8x8_t buf8 = vqmovn_s16(vld1q_s16(buf));
            buf8 = vmin_s8(buf8, vdup_n_s8(params.quantized_activation_max));
            buf8 = vmax_s8(buf8, vdup_n_s8(params.quantized_activation_min));
            vst1_s8(output_ptr + channel, buf8);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            int16_t a = acc[channel] > 0
                            ? (acc[channel] + filter_count / 2) / filter_count
                            : (acc[channel] - filter_count / 2) / filter_count;
            a = std::max<int16_t>(a, params.quantized_activation_min);
            a = std::min<int16_t>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<int8_t>(a);
          }
        }
      }
    }
  }
  return true;
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_
