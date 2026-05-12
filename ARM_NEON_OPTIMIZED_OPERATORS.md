# LiteRT ARM NEON 优化算子详细文档

> 本文档记录了 LiteRT 中每一个针对 ARM NEON 指令集做了专门优化的内置算子，包括优化实现方式、涉及的关键 NEON intrinsic/汇编指令、以及对应的源码文件。

---

## 目录

1. [编译时检测与宏机制](#1-编译时检测与宏机制)
2. [算术运算类](#2-算术运算类)
3. [激活函数类](#3-激活函数类)
4. [卷积/池化类](#4-卷积池化类)
5. [全连接层类](#5-全连接层类)
6. [张量操作类](#6-张量操作类)
7. [归约/比较/Arg 类](#7-归约比较arg-类)
8. [量化/反量化类](#8-量化反量化类)
9. [RNN/LSTM 类（通过 neon_tensor_utils）](#9-rnnlstm-类)
10. [Resize 类](#10-resize-类)
11. [其他有 NEON 优化的算子](#11-其他有-neon-优化的算子)

---

## 1. 编译时检测与宏机制

### 1.1 NEON 编译宏定义

**文件**: `tflite/kernels/internal/optimized/neon_check.h`

```cpp
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#include <arm_neon.h>
#endif
```

所有 NEON 优化代码通过 `#ifdef USE_NEON` 控制编译。当目标平台为 ARM 时自动启用。

### 1.2 统一架构检测宏

**文件**: `tflite/kernels/internal/optimized/arch_check.h`

```cpp
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define TFLITE_USE_NEON
#endif
```

定义了 `NEON_OR_PORTABLE(funcname, ...)` 宏，自动选择 `Neon##funcname` 或 `Portable##funcname`。

### 1.3 Dot Product 运行时检测

**文件**: `tflite/kernels/internal/optimized/cpu_check.cc`

```cpp
bool DetectArmNeonDotprod() {
#if defined __linux__ && defined __aarch64__
  const int kLocalHwcapAsimddp = 1 << 20;
  return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
#else
  return false;
#endif
}
```

通过 Linux `getauxval(AT_HWCAP)` 在运行时检测 ARM Dot Product 扩展（`HWCAP_ASIMDDP`）。

---

## 2. 算术运算类

### 2.1 ADD

**优化方式**: 编译期 `#ifdef USE_NEON` 选择 NEON 版本注册。

**NEON 优化实现**:
- **Float32**: 4×展开循环，使用 `vld1q_f32` 加载 4 个 float，`vaddq_f32` 向量加法，`vmaxq_f32`/`vminq_f32` 钳位，`vst1q_f32` 存储。一次循环处理 16 个元素。
- **Uint8 量化**: `vmovl_u8` 扩展到 int16，`vaddq_s16` 加法，`vqmovun_s16` 饱和窄化回 uint8。
- **Int8 量化**: 通过 `optimized_integer_ops::AddElementwiseInt8`，使用 `vmovl_s8`/`vmull_s8`/`vqrdmulh_s32`（SQRDMULH 指令）进行定点量化缩放。
- **Int16 量化**: 通过 `optimized_integer_ops::AddElementwiseInt16`，使用 `vmovl_s16`/`vmull_s32` 进行定点运算。
- **标量广播**: 当一个输入是标量时，使用 `vdupq_n_*` 广播标量到向量寄存器，避免内存加载。

**关键 NEON intrinsic**: `vld1q_f32`, `vaddq_f32`, `vmaxq_f32`, `vminq_f32`, `vmovl_u8`, `vaddq_s16`, `vqmovun_s16`, `vqrdmulh_s32`

**源码文件**:
- `tflite/kernels/add.cc` — 算子注册，编译期选择 `Register_ADD_NEON_OPT()`
- `tflite/kernels/internal/optimized/optimized_ops.h` — `AddElementwise()`, `AddScalarBroadcast()` (L1481-L1760)
- `tflite/kernels/internal/optimized/integer_ops/add.h` — `AddElementwiseInt8()`, `AddElementwiseInt16()`, `AddScalarBroadcast()`

### 2.2 SUB

**优化方式**: 与 ADD 相同的编译期 NEON 选择模式。

**NEON 优化实现**: 使用 `vsubq_f32` 替代 `vaddq_f32`，其余结构与 ADD 完全相同。

**关键 NEON intrinsic**: `vsubq_f32`, `vsubq_s16`

**源码文件**:
- `tflite/kernels/sub.cc` — `Register_SUB_NEON_OPT()`，编译期选择
- `tflite/kernels/internal/optimized/optimized_ops.h` — `SubElementwise()`

### 2.3 MUL

**优化方式**: 编译期 NEON 选择。

**NEON 优化实现**:
- **Float32**: 4×展开，`vmulq_f32` 向量乘法，加上 `vmaxq_f32`/`vminq_f32` 钳位。
- **Int32**: `vld1q_s32` 加载，`vmulq_s32` 乘法，`vmaxq_s32`/`vminq_s32` 钳位。
- **Uint8 量化**: 使用 `vmovl_u8` 扩展，`vmull_s16` 乘法，通过 `vqrdmulh_s32`（SQRDMULH）进行量化缩放。

**关键 NEON intrinsic**: `vmulq_f32`, `vmulq_s32`, `vmull_s16`, `vqrdmulh_s32`

**源码文件**:
- `tflite/kernels/mul.cc` — `Register_MUL_NEON_OPT()`，编译期选择
- `tflite/kernels/internal/optimized/optimized_ops.h` — `MulElementwise()`, `MulSimpleBroadcast()` (L1903-L2290)

### 2.4 DIV

**优化方式**: 编译期 NEON 选择。

**NEON 优化实现**: Float32 使用 `vdivq_f32`（AArch64 专用除法指令），4×展开循环。

**关键 NEON intrinsic**: `vdivq_f32`（AArch64）

**源码文件**:
- `tflite/kernels/div.cc` — `Register_DIV_NEON_OPT()`，编译期选择

### 2.5 SQUARED_DIFFERENCE

**NEON 优化实现**: 通过 `optimized_ops::MulElementwise` 间接使用 NEON 乘法路径，先 `vsubq_f32` 再 `vmulq_f32`。

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h`

---

## 3. 激活函数类

### 3.1 RELU / RELU6 / RELU_N1_TO_1 / LEAKY_RELU / HARD_SWISH / GELU / ELU

**优化方式**: 通过 `optimized_ops::` 函数间接使用 NEON。

**NEON 优化实现**:

#### HardSwish
- 4×展开循环，使用 `vld1q_f32` 加载，`vaddq_f32` 加常量 3.0，`vmaxq_f32` 钳位到 0，`vmulq_f32` 乘 1/6，再乘输入。
- 使用 `vfmaq_f32`（AArch64 FMA 指令）融合乘加。

**关键 NEON intrinsic**: `vmaxq_f32`, `vminq_f32`, `vaddq_f32`, `vmulq_f32`, `vfmaq_f32`

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `HardSwish()` (L6062)

#### LeakyRelu
**源码文件**: `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h`

#### PRelu
- Float32 标量广播版: `vmulq_f32` 乘 alpha，`vmaxq_f32` 钳位。
- Float32 逐元素版: 加载 input 和 alpha 两路向量，选择性乘加。

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `PReluScalarBroadcast()`, `PReluElementWise()` (L7401-L7500)

### 3.2 TANH / LOGISTIC (Sigmoid)

**优化方式**: 通过 `optimized_ops::` 函数使用 NEON。Uint8 版本使用 LUT（查找表）。

**NEON 优化实现**: Uint8 版本通过 `vqtbl4q_u8`（AArch64 表查找指令）进行 256 元素 LUT 查找，避免标量数学运算。

**关键 NEON intrinsic**: `vqtbl4q_u8`（AArch64）

**源码文件**:
- `tflite/kernels/activations.cc` — 算子注册，`Register_TANH_GENERIC_OPT()`, `Register_LOGISTIC_GENERIC_OPT()`
- `tflite/kernels/internal/optimized/optimized_ops.h` — `aarch64_lookup_vector()` (L247)

### 3.3 SOFTMAX / LOG_SOFTMAX

**优化方式**: 通过 `optimized_ops::Softmax()` 和 `optimized_ops::LogSoftmax()` 使用 NEON。

**源码文件**:
- `tflite/kernels/activations.cc` — `Register_SOFTMAX()`, `Register_LOG_SOFTMAX()`
- `tflite/kernels/internal/optimized/optimized_ops.h`

---

## 4. 卷积/池化类

### 4.1 CONV_2D

**优化方式**: 通过 `optimized_ops::Conv()` 和 RUY/gemmlowp GEMM 库间接使用 NEON。

**NEON 优化实现**:
- Float32: 通过 `cpu_backend_gemm::Gemm` 调用 RUY 库，RUY 内部有 AArch64 手写汇编 GEMM 内核。
- Uint8 量化: 通过 gemmlowp 库，内部使用 NEON `vmull_u8`/`vmlal_u8` 等进行量化矩阵乘法。
- Int8 量化: 通过 RUY 库，支持 Dot Product 扩展（`UDOT`/`SDOT` 指令）。
- Hybrid (Int8 权重 + Float 输入): 通过 `optimized_ops::HybridConv()` 和 `NeonMatrixBatchVectorMultiplyAccumulate`。

**关键 NEON intrinsic**: 由 RUY/gemmlowp 内部提供，包括 `vmlal_s8`, `vdotq_s32`, `sqrdmulh` 等

**源码文件**:
- `tflite/kernels/conv.cc` — `Register_CONV_2D()`，使用 `optimized_ops::Conv()`
- `tflite/kernels/internal/optimized/optimized_ops.h` — `Conv()` (L917-L1320)
- `tflite/kernels/internal/optimized/integer_ops/conv.h` — 整数量化卷积
- `tflite/kernels/internal/optimized/multithreaded_conv.h` — 多线程卷积
- `tflite/kernels/cpu_backend_gemm.h` — GEMM 后端选择（RUY/gemmlowp/Eigen）

### 4.2 DEPTHWISE_CONV_2D

**优化方式**: 编译期 NEON 选择 + Dot Product 运行时检测。

**NEON 优化实现**（这是 LiteRT 中 ARM 优化最深入的算子之一）:

#### 通用 NEON 路径 (`depthwiseconv_uint8.h`)
- 使用 `vmovl_u8` 扩展 uint8 到 int16，`vmlal_s16` 乘累加到 int32。
- 8 通道展开，同时处理 2 个输出像素。

#### 3×3 Dot Product 路径 (`depthwiseconv_3x3_filter_common.h`)
- **有 Dot Product 扩展**: 使用 `vdotq_lane_s32`（`SDOT` 指令），单指令完成 4 个 int8 点积。
- **无 Dot Product 扩展**: 软件模拟 `vdotq_s32`，用 `vmull_s8` + `vpaddlq_s16` + `vpaddq_s32` 组合。

#### AArch64 手写汇编路径 (`depthwiseconv_uint8_3x3_filter.h`)
- 完整的 AArch64 内联汇编内核，使用 32 个 NEON 寄存器（v0-v31）。
- 核心指令: `sqrdmulh`（饱和舍入加倍高位乘法）、`smlal2`（高位乘累加）、`sqrshl`（饱和舍入移位）。
- 支持多种 kernel 变体: 8 通道 stride 1、8 通道 stride 2、per-channel int8 等。

#### Int8 DepthwiseConv (`depthwise_conv.h` integer_ops)
- 专用的 int8 深度可分离卷积内核，使用 `vmull_s8`/`vmlal_s8`。
- 支持 depth_multiplier = 1/2/3/4/20 等多种配置。
- 使用 VTBL（表查找）指令进行 3 倍数据复制。

**关键 NEON intrinsic**: `vmlal_s16`, `vmull_s8`, `vdotq_lane_s32`, `sqrdmulh`（汇编）, `smlal2`（汇编）, `sqrshl`（汇编）, `vqtbl1_s8`（VTBL）

**源码文件**:
- `tflite/kernels/depthwise_conv.cc` — `Register_DEPTHWISE_CONVOLUTION_NEON_OPT()`, `Register_DEPTHWISE_CONV_2D_UINT8()`
- `tflite/kernels/internal/optimized/depthwiseconv_uint8.h` — 通用 NEON 量化 DepthwiseConv
- `tflite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h` — AArch64 手写汇编 3×3
- `tflite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h` — Dot Product 分支与软件模拟
- `tflite/kernels/internal/optimized/depthwiseconv_float.h` — 浮点 DepthwiseConv
- `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h` — int8 专用内核
- `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_3x3_filter.h` — int8 per-channel 3×3 AArch64 汇编

### 4.3 CONV_3D / CONV_3D_TRANSPOSE

**优化方式**: 通过 `optimized_ops::Conv3D()` 使用 NEON。

**源码文件**:
- `tflite/kernels/conv3d.cc`
- `tflite/kernels/conv3d_transpose.cc`
- `tflite/kernels/internal/optimized/optimized_ops.h`

### 4.4 TRANSPOSE_CONV

**优化方式**: 通过 `optimized_ops::TransposeConv()` 使用 NEON。

**源码文件**: `tflite/kernels/transpose_conv.cc`

### 4.5 AVERAGE_POOL_2D / MAX_POOL_2D

**优化方式**: `optimized_ops.h` 中有 NEON 专用路径。

**NEON 优化实现**:

#### AveragePool (Uint8)
- 16 通道展开，`vld1q_u8` 加载，`vmovl_u8` 扩展到 uint16 累加。
- 对于常见 filter_count（2, 4, 8），使用预计算除法常量和 `vqmovn_u16` 饱和窄化。

#### MaxPool (Uint8)
- 16 通道展开，`vmaxq_u8` 向量取最大值。
- 输出时使用 `vminq_u8`/`vmaxq_u8` 钳位到激活范围。

**关键 NEON intrinsic**: `vmaxq_u8`, `vminq_u8`, `vmovl_u8`, `vqmovn_u16`

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `AveragePool()` (L3099), `MaxPool()` (L3288)

### 4.6 L2_POOL_2D

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h`

---

## 5. 全连接层类

### 5.1 FULLY_CONNECTED

**优化方式**: 通过 RUY GEMM 库 + 专用 NEON 路径 + 4-bit NEON 内核。

**NEON 优化实现**:

#### 标准路径（通过 cpu_backend_gemm/RUY）
- Float32: RUY 库的 AArch64 GEMM 内核。
- Uint8/Int8: gemmlowp 或 RUY 的量化 GEMM 内核。

#### ShuffledFullyConnected（Uint8 Int16 专用）
- 使用 `veorq_u8` 对输入做 XOR 0x80（符号翻转）。
- 4×展开的权重/输入加载与乘累加。
- 使用 `vqrdmulh_s32`（SQRDMULH）进行量化缩放。

#### 4-bit FullyConnected（`4bit/neon_fully_connected.cc`）
- **AArch64 路径**: 使用 `vfmaq_f32`（融合乘加）、`vld1q_f32_x2`/`vld1q_f32_x4`（多寄存器加载）、`vcvtaq_s32_f32`（舍入转换）。
- **有 SDot 路径**: 调用 `NeonRunKernelSDot<>()`，使用 Dot Product 扩展。
- **ARM32 手写汇编路径**（`4bit/neon_fully_connected_arm32.cc`）: 完整的 ARM32 内联汇编，使用 `vmlal.s8`（int8 乘累加）、`vshr.u8`（右移解包 4-bit）、`vand`（mask 提取低 4-bit）、`vpaddl.s16`/`vpadd.i32`（归约求和）。

**关键 NEON intrinsic**: `vfmaq_f32`, `vld1q_f32_x4`, `vcvtaq_s32_f32`, `vmlal.s8`（汇编）, `vshr.u8`（汇编）, `vdotq_s32`

**源码文件**:
- `tflite/kernels/fully_connected.cc` — `Register_FULLY_CONNECTED()`
- `tflite/kernels/internal/optimized/optimized_ops.h` — `FullyConnected()`, `ShuffledFullyConnected()` (L278-L880)
- `tflite/kernels/internal/optimized/4bit/neon_fully_connected.cc` — AArch64 4-bit FC
- `tflite/kernels/internal/optimized/4bit/neon_fully_connected_arm32.cc` — ARM32 手写汇编 4-bit FC
- `tflite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h` — 4-bit 内核实现模板
- `tflite/kernels/internal/optimized/sparse_ops/fully_connected.h` — 稀疏 FC

---

## 6. 张量操作类

### 6.1 CONCATENATION

**优化方式**: 通过 `optimized_ops::Concatenation()` 使用 NEON。

**源码文件**: `tflite/kernels/concatenation.cc`

### 6.2 PACK / UNPACK

**NEON 优化实现**: `StoreValue()` 函数使用 `vqmovn_s32`→`vcombine_s16`→`vqmovn_s16`→`vcombine_s8`→`vst1q_s8` 进行高效的 int32→int8 饱和窄化与打包。

**关键 NEON intrinsic**: `vqmovn_s32`, `vcombine_s16`, `vqmovn_s16`, `vcombine_s8`

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `StoreValue()` (L3679)

### 6.3 DEPTH_TO_SPACE / SPACE_TO_DEPTH

**优化方式**: 通过 `optimized_ops::DepthToSpace()`/`SpaceToDepth()` 使用 NEON。

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` (L1320-L1395)

### 6.4 PAD / PADV2 / MIRROR_PAD

**优化方式**: 通过 `optimized_ops::Pad()` 使用 NEON。

**源码文件**:
- `tflite/kernels/pad.cc`
- `tflite/kernels/mirror_pad.cc`

### 6.5 SLICE / STRIDED_SLICE

**优化方式**: 通过 `optimized_ops::Slice()`/`StridedSlice()` 使用 NEON。

**源码文件**: `tflite/kernels/slice.cc`, `tflite/kernels/strided_slice.cc`

### 6.6 GATHER / GATHER_ND

**优化方式**: 通过 `optimized_ops::Gather()` 使用 NEON。

**源码文件**: `tflite/kernels/gather.cc`, `tflite/kernels/gather_nd.cc`

### 6.7 RESHAPE / EXPAND_DIMS / SQUEEZE / TRANSPOSE / TILE

**优化方式**: 通过对应的 `optimized_ops::` 函数使用 NEON。

**源码文件**: 各对应算子的 `.cc` 文件

### 6.8 CAST（Int4→Float 专用 NEON 优化）

**NEON 优化实现**:
- 使用 `vld1q_s8` 加载 16 字节（包含 32 个 int4 值）。
- `vshlq_n_s8`/`vshrq_n_s8` 左移 4 再右移 4，提取低 4 位。
- `vshrq_n_s8` 右移 4，提取高 4 位。
- `vzipq_s8` 交错合并低/高 4 位。
- `vmovl_s8` 扩展到 int16，再 `vcvtq_f32_s16` 转换到 float32。

**关键 NEON intrinsic**: `vld1q_s8`, `vshlq_n_s8`, `vshrq_n_s8`, `vzipq_s8`, `vmovl_s8`, `vcvtq_f32_s16`

**源码文件**: `tflite/kernels/cast.cc` — `castInt4ToFloat()` (L194)

### 6.9 BROADCAST_TO

**优化方式**: 通过 `optimized_ops::BroadcastTo()` 使用 NEON。

**源码文件**: `tflite/kernels/broadcast_to.cc`

---

## 7. 归约/比较/Arg 类

### 7.1 SUM / MEAN / REDUCE_MAX / REDUCE_MIN / REDUCE_PROD / REDUCE_ANY / REDUCE_ALL

**优化方式**: `optimized_ops::MeanImpl()` 中有 NEON 专用路径。

**NEON 优化实现**（以 Mean 为例，Uint8 版本）:
- 16 通道展开，`vld1q_u8` 加载，`vmovl_u8` 扩展到 uint16。
- `vaddq_u16` 累加到 `int32x4x4_t`。
- 最终除以 count 时使用 NEON 向量除法或移位。

**关键 NEON intrinsic**: `vaddq_u16`, `vmovl_u8`, `vst1q_u8`

**源码文件**:
- `tflite/kernels/internal/optimized/reduce.h` — `MeanImpl()` (L61)
- `tflite/kernels/internal/optimized/optimized_ops.h`

### 7.2 ARG_MAX / ARG_MIN

**NEON 优化实现**:
- **Float32**: `vld1q_f32` 加载 4 个值，`vcltq_f32`/`vcgtq_f32` 比较，`vbslq_f32` 条件选择更新最小/大值和索引。
- **AArch64 特化**: 使用 `vminvq_f32`/`vmaxvq_f32`（水平归约指令）直接求向量最小/大值，ARMv7 使用 `vpmin_f32`/`vpmax_f32` 配对归约。
- **Int8**: 使用 `vmaxvq_s8`（AArch64）或 `vpmax_s8`（ARMv7）进行水平最大值归约。
- **Uint8**: 使用 `vmaxvq_u8`（AArch64）或 `vpmax_u8`（ARMv7）。

**关键 NEON intrinsic**: `vminvq_f32`（AArch64）, `vmaxvq_f32`（AArch64）, `vpmin_f32`（ARMv7）, `vcltq_f32`, `vbslq_f32`, `vmaxvq_s8`（AArch64）, `vmaxvq_u8`（AArch64）

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `ArgMinVector()`, `ArgMaxVector()` (L7552-L7730)

### 7.3 MAXIMUM / MINIMUM

**NEON 优化实现**:
- Int8 逐元素: `vmaxq_s8`/`vminq_s8` 向量比较。
- Int8 标量广播: `vdupq_n_s8` 广播标量，`vmaxq_s8`/`vminq_s8`。
- Float32: 通过 `vmaxq_f32`/`vminq_f32`。

**关键 NEON intrinsic**: `vmaxq_s8`, `vminq_s8`, `vmaxq_f32`, `vminq_f32`

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `MaximumElementwise()`, `MaximumScalarBroadcast()`, `MinimumElementwise()`, `MinimumScalarBroadcast()` (L7235-L7310)

### 7.4 EQUAL / NOT_EQUAL / LESS / LESS_EQUAL / GREATER / GREATER_EQUAL

**优化方式**: 通过 `optimized_ops::` 比较函数使用 NEON。

**源码文件**: `tflite/kernels/comparisons.cc`

---

## 8. 量化/反量化类

### 8.1 QUANTIZE

**NEON 优化实现**:
- **Int32→Uint8**: `vld1q_s32` 加载，`SaturatingRoundingDoublingHighMul`（SQRDMULH 指令）缩放，`RoundingDivideByPOT` 移位，`vqmovn_s32`→`vqmovun_s16` 饱和窄化到 uint8。
- **Per-channel 版本**: `vld1q_s32` 加载 shift，`vminq_s32`/`vmaxq_s32` 分离左右移位，`vmulq_s32` 缩放。

**关键 NEON intrinsic**: `SQRDMULH`（通过 gemmlowp 封装）, `vqmovn_s32`, `vqmovun_s16`

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `Quantize()` (L5134-L5560)

### 8.2 DEQUANTIZE

**优化方式**: 编译期 NEON 选择 `Register_DEQUANTIZE_OPT()`。

**NEON 优化实现**:
- **Uint8→Float**: `vld1_u8` 加载，`vmovl_u8` 扩展，`vcvtq_f32_s32` 转换，`vfmaq_f32`（AArch64 FMA）或 `vmlaq_f32` 融合乘加。
- **Int8→Float**: 类似路径，使用 `vmovl_s8`。
- **Int16→Float**: `vld1_s16` 加载，`vmovl_s16` 扩展。

**关键 NEON intrinsic**: `vcvtq_f32_s32`, `vfmaq_f32`（AArch64）, `vmlaq_f32`

**源码文件**:
- `tflite/kernels/dequantize.cc` — `Register_DEQUANTIZE_OPT()`
- `tflite/kernels/internal/optimized/optimized_ops.h` — `Dequantize()` (L6380-L6490)

### 8.3 FAKE_QUANT

**源码文件**: `tflite/kernels/fake_quant.cc`

### 8.4 Requantize（Int8↔Uint8 互转）

**NEON 优化实现**:
- `vld1q_s8`/`vld1q_u8` 加载，`vmovl_s8`/`vmovl_u8` 扩展。
- `SaturatingRoundingDoublingHighMul` 缩放。
- `vqmovn_s32`→`vqmovn_s16` 饱和窄化。

**源码文件**: `tflite/kernels/internal/optimized/optimized_ops.h` — `Requantize<>()` (L5741-L6040)

---

## 9. RNN/LSTM 类

**优化方式**: 通过 `neon_tensor_utils` 提供 NEON 加速，供 LSTM/RNN/SVDF/GRUCell/BidirectionalSequenceLSTM 等算子使用。

### 9.1 NeonMatrixBatchVectorMultiplyAccumulate

**NEON 优化实现**:
- **Float 版本**: `vld1q_f32` 加载 4 个 float，`vmlaq_f32` 融合乘加。
- **Int8 版本**: `vld1q_s8` 加载 16 个 int8，`vmull_s8`/`vmlal_s8` 乘累加到 int16，`vpadalq_s16` 归约到 int32。
- **Int8 带缩放版本**: 使用 `SaturatingRoundingDoublingHighMul` 和 `RoundingDivideByPOT` 进行量化缩放。

### 9.2 NeonApplySigmoid / NeonApplyTanh

**NEON 优化实现**: 使用 NEON 向量化的定点 Sigmoid/Tanh 近似计算。

### 9.3 NeonApplyLayerNorm

**NEON 优化实现**: NEON 向量化的层归一化，使用 `vld1q_s16` 加载，`vmull_s16` 乘法，`vqrdmulh_s32` 缩放。

### 9.4 NeonCwiseMul / NeonCwiseAdd / NeonCwiseClipping

**NEON 优化实现**:
- `NeonCwiseMul`: int16 逐元素乘法 + 移位缩放。
- `NeonCwiseAdd`: int16 逐元素加法。
- `NeonCwiseClipping`: float/int16/int8 向量钳位。

### 9.5 NeonSparseMatrixBatchVectorMultiplyAccumulate

**NEON 优化实现**: 稀疏矩阵乘法，支持 1×4 和 1×16 稀疏格式。

### 9.6 NeonMatrixScalarMultiplyAccumulate

**NEON 优化实现**: 矩阵标量乘累加。

**关键 NEON intrinsic**: `vmlaq_f32`, `vmull_s8`, `vmlal_s8`, `vpadalq_s16`, `vqrdmulh_s32`

**源码文件**:
- `tflite/kernels/internal/optimized/neon_tensor_utils.h` — NEON 函数声明
- `tflite/kernels/internal/optimized/neon_tensor_utils.cc` — NEON 函数实现
- `tflite/kernels/internal/optimized/neon_tensor_utils_impl.h` — 内联实现
- `tflite/kernels/internal/tensor_utils.cc` — 分发到 NEON 或 Portable 版本
- `tflite/kernels/lstm.cc` — LSTM 算子（使用 neon_tensor_utils）
- `tflite/kernels/lstm_eval.cc` — LSTM 评估
- `tflite/kernels/rnn.cc` / `tflite/kernels/basic_rnn.cc` — RNN 算子
- `tflite/kernels/svdf.cc` — SVDF 算子
- `tflite/kernels/gru_cell.cc` — GRU 算子
- `tflite/kernels/bidirectional_sequence_lstm.cc` — 双向 LSTM

---

## 10. Resize 类

### 10.1 RESIZE_BILINEAR

**NEON 优化实现**:
- **Uint8 8× 放大专用路径**（`ResizeBilinear888Uint8`）: 针对 8 倍放大优化，使用 `vld1_u8` 加载，`vmovl_u8` 扩展，`vaddq_u16` 累加，`vqmovn_u16` 饱和窄化。
- **Float 通用路径**（`ResizeBilinearKernel`）: 32 通道展开，`vld1q_f32` 加载，`vmulq_f32` 缩放。
- **Float 2×2 版本**（`ResizeBilinearKernel2x2`）: 8 通道展开，双线性插值的 4 个角使用 `vld1q_f32`/`vmlaq_f32`。

**关键 NEON intrinsic**: `vld1_u8`, `vmovl_u8`, `vaddq_u16`, `vqmovn_u16`, `vmlaq_f32`

**源码文件**: `tflite/kernels/internal/optimized/resize_bilinear.h` (L41-L1400)

### 10.2 RESIZE_NEAREST_NEIGHBOR

**优化方式**: 编译期 NEON 选择。

**NEON 优化实现**: 通过 `optimized_ops::ResizeNearestNeighbor()` 中的 NEON 路径。

**源码文件**: `tflite/kernels/resize_nearest_neighbor.cc` — `Register_RESIZE_NEAREST_NEIGHBOR_NEON_OPT()`

---

## 11. 其他有 NEON 优化的算子

### 11.1 L2_NORMALIZATION

**NEON 优化实现**: Float32 版本使用 `vmulq_f32` 进行平方和累加，`vrsqrteq_f32` 近似倒数平方根（Newton-Raphson 迭代），`vmulq_f32` 归一化。

**源码文件**: `tflite/kernels/l2norm.cc`, `tflite/kernels/internal/optimized/optimized_ops.h` — `L2Normalization()` (L1410)

### 11.2 LOCAL_RESPONSE_NORMALIZATION

**源码文件**: `tflite/kernels/local_response_norm.cc`

### 11.3 FLOOR / CEIL / ROUND / NEG / ABS / SQRT / RSQRT / SQUARE / EXP / LOG / SIN / COS

**优化方式**: 通过 `optimized_ops::` 函数使用 NEON 向量化数学运算。

**源码文件**: 各对应算子的 `.cc` 文件

### 11.4 BATCH_MATMUL

**优化方式**: 通过 `optimized_ops::BatchMatMul()` 调用 `cpu_backend_gemm::Gemm`，使用 RUY 库的 NEON GEMM 内核。

**源码文件**:
- `tflite/kernels/batch_matmul.cc`
- `tflite/kernels/cpu_backend_gemm.h`

### 11.5 Cumsum

**源码文件**: `tflite/kernels/cumsum.cc`

### 11.6 ADD_N

**优化方式**: 通过 `optimized_ops::AddN()` 使用 NEON。

**源码文件**: `tflite/kernels/add_n.cc`

### 11.7 AudioSpectrogram / MFCC（Custom 算子）

**源码文件**: `tflite/kernels/audio_spectrogram.cc`, `tflite/kernels/mfcc.cc`

### 11.8 DetectionPostProcess（Custom 算子）

**源码文件**: `tflite/kernels/detection_postprocess.cc`

---

## 附录 A: 通过 RUY/gemmlowp 间接获得 NEON 优化的算子

以下算子通过调用 `cpu_backend_gemm::Gemm()` 间接使用 RUY 或 gemmlowp 库中的 ARM NEON 优化：

| 算子 | GEMM 库 | 文件 |
|------|---------|------|
| CONV_2D | RUY (default) | `tflite/kernels/conv.cc` |
| FULLY_CONNECTED | RUY (default) | `tflite/kernels/fully_connected.cc` |
| BATCH_MATMUL | RUY (default) | `tflite/kernels/batch_matmul.cc` |

RUY 库内部的 NEON 优化包括：
- AArch64 手写汇编 GEMM 内核
- Dot Product 扩展（SDOT/UDOT）支持
- 多核线程化矩阵分块
- gemmlowp 作为量化 fallback，也使用 NEON intrinsics

**源码文件**: `tflite/kernels/cpu_backend_gemm.h`, `tflite/kernels/cpu_backend_gemm_ruy.h`, `tflite/kernels/cpu_backend_gemm_gemmlowp.h`

---

## 附录 B: NEON intrinsic 使用频率最高的指令

| NEON Intrinsic | 对应 ARM 指令 | 使用频率 | 主要用途 |
|---------------|-------------|---------|---------|
| `vld1q_f32` | `LDR Q` | 极高 | 加载 4 个 float |
| `vst1q_f32` | `STR Q` | 极高 | 存储 4 个 float |
| `vaddq_f32` | `FADD` | 极高 | float32 向量加法 |
| `vmulq_f32` | `FMUL` | 极高 | float32 向量乘法 |
| `vmlaq_f32` | `FMLA` | 高 | float32 融合乘加 |
| `vmaxq_f32` / `vminq_f32` | `FMAX` / `FMIN` | 高 | 向量取最大/小值 |
| `vld1q_s8` | `LDR Q` (int8) | 高 | 加载 16 个 int8 |
| `vmull_s8` | `SMULL` | 高 | int8 乘法扩展到 int16 |
| `vmlal_s8` | `SMLAL` | 高 | int8 乘累加到 int16 |
| `vmlal_s16` | `SMLAL` | 高 | int16 乘累加到 int32 |
| `vqrdmulh_s32` | `SQRDMULH` | 高 | 量化缩放（饱和舍入加倍高位乘法） |
| `vfmaq_f32` | `FMLA` (AArch64) | 中 | 融合乘加（AArch64 专用） |
| `vdotq_lane_s32` | `SDOT` (DotProd) | 中 | int8 点积（Dot Product 扩展） |
| `vcvtaq_s32_f32` | `FRINTA+FCVTZS` (AArch64) | 中 | 舍入转换（AArch64 专用） |
| `vld1q_f32_x4` | `LDP Q×4` (AArch64) | 低 | 4 寄存器加载（AArch64 专用） |
| `vqmovn_s32` | `SQXTN` | 中 | int32→int16 饱和窄化 |
| `vqmovun_s16` | `UQXTN` | 中 | int16→uint8 饱和窄化 |
| `vmovl_u8` / `vmovl_s8` | `USXTL` / `SXTL` | 高 | 8-bit 扩展到 16-bit |
| `vdupq_n_f32` | `DUP` | 极高 | 广播标量到向量 |
| `sqrdmulh`（汇编） | `SQRDMULH` | 中 | AArch64 汇编中的量化缩放 |
| `smlal2`（汇编） | `SMLAL2` | 低 | AArch64 汇编高位乘累加 |
| `sqrshl`（汇编） | `SQRSHL` | 低 | AArch64 汇编饱和舍入移位 |
| `vmlal.s8`（汇编） | `VMLAL.S8` | 低 | ARM32 汇编 int8 乘累加 |

---

## 附录 C: 文件索引

| 文件路径 | 包含的 NEON 优化 |
|---------|----------------|
| `tflite/kernels/internal/optimized/neon_check.h` | USE_NEON 宏定义 |
| `tflite/kernels/internal/optimized/arch_check.h` | TFLITE_USE_NEON 宏定义 |
| `tflite/kernels/internal/optimized/cpu_check.cc` | Dot Product 运行时检测 |
| `tflite/kernels/internal/optimized/optimized_ops.h` | 主要优化算子集合（Add/Sub/Mul/Conv/Pool/FC/Quantize/Dequantize/ArgMax/ArgMin/MaxMin/PRelu/HardSwish/Resize/Concat 等） |
| `tflite/kernels/internal/optimized/integer_ops/add.h` | Int8/Int16 Add NEON 实现 |
| `tflite/kernels/internal/optimized/integer_ops/conv.h` | Int8 Conv NEON 实现 |
| `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h` | Int8 DepthwiseConv NEON 内核 |
| `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_3x3_filter.h` | Int8 per-channel 3×3 AArch64 汇编 |
| `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h` | LeakyRelu NEON 实现 |
| `tflite/kernels/internal/optimized/depthwiseconv_uint8.h` | Uint8 DepthwiseConv NEON 实现 |
| `tflite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h` | AArch64 手写汇编 3×3 DepthwiseConv |
| `tflite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h` | Dot Product 分支与软件模拟 |
| `tflite/kernels/internal/optimized/depthwiseconv_float.h` | Float DepthwiseConv NEON 实现 |
| `tflite/kernels/internal/optimized/resize_bilinear.h` | ResizeBilinear NEON 实现 |
| `tflite/kernels/internal/optimized/reduce.h` | Mean NEON 实现 |
| `tflite/kernels/internal/optimized/neon_tensor_utils.h` | RNN/LSTM NEON 函数声明 |
| `tflite/kernels/internal/optimized/neon_tensor_utils.cc` | RNN/LSTM NEON 函数实现 |
| `tflite/kernels/internal/optimized/neon_tensor_utils_impl.h` | RNN/LSTM NEON 内联实现 |
| `tflite/kernels/internal/optimized/4bit/neon_fully_connected.cc` | AArch64 4-bit FC NEON 实现 |
| `tflite/kernels/internal/optimized/4bit/neon_fully_connected_arm32.cc` | ARM32 手写汇编 4-bit FC |
| `tflite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h` | 4-bit FC NEON 模板实现 |
| `tflite/kernels/internal/common.h` | BiasAndClamp NEON 实现、MultiplyByQuantizedMultiplier4Rows NEON 实现 |
| `tflite/kernels/cast.cc` | Int4→Float NEON 优化 |
| `tflite/kernels/cpu_backend_gemm.h` | GEMM 后端选择（RUY/gemmlowp/Eigen） |
| `tflite/kernels/cpu_backend_gemm_custom_gemv.h` | 自定义 GEMV NEON 实现 |
| `tflite/kernels/add.cc` | ADD 编译期 NEON 选择 |
| `tflite/kernels/sub.cc` | SUB 编译期 NEON 选择 |
| `tflite/kernels/mul.cc` | MUL 编译期 NEON 选择 |
| `tflite/kernels/div.cc` | DIV 编译期 NEON 选择 |
| `tflite/kernels/depthwise_conv.cc` | DepthwiseConv 编译期 NEON 选择 |
| `tflite/kernels/dequantize.cc` | Dequantize 编译期 NEON 选择 |
| `tflite/kernels/resize_nearest_neighbor.cc` | ResizeNearestNeighbor 编译期 NEON 选择 |
