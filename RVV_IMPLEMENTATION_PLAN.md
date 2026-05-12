# LiteRV RVV（RISC-V Vector Extension）支持实现计划

> 本文档规划了在 LiteRT 中实现 RISC-V RVV 1.0 向量扩展支持的完整方案。
> 采用渐进式策略：先实现最简单的算子验证整条链路，再逐步推广。

---

## 一、现状评估

### 已有的基础设施

| 文件 | 状态 | 内容 |
|------|------|------|
| `tflite/kernels/internal/optimized/arch_check.h` | ✅ 已完成 | `TFLITE_USE_RVV` 检测宏 + `NEON_OR_PORTABLE` → `Rvv##funcname` 重定向 |
| `tflite/kernels/internal/optimized/rvv_check.h` | ✅ 已完成 | `USE_RVV` 检测宏 + `RVV_OR_PORTABLE` 宏 |
| `tflite/CMakeLists.txt` | ✅ 部分完成 | RISC-V 64 检测、`TFLITE_ENABLE_RVV` 选项、`-DTFLITE_RISCV_RVV` 编译标志 |

### 缺失的部分

| 类别 | 状态 | 需要做什么 |
|------|------|----------|
| RVV 内核实现在 | ❌ 完全缺失 | 没有任何 `RvvXxx()` 函数实现 |
| Bazel 构建支持 | ❌ 缺失 | 只有 CMake 支持，没有 Bazel rules |
| RVV tensor_utils | ❌ 缺失 | 没有 `rvv_tensor_utils.h/cc/impl.h` |
| optimized_ops.h 中的 RVV 分支 | ❌ 缺失 | 所有 `#ifdef USE_NEON` 块都没有 `#elif defined(USE_RVV)` 对应 |
| 内核注册中的 RVV 选择 | ❌ 缺失 | `Register_ADD()` 等只有 NEON/Generic 两条路径 |
| 测试 | ❌ 缺失 | 没有任何 RVV 相关测试 |

### 关键架构模式

LiteRT 中 ARM NEON 优化有两种代码模式，RVV 需要分别适配：

**模式 A：宏分发**（`NEON_OR_PORTABLE` 宏）

```cpp
// tensor_utils.cc 中的调用方式
NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, args...);
// → ARM 上展开为: NeonMatrixBatchVectorMultiplyAccumulate(args...)
// → RVV 上展开为: RvvMatrixBatchVectorMultiplyAccumulate(args...)  ← arch_check.h 已支持
// → 其他平台:     PortableMatrixBatchVectorMultiplyAccumulate(args...)
```

**模式 B：内联条件编译**（`#ifdef USE_NEON` 块）

```cpp
// optimized_ops.h 中的模式
inline void AddElementwise(int size, ...) {
  int i = 0;
#ifdef USE_NEON
  // NEON intrinsics 代码
  for (; i <= size - 16; i += 16) { ... }
#elif defined(USE_RVV)   // ← 需要新增这个分支
  // RVV intrinsics 代码
#endif
  // 标量 fallback
  for (; i < size; i++) { ... }
}
```

**模式 C：编译期内核注册选择**

```cpp
// add.cc 中的模式
TfLiteRegistration* Register_ADD() {
#ifdef USE_NEON
  return Register_ADD_NEON_OPT();
#elif defined(USE_RVV)   // ← 需要新增这个分支
  return Register_ADD_RVV_OPT();
#else
  return Register_ADD_GENERIC_OPT();
#endif
}
```

---

## 二、总体实施策略

### 原则

1. **先通后优**：先让标量代码在 RISC-V 上正确运行，再添加 RVV 加速
2. **先简后繁**：从单个简单算子开始，验证全链路，再扩展
3. **复用框架**：利用已有的 `NEON_OR_PORTABLE` → `Rvv##funcname` 重定向机制
4. **保持兼容**：不破坏现有 ARM/x86 路径

### 分阶段路线图

```
阶段 1: 基础设施 + ADD Float32（验证全链路）
  ↓
阶段 2: 基础算术算子扩展（SUB/MUL/DIV Float32 + Int8 Add）
  ↓
阶段 3: 激活函数（Relu/HardSwish/Softmax）
  ↓
阶段 4: 池化与归约（MaxPool/AveragePool/ArgMax/Reduce）
  ↓
阶段 5: 量化操作（Quantize/Dequantize/Requantize）
  ↓
阶段 6: 卷积与全连接（Conv2D/FullyConnected via RVV GEMM）
  ↓
阶段 7: DepthwiseConv（RVV 专用深度可分离卷积）
  ↓
阶段 8: RNN/LSTM 系列（通过 rvv_tensor_utils）
  ↓
阶段 9: Resize/张量操作（ResizeBilinear/Concat/Pad/Gather）
  ↓
阶段 10: StableHLO 算子（按需逐步添加）
```

---

## 三、阶段 1：基础设施 + ADD Float32（详细实施方案）

### 目标

实现 `AddElementwise` 的 RVV 浮点版本，完成从编译到运行的全链路验证。

### 涉及的 RVV 指令

| RVV Intrinsic | 用途 | 对应的 NEON |
|---|---|---|
| `__riscv_vsetvl_e32m4(avl)` | 设置向量长度 | 无（NEON 固定 4 元素） |
| `__riscv_vle32_v_f32m4(ptr, vl)` | 加载 float32×N | `vld1q_f32` |
| `__riscv_vse32_v_f32m4(ptr, v, vl)` | 存储 float32×N | `vst1q_f32` |
| `__riscv_vfadd_vv_f32m4(a, b, vl)` | 向量加法 | `vaddq_f32` |
| `__riscv_vfmax_vf_f32m4(a, scalar, vl)` | 向量取最大值 | `vmaxq_f32` |
| `__riscv_vfmin_vf_f32m4(a, scalar, vl)` | 向量取最小值 | `vminq_f32` |

> 注：`e32m4` 表示 float32 元素、LMUL=4。对于 VLEN=128 的实现，一次处理 16 个 float32 元素，与 NEON 的 4×4 展开等效。

### 步骤 1.1：修复 `arch_check.h` 与 `rvv_check.h` 的不一致

**问题**：`arch_check.h` 要求定义 `TFLITE_RISCV_RVV`（由 CMake 设置），而 `rvv_check.h` 只检查 `__riscv_vector`（编译器内置宏），两者检测条件不同。

**文件**：`tflite/kernels/internal/optimized/rvv_check.h`

**修改内容**：统一检测条件，与 `arch_check.h` 保持一致。

```cpp
// 修改前：
#if defined(__riscv) && defined(__riscv_vector)
#define USE_RVV

// 修改后：
#if defined(__riscv) && defined(__riscv_vector) && defined(TFLITE_RISCV_RVV)
#define USE_RVV
```

### 步骤 1.2：在 `optimized_ops.h` 中添加 RVV 分支

**文件**：`tflite/kernels/internal/optimized/optimized_ops.h`

**修改位置**：`AddElementwise()` 函数（约 L1476-L1520）

**修改内容**：在 `#ifdef USE_NEON` 块之后添加 `#elif defined(USE_RVV)` 块。

```cpp
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const float* input1_data, const float* input2_data,
                           float* output_data) {
  int i = 0;

#ifdef USE_NEON
  // ... 现有 NEON 代码不变 ...

#elif defined(USE_RVV)
  // ---- RVV 实现 ----
  const float act_min = params.float_activation_min;
  const float act_max = params.float_activation_max;
  size_t vl;
  for (; i < size; ) {
    vl = __riscv_vsetvl_e32m4(size - i);
    vfloat32m4_t a = __riscv_vle32_v_f32m4(input1_data + i, vl);
    vfloat32m4_t b = __riscv_vle32_v_f32m4(input2_data + i, vl);
    vfloat32m4_t x = __riscv_vfadd_vv_f32m4(a, b, vl);
    x = __riscv_vfmax_vf_f32m4(x, act_min, vl);
    x = __riscv_vfmin_vf_f32m4(x, act_max, vl);
    __riscv_vse32_v_f32m4(output_data + i, x, vl);
    i += vl;
  }
#endif  // USE_NEON / USE_RVV

  // 标量 fallback（NEON 和 RVV 的残余元素都走这里）
  for (; i < size; i++) {
    auto x = input1_data[i] + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(
        x, params.float_activation_min, params.float_activation_max);
  }
}
```

**关键设计点**：
- RVV 使用 `vsetvl` 动态设置向量长度，天然处理了尾部元素（不像 NEON 需要手动处理余数）
- `e32m4` 表示 float32 + LMUL=4，对于 VLEN=128 一次处理 4×128/32=16 个元素
- `vfmax_vf` / `vfmin_vf` 的 `vf` 后缀表示向量×标量操作，标量直接作为参数传入

### 步骤 1.3：验证 `add.cc` 的内核注册

**文件**：`tflite/kernels/add.cc`

**分析**：当前 `Register_ADD()` 的逻辑是：

```cpp
TfLiteRegistration* Register_ADD() {
#ifdef USE_NEON
  return Register_ADD_NEON_OPT();    // → Eval<kNeonOptimized>
#else
  return Register_ADD_GENERIC_OPT(); // → Eval<kGenericOptimized>
#endif
}
```

`Eval<kGenericOptimized>` 最终调用 `optimized_ops::Add()` → `AddElementwise()`。
由于我们在步骤 1.2 中已经给 `AddElementwise()` 添加了 RVV 分支，
所以 **不需要修改 `add.cc`**——RVV 会通过 `Register_ADD_GENERIC_OPT()` 路径自动获得加速。

**但需要注意**：如果未来 RVV 需要特殊的 `Prepare` 逻辑（不同的 buffer 分配策略），
则需要添加 `Register_ADD_RVV_OPT()` 和 `kRvvOptimized` 内核类型。

### 步骤 1.4：添加 Bazel 构建支持

**文件**：`tflite/kernels/internal/optimized/BUILD`（新建或修改）

需要添加 RVV 源文件的 Bazel 规则：

```python
# RVV 优化的 tensor_utils
cc_library(
    name = "rvv_tensor_utils",
    srcs = ["rvv_tensor_utils.cc"],
    hdrs = ["rvv_tensor_utils.h", "rvv_tensor_utils_impl.h"],
    copts = [
        "-DTFLITE_RISCV_RVV",
        "-march=rv64gcv",
    ],
    deps = [
        "//tflite/kernels:cpu_backend_context",
        "//tflite/kernels/internal/optimized:rvv_check",
    ],
)
```

**文件**：`tflite/CMakeLists.txt`

添加 RVV 源文件到构建：

```cmake
if(TFLITE_ENABLE_RVV)
  list(APPEND TFLITE_OPTIMIZED_SRCS
    "tflite/kernels/internal/optimized/rvv_tensor_utils.cc"
  )
endif()
```

### 步骤 1.5：添加冒烟测试

**文件**：`tflite/kernels/add_test.cc`

添加 RVV 相关测试（或在现有测试上运行确认）：

```cpp
// 测试 ADD float32 在 RVV 上的正确性
// 可以通过 QEMU 用户模式模拟 RISC-V 运行
// 或在真实 RISC-V 硬件上运行
```

### 阶段 1 交付物

| 交付物 | 文件 | 说明 |
|-------|------|------|
| 修复 RVV 检测一致性 | `rvv_check.h` | 统一 `TFLITE_RISCV_RVV` 检测 |
| ADD Float32 RVV 实现 | `optimized_ops.h` | `AddElementwise` 新增 `#elif defined(USE_RVV)` 块 |
| Bazel 构建支持 | `BUILD` / `CMakeLists.txt` | RVV 编译标志和源文件 |
| 测试验证 | 现有 add_test | 在 RISC-V 上运行通过 |

### 阶段 1 预计工作量

| 任务 | 工时 |
|------|------|
| 修复 rvv_check.h | 0.5 天 |
| 实现 AddElementwise RVV | 1 天 |
| Bazel/CMake 构建适配 | 1 天 |
| QEMU 环境搭建 + 测试 | 2 天 |
| 代码审查 + 调试 | 1 天 |
| **合计** | **~5.5 天** |

---

## 四、阶段 2：基础算术算子扩展

### 目标

扩展到 SUB、MUL、DIV Float32 以及 Int8 Add。

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `optimized_ops.h` | `SubElementwise` / `MulElementwise` / `MulSimpleBroadcast` 添加 RVV 分支 |
| `integer_ops/add.h` | `AddElementwiseInt8` 添加 RVV 分支 |
| `sub.cc` | 评估是否需要 `Register_SUB_RVV_OPT()` |
| `mul.cc` | 评估是否需要 `Register_MUL_RVV_OPT()` |
| `div.cc` | 评估是否需要 `Register_DIV_RVV_OPT()` |

### 新增 RVV 指令

| RVV Intrinsic | 用途 |
|---|---|
| `__riscv_vfsub_vv_f32m4(a, b, vl)` | float32 向量减法 |
| `__riscv_vfmul_vv_f32m4(a, b, vl)` | float32 向量乘法 |
| `__riscv_vfdiv_vv_f32m4(a, b, vl)` | float32 向量除法 |
| `__riscv_vle8_v_i8m1(ptr, vl)` | 加载 int8 向量 |
| `__riscv_vadd_vv_i8m1(a, b, vl)` | int8 向量加法 |
| `__riscv_vwadd_vv_i16m2(a, b, vl)` | 宽化加法 int8→int16 |
| `__riscv_vzext_vf2_i16m2(a, vl)` | 零扩展 int8→int16 |
| `__riscv_vsext_vf2_i16m2(a, vl)` | 符号扩展 int8→int16 |

### 预计工作量：~5 天

---

## 五、阶段 3：激活函数

### 目标

实现 Relu、HardSwish、LeakyRelu、Softmax、PRelu 的 RVV 加速。

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `optimized_ops.h` | `HardSwish` / `PReluScalarBroadcast` / `PReluElementWise` 添加 RVV 分支 |
| `integer_ops/leaky_relu.h` | 整个文件添加 RVV 路径 |
| `activations.cc` | 评估是否需要 `Register_*_RVV_OPT()` |

### 新增 RVV 指令

| RVV Intrinsic | 用途 |
|---|---|
| `__riscv_vfmacc_vv_f32m4(acc, a, b, vl)` | 融合乘加（HardSwish 用） |
| `__riscv_vfmerge_vfm(a, scalar, mask, vl)` | 条件合并（PRelu 用） |

### 预计工作量：~4 天

---

## 六、阶段 4：池化与归约

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `optimized_ops.h` | `AveragePool` / `MaxPool` (uint8) / `ArgMinVector` / `ArgMaxVector` 添加 RVV 分支 |
| `reduce.h` | `MeanImpl` 添加 RVV 分支 |

### 新增 RVV 指令

| RVV Intrinsic | 用途 |
|---|---|
| `__riscv_vmaxu_vv_u8m1(a, b, vl)` | uint8 取最大值（MaxPool） |
| `__riscv_vzext_vf2_u16m2(a, vl)` | uint8→uint16 零扩展（AveragePool 累加） |
| `__riscv_vfredusum_vs_f32m4_f32m1(v, scalar, vl)` | float32 无序求和归约 |
| `__riscv_vfredmin_vs_f32m4_f32m1(v, scalar, vl)` | float32 最小值归约 |
| `__riscv_vfredmax_vs_f32m4_f32m1(v, scalar, vl)` | float32 最大值归约 |

### 预计工作量：~5 天

---

## 七、阶段 5：量化操作

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `optimized_ops.h` | `Quantize` / `Dequantize` / `Requantize` 添加 RVV 分支 |
| `dequantize.cc` | 评估是否需要 `Register_DEQUANTIZE_RVV_OPT()` |

### 新增 RVV 指令

| RVV Intrinsic | 用途 |
|---|---|
| `__riscv_vfcvt_f_x_v_f32m4(int_vec, vl)` | int32→float32 转换 |
| `__riscv_vfcvt_x_f_v_i32m4(float_vec, vl)` | float32→int32 转换 |
| `__riscv_vnclip_wv_i8m1(wide, narrow, vl)` | 窄化裁剪（带饱和） |
| `__riscv_vsmul_vv_i32m4(a, b, vl)` | 饱和舍入定点乘法 |

### 预计工作量：~5 天

---

## 八、阶段 6-10：高级算子

| 阶段 | 算子 | 关键 RVV 指令 | 预计工时 |
|------|------|-------------|---------|
| 6 | Conv2D / FullyConnected | `vfmacc_vv` (GEMM 内核) | ~10 天 |
| 7 | DepthwiseConv | `vwmacc_vv` + 手工循环优化 | ~8 天 |
| 8 | LSTM / RNN / SVDF | 通过 `rvv_tensor_utils` | ~8 天 |
| 9 | Resize / Concat / Pad / Gather | `vrgather_vv` + scatter/gather load | ~6 天 |
| 10 | StableHLO 按需 | 视具体需求 | 按需评估 |

---

## 九、需要新建的文件

| 文件路径 | 说明 |
|---------|------|
| `tflite/kernels/internal/optimized/rvv_tensor_utils.h` | RVV tensor_utils 头文件，声明 `RvvXxx()` 函数 |
| `tflite/kernels/internal/optimized/rvv_tensor_utils.cc` | RVV tensor_utils 分发文件（类似 `neon_tensor_utils.h` 的角色） |
| `tflite/kernels/internal/optimized/rvv_tensor_utils_impl.h` | RVV tensor_utils 内联实现（类似 `neon_tensor_utils_impl.h`） |
| `tflite/kernels/internal/optimized/rvv_ops.h` | RVV 公共辅助函数（如 RVV 版 `MultiplyByQuantizedMultiplier`） |

---

## 十、需要修改的文件清单

### 基础设施层

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `tflite/kernels/internal/optimized/rvv_check.h` | 修改 | 统一 `TFLITE_RISCV_RVV` 检测条件 |
| `tflite/kernels/internal/optimized/BUILD` | 修改 | 添加 RVV 源文件的 Bazel 规则 |
| `tflite/CMakeLists.txt` | 修改 | 添加 RVV 源文件到构建 |
| `tflite/kernels/internal/tensor_utils.cc` | 修改 | 添加 `#elif defined(USE_RVV)` include 分支 |

### 优化算子层（按阶段递增修改）

| 文件 | 修改类型 | 涉及阶段 |
|------|---------|---------|
| `tflite/kernels/internal/optimized/optimized_ops.h` | 添加 RVV 分支 | 阶段 1-9（核心文件） |
| `tflite/kernels/internal/optimized/integer_ops/add.h` | 添加 RVV 分支 | 阶段 2 |
| `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h` | 添加 RVV 分支 | 阶段 3 |
| `tflite/kernels/internal/optimized/reduce.h` | 添加 RVV 分支 | 阶段 4 |
| `tflite/kernels/internal/optimized/resize_bilinear.h` | 添加 RVV 分支 | 阶段 9 |

### 内核注册层（按需修改）

| 文件 | 修改类型 | 涉及阶段 |
|------|---------|---------|
| `tflite/kernels/add.cc` | 可能需要添加 `Register_ADD_RVV_OPT()` | 阶段 1（如果 generic 路径不够） |
| `tflite/kernels/sub.cc` | 同上 | 阶段 2 |
| `tflite/kernels/mul.cc` | 同上 | 阶段 2 |
| `tflite/kernels/div.cc` | 同上 | 阶段 2 |
| `tflite/kernels/depthwise_conv.cc` | 需要添加 RVV 内核 | 阶段 7 |
| `tflite/kernels/fully_connected.cc` | 需要添加 RVV 内核 | 阶段 6 |

---

## 十一、测试策略

### 11.1 测试环境

| 环境 | 用途 | 工具 |
|------|------|------|
| QEMU 用户模式 | CI 自动化测试 | `qemu-riscv64` |
| QEMU 系统模式 | 完整系统测试 | `qemu-system-riscv64` |
| 真实硬件 | 性能验证 | SiFive HiFive Unmatched / StarFive VisionFive 2 |

### 11.2 测试方法

1. **单元测试**：复用现有 `add_test.cc` 等，在 RISC-V 环境中运行
2. **正确性对比**：RVV 结果与 Portable（标量）结果逐 bit 对比
3. **性能基准**：RVV vs Portable 的吞吐量对比（元素/周期）
4. **端到端测试**：使用 LiteRT 加载 MobileNet 等模型，在 RISC-V 上运行推理

### 11.3 QEMU 交叉编译测试命令

```bash
# 交叉编译
cmake -B build-rv64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-linux-gnu.cmake \
  -DTFLITE_ENABLE_RVV=ON \
  -DCMAKE_C_FLAGS="-march=rv64gcv" \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv"

cmake --build build-rv64 --target add_test

# QEMU 运行
qemu-riscv64 -L /usr/riscv64-linux-gnu build-rv64/tflite/kernels/add_test
```

---

## 十二、RVV 编程要点速查

### 12.1 与 NEON 的核心差异

```cpp
// NEON: 固定宽度，手动处理余数
for (; i <= size - 16; i += 16) {
    float32x4_t a = vld1q_f32(ptr + i);      // 固定加载 4 个 float
    // ... 处理 4 个元素 ...
}
for (; i < size; i++) { /* 标量处理余数 */ }

// RVV: 动态宽度，自动处理余数
size_t vl;
for (; i < size; ) {
    vl = __riscv_vsetvl_e32m4(size - i);     // 自动计算可处理的元素数
    vfloat32m4_t a = __riscv_vle32_v_f32m4(ptr + i, vl); // 加载 vl 个 float
    // ... 处理 vl 个元素 ...
    i += vl;  // vl 在最后一次迭代自动变小，无需标量余数处理
}
```

### 12.2 LMUL 选择指南

| LMUL | 寄存器分组 | 适用场景 |
|------|----------|---------|
| `m1` | 1 个寄存器 | 寄存器压力大、需要更多寄存器组 |
| `m2` | 2 个寄存器 | 一般用途 |
| `m4` | 4 个寄存器 | 高吞吐量（类似 NEON 4×展开） |
| `m8` | 8 个寄存器 | 最大吞吐量，寄存器组最少 |

### 12.3 类型后缀命名规则

```
__riscv_<op>_<type_suffix>_<result_type><lmul>(args, vl)
```

| 后缀 | 含义 | 示例 |
|------|------|------|
| `vv` | 向量 op 向量 | `vfadd_vv_f32m4(a, b, vl)` |
| `vf` | 向量 op float 标量 | `vfadd_vf_f32m4(a, 3.0f, vl)` |
| `vi` | 向量 op 立即数 | `vadd_vi_i32m4(a, 1, vl)` |
| `vx` | 向量 op 整数标量寄存器 | `vadd_vx_i32m4(a, x[rs1], vl)` |

---

## 十三、风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| RVV 编译器支持不成熟 | 编译错误、代码生成质量差 | 使用 GCC 14+ 或 Clang 17+，定期更新工具链 |
| 缺少真实硬件测试 | QEMU 模拟可能不完全反映真实性能 | 尽早获取 RISC-V 开发板（SiFive/StarFive） |
| `optimized_ops.h` 文件过大 | 添加 RVV 分支后文件更难维护 | 考虑将 RVV 实现拆分到独立文件，用 `#include` 引入 |
| RVV 与 NEON 的精度差异 | 量化计算可能产生微小差异 | 设置合理的容差范围进行测试 |
| XNNPACK 对 RISC-V 的支持 | XNNPACK 可能尚未完整支持 RVV | 依赖 LiteRT 内置 RVV 路径，不依赖 XNNPACK |

---

## 十四、里程碑时间线

```
Week 1-2:   阶段 1 — 基础设施 + ADD Float32 全链路验证
Week 3-4:   阶段 2 — SUB/MUL/DIV + Int8 Add
Week 5:     阶段 3 — 激活函数
Week 6-7:   阶段 4 — 池化与归约
Week 8-9:   阶段 5 — 量化操作
Week 10-12: 阶段 6 — Conv2D / FullyConnected GEMM
Week 13-14: 阶段 7 — DepthwiseConv
Week 15-16: 阶段 8 — RNN/LSTM
Week 17-18: 阶段 9 — Resize/张量操作
Week 19+:   阶段 10 — StableHLO 按需
```

**总计约 18 周（4.5 个月）完成核心算子覆盖，覆盖 LiteRT 最常用的 ~50 个算子。**
