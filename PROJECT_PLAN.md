# LiteRT RISC-V 架构适配与 RVV 优化 - 项目规划文档

## 1. 项目概述

本项目是 RISC-V 软件移植及优化挑战赛 2026 的赛题 S2601，要求将 LiteRT（原 TensorFlow Lite）完整移植至 RISC-V 平台，并利用 RVV 1.0 向量指令对推理内核进行性能加速。

- **赛题编号**: S2601
- **奖金**: 人民币 16000
- **截止时间**: 2026-08-31 (AoE)
- **参考仓库**: https://github.com/google-ai-edge/LiteRT
- **目标架构**: RISC-V（RV64GCV，支持 RVV 1.0）

---

## 2. 任务目标

### 2.1 核心目标
1. 将 LiteRT 完整移植至 RISC-V 平台
2. 对所有已有 ARM Neon/SVE 优化实现的推理内核完成对应的 RVV 向量化加速
3. 确保移植内核通过精度验证
4. 提供完整的性能基准测试数据

### 2.2 关键指标
- **覆盖率**: 已实现 ARM Neon/SVE 优化的算子中，至少 90% 需提供 RVV 或内联汇编加速版本
- **精度要求**: 
  - FP32 推理：Top-1 精度与 x86 参考差异 ≤ 0.1%
  - INT8 量化推理：Top-1 精度与 x86 参考差异 ≤ 1%
  - 算子级输出（FP32）：相对误差 ≤ 1e-5
  - 算子级输出（INT8）：差值 ≤ 1 LSB
- **性能要求**: 推理延迟 ≤ 110 ms

---

## 3. 需要完成的具体任务

### 3.1 RISC-V 平台移植

#### 3.1.1 构建系统修改
- [x] 修改 CMakeLists.txt，添加 RISC-V 架构支持 ✅ **已完成**
- [x] 添加 riscv64-unknown-linux-gnu 交叉编译工具链配置 ✅ **已完成**
- [x] 添加 RISC-V 架构检测逻辑 ✅ **已完成**
- [x] 配置 RVV 1.0 编译选项（`-march=rv64gcv`）✅ **已完成**

#### 3.1.2 架构适配
- [x] 添加 RISC-V 平台检测宏定义 ✅ **已完成**
- [x] 实现 `__riscv_vector` 条件编译隔离 ✅ **已完成**
- [x] 确保标量回退实现可用（无 RVV 环境下可编译运行）✅ **已完成**

### 3.2 RVV 1.0 向量化优化

#### 3.2.1 深度卷积（DepthwiseConv）优化
需要优化的文件：
- [ ] `tflite/kernels/internal/optimized/depthwiseconv_float.h` (约 192 处 ARM Neon 代码)
- [ ] `tflite/kernels/internal/optimized/depthwiseconv_uint8.h` (约 393 处)
- [ ] `tflite/kernels/internal/optimized/depthwiseconv_uint8_transitional.h` (约 610 处)
- [ ] `tflite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h` (约 161 处)
- [ ] `tflite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h` (约 8 处)

#### 3.2.2 优化操作（Optimized Ops）优化
- [ ] `tflite/kernels/internal/optimized/optimized_ops.h` (约 499 处 ARM Neon 代码)
- [ ] `tflite/kernels/internal/optimized/legacy_optimized_ops.h` (约 298 处)

#### 3.2.3 整数操作（Integer Ops）优化
需要优化的文件：
- [ ] `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h` (约 342 处)
- [ ] `tflite/kernels/internal/optimized/integer_ops/add.h` (约 45 处)
- [ ] `tflite/kernels/internal/optimized/integer_ops/mul.h` (约 25 处)
- [ ] `tflite/kernels/internal/optimized/integer_ops/pooling.h` (约 22 处)
- [ ] `tflite/kernels/internal/optimized/integer_ops/mean.h` (约 14 处)
- [ ] `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h` (约 11 处)
- [ ] `tflite/kernels/internal/optimized/integer_ops/lut.h` (约 6 处)

#### 3.2.4 张量工具（Tensor Utils）优化
- [ ] `tflite/kernels/internal/optimized/neon_tensor_utils.cc` (约 2806 行代码)
- [ ] `tflite/kernels/internal/optimized/neon_tensor_utils_impl.h`

#### 3.2.5 其他操作优化
- [ ] `tflite/kernels/internal/optimized/resize_bilinear.h` (约 171 处)
- [ ] `tflite/kernels/internal/optimized/reduce.h` (约 14 处)
- [ ] `tflite/kernels/internal/optimized/4bit/neon_fully_connected.cc` (约 67 处)
- [ ] `tflite/kernels/internal/optimized/4bit/neon_fully_connected_arm32.cc` (约 42 处)

### 3.3 RVV 实现技术要求

#### 3.3.1 编程接口
- 优先使用 **RVV Intrinsics**（`<riscv_vector.h>`）
- 可辅以内联汇编
- 必须包含标量（scalar）回退实现

#### 3.3.2 VLEN 自适应
- 支持不同 VLEN（128/256/512 bit）的自适应实现
- 利用 RVV 的 `vsetvl` 灵活向量长度特性
- 可实现自动向量长度调优（VLEN 探测 + 运行时分发）

#### 3.3.3 代码规范
- 语言：C/C++（C++17 或以上）
- 代码风格参考 Google C++ Style Guide
- 新增 RVV 内核须通过条件编译（`#ifdef __riscv_vector`）与通用代码隔离

### 3.4 精度验证

#### 3.4.1 验证标准
| 验证项 | 标准 |
|--------|------|
| FP32 推理（模型级） | Top-1 精度与 x86 参考差异 ≤ 0.1% |
| INT8 量化推理（模型级） | Top-1 精度与 x86 参考差异 ≤ 1% |
| 算子级输出（FP32） | 相对误差 ≤ 1e-5 |
| 算子级输出（INT8） | 差值 ≤ 1 LSB |

#### 3.4.2 验证方法
- [ ] 实现算子级精度测试框架
- [ ] 实现模型级精度测试框架
- [ ] 与 x86 或 ARM 标量版本进行对比验证

### 3.5 性能测试

#### 3.5.1 性能指标
| 验证项 | 要求 |
|--------|------|
| 推理延迟（Latency） | ≤ 110 ms，包含 avg、p50、p95 |
| 吞吐率（Throughput） | throughput = FPS = batch_size x 1000 / latency_ms |
| 稳定性 | 包含 std、p95 / avg 比值 |
| 内存占用 | Overall footprint 值 |
| 启动时间 | Model initialization 值 |

#### 3.5.2 测试模型
- [ ] MobileNetV1
- [ ] MobileNetV2
- [ ] EfficientDet-Lite0

#### 3.5.3 基准测试
- [ ] 与 ARM Neon 版本进行对比基准测试
- [ ] 提供完整 benchmark 数据

### 3.6 硬件适配

#### 3.6.1 模拟环境
- [ ] QEMU `virt` 机器（`-cpu rv64,v=true,vlen=256`）
- [ ] 不同 VLEN 配置测试（128/256/512 bit）

#### 3.6.2 真实硬件
- [ ] SG2044 开发板
- [ ] A210 开发板（组委会提供远程调试环境）
- [ ] 提供实测数据，附板卡型号及测试截图/日志

---

## 4. 执行步骤

### 阶段一：环境搭建与基础移植（第 1-2 周）
1. 搭建 RISC-V 交叉编译环境
   - 安装 riscv64-unknown-linux-gnu 工具链
   - 配置 QEMU RISC-V 模拟环境
2. 修改构建系统
   - 添加 RISC-V CMake 配置
   - 实现交叉编译支持
3. 基础平台移植
   - 确保 LiteRT 能在 RISC-V 上编译运行
   - 实现标量回退版本

### 阶段二：核心内核 RVV 优化（第 3-8 周）
1. 学习 RVV 1.0 规范和 Intrinsics
2. 逐个实现 RVV 优化内核
   - 优先级：深度卷积 > 优化操作 > 整数操作 > 张量工具 > 其他
3. 每个内核实现后进行单元测试

### 阶段三：精度验证与性能优化（第 9-12 周）
1. 实现精度验证框架
2. 进行算子级和模型级精度测试
3. 性能分析与优化
4. 不同 VLEN 适配

### 阶段四：硬件测试与文档编写（第 13-16 周）
1. 在真实 RISC-V 硬件上测试
2. 收集性能数据和测试截图
3. 编写验证文档
4. 准备提交材料

### 阶段五：提交与完善（第 17-20 周）
1. 提交 Pull Request
2. 根据评审反馈进行修改
3. 完善文档和代码

---

## 5. 技术难点与解决方案

### 5.1 RVV 与 ARM Neon 的差异
- **向量长度**: RVV 支持可变长度向量（VLEN），Neon 固定 128-bit
- **解决方案**: 使用 `vsetvl` 动态设置向量长度，实现 VLEN 无关的代码

### 5.2 内存对齐
- RVV 对内存对齐有要求
- **解决方案**: 使用 RVV 的加载/存储指令处理非对齐数据

### 5.3 性能优化
- 需要针对不同 VLEN 进行优化
- **解决方案**: 实现运行时 VLEN 检测和分发机制

---

## 6. 参考资料

| 资源 | 链接 |
|------|------|
| LiteRT 源码 | https://github.com/google-ai-edge/LiteRT |
| LiteRT 官方文档 | https://ai.google.dev/edge/litert |
| RISC-V RVV 1.0 规范 | https://github.com/riscv/riscv-v-spec |
| RVV Intrinsics 参考手册 | https://github.com/riscv-non-isa/rvv-intrinsic-doc |
| RISC-V GNU 工具链 | https://github.com/riscv-collab/riscv-gnu-toolchain |
| QEMU RISC-V 文档 | https://www.qemu.org/docs/master/system/target-riscv.html |
| LiteRT ARM Neon 内核参考 | https://github.com/google-ai-edge/LiteRT/tree/main/tflite/kernels/internal/optimized |
| MobileNetV1 模型 | https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md |

---

## 7. 提交要求

### 7.1 提交仓库
- https://github.com/rv2036/rvspoc-S2601-litert

### 7.2 提交内容
1. 完整的源码或二进制文件
2. 配置文件（若有）
3. 额外的库文件（若有）
4. 额外的补丁（若有）
5. 说明文件，包含：
   - 验证平台信息（OS 名称、版本和安装说明）
   - 依赖库信息及安装说明
   - 程序编译及安装步骤
   - 程序运行步骤
   - 程序运行结果
   - 其他验证所必要的信息

### 7.3 注意事项
- 遵循开源协议和版权规范（Apache 2.0）
- 禁止直接搬运已有第三方 RISC-V TFLite/LiteRT 移植代码
- 允许参考 LiteRT 原始 ARM Neon 内核实现逻辑
- 若使用 AI 辅助编写代码，需在提交报告中说明使用方式及占比

---

## 8. 风险评估

### 8.1 技术风险
- RVV 1.0 规范复杂，学习曲线陡峭
- 部分 ARM Neon 内核可能难以直接映射到 RVV
- 不同 VLEN 的性能差异需要仔细调优

### 8.2 时间风险
- 项目工作量大，需要合理分配时间
- 硬件测试可能受限于设备可用性

### 8.3 应对措施
- 提前学习 RVV 规范，参考现有 RVV 实现
- 分阶段实施，优先完成核心内核
- 利用 QEMU 进行早期测试，减少对硬件的依赖

---

## 9. 进度跟踪

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 环境搭建 | 待开始 | | |
| 构建系统修改 | ✅ 已完成 | 2026-05-01 | 完成RISC-V架构支持、工具链配置、架构检测 |
| 深度卷积优化 | 待开始 | | |
| 优化操作优化 | 待开始 | | |
| 整数操作优化 | 待开始 | | |
| 张量工具优化 | 待开始 | | |
| 其他操作优化 | 待开始 | | |
| 精度验证 | 待开始 | | |
| 性能测试 | 待开始 | | |
| 硬件测试 | 待开始 | | |
| 文档编写 | 待开始 | | |
| 提交准备 | 待开始 | | |

---

## 8. 完成记录

### 2026-05-01: 完成3.1节 RISC-V 平台移植工作

#### 已完成的具体任务：

**3.1.1 构建系统修改：**
1. **修改CMakeLists.txt** - 在`tflite/CMakeLists.txt`中添加了RISC-V架构支持：
   - 添加了RISC-V架构检测逻辑（第188-212行）
   - 添加了`TFLITE_RISCV64`和`TFLITE_ENABLE_RVV`选项
   - 配置了RVV 1.0编译选项（`-DTFLITE_RISCV_RVV`）
   - 在XNNPACK配置中添加了RISC-V架构处理

2. **创建交叉编译工具链文件** - 创建了`tflite/tools/cmake/riscv64-linux-gnu.cmake`：
   - 配置了riscv64-unknown-linux-gnu-gcc/g++编译器
   - 设置了`-march=rv64gcv -mabi=lp64d`编译选项
   - 配置了sysroot和搜索路径

**3.1.2 架构适配：**
1. **创建RVV检测头文件** - 创建了`tflite/kernels/internal/optimized/rvv_check.h`：
   - 检测`__riscv`和`__riscv_vector`宏
   - 定义`USE_RVV`宏
   - 提供`RVV_OR_PORTABLE`宏用于选择RVV或标量实现

2. **创建统一架构检测头文件** - 创建了`tflite/kernels/internal/optimized/arch_check.h`：
   - 统一检测ARM NEON、x86 SSE、RISC-V RVV架构
   - 定义`TFLITE_USE_RVV`宏
   - 提供统一的`NEON_OR_PORTABLE`和`RVV_OR_PORTABLE`宏

#### 技术实现要点：

1. **架构检测**：使用`__riscv`和`__riscv_vector`预处理器宏检测RISC-V架构和RVV支持
2. **条件编译**：通过`#ifdef __riscv_vector`实现RVV代码隔离
3. **标量回退**：确保在无RVV环境下使用标量实现
4. **VLEN自适应**：RVV的`vsetvl`指令天然支持不同VLEN
5. **编译选项**：通过`-march=rv64gcv`启用RVV 1.0支持

#### 文件修改清单：

1. `tflite/CMakeLists.txt` - 添加RISC-V架构支持
2. `tflite/tools/cmake/riscv64-linux-gnu.cmake` - 新建RISC-V工具链文件
3. `tflite/kernels/internal/optimized/rvv_check.h` - 新建RVV检测头文件
4. `tflite/kernels/internal/optimized/arch_check.h` - 新建统一架构检测头文件
5. `PROJECT_PLAN.md` - 更新任务状态

#### 下一步工作：

完成3.1节后，可以开始3.2节的RVV 1.0向量化优化工作，包括：
- 深度卷积（DepthwiseConv）优化
- 优化操作（Optimized Ops）优化
- 整数操作（Integer Ops）优化
- 张量工具（Tensor Utils）优化
- 其他操作优化

---

*文档创建日期: 2026-05-01*
*最后更新: 2026-05-01*
