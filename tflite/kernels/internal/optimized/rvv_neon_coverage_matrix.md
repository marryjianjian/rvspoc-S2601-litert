# RVV Coverage Matrix for ARM NEON/SVE Optimized Paths

This document tracks the RISC-V RVV coverage for LiteRT CPU kernels that have
ARM NEON, ARM64, GEMMLOWP_NEON, or similar ARM-oriented optimized paths.

The matrix is intentionally conservative:

- `Done` means an RVV path is present and there is targeted correctness coverage
  in `rvv_ops_test.cc` or model-level coverage.
- `Partial` means only some data types, shapes, or fast paths are covered.
- `Missing` means an ARM optimized path was found but no equivalent RVV path was
  found in the inspected source.
- `Need audit` means the path is indirect, legacy, or hidden behind a library
  backend and needs a deeper source/performance audit before counting it toward
  the RVSPOC 90% requirement.

SVE-specific CPU kernel paths were not found in the inspected LiteRT CPU kernel
tree. The current ARM coverage source is primarily NEON, ARM64-specific code,
and GEMMLOWP_NEON.

## Source Areas

- `tflite/kernels/internal/optimized/optimized_ops.h`
- `tflite/kernels/internal/optimized/legacy_optimized_ops.h`
- `tflite/kernels/internal/optimized/depthwiseconv_*.h`
- `tflite/kernels/internal/optimized/integer_ops/*.h`
- `tflite/kernels/internal/optimized/neon_tensor_utils*.h`
- `tflite/kernels/internal/optimized/reduce.h`
- `tflite/kernels/internal/optimized/resize_bilinear.h`
- `tflite/kernels/cpu_backend_gemm*.h`
- builtin kernel registration files under `tflite/kernels/*.cc`

## High-Level Matrix

| Area | ARM optimized path | Current RVV/RISC-V path | Status | Correctness coverage | Remaining gap |
| --- | --- | --- | --- | --- | --- |
| Build selection | `USE_NEON`, `GEMMLOWP_NEON`, NEON registrations | `USE_RVV`, `TFLITE_RISCV_RVV`, `TFLITE_DISABLE_RISCV_RVV`, `TFLITE_RISCV_SCALAR_BASELINE` | Partial | RVV tests require RVV build | Need one documented build matrix for RVV, no-RVV scalar, and host reference |
| ADD float | NEON elementwise and scalar broadcast | RVV elementwise and scalar broadcast | Done | `AddElementwise*`, kernel entry tests | Registration still uses optimized/generic path rather than explicit RVV registration |
| ADD uint8 | NEON elementwise and scalar broadcast | RVV elementwise and scalar broadcast, non-single-rounding | Done | uint8 add tests | Single-rounding path remains scalar/reference |
| ADD int8 | NEON-style quantized path in optimized ops | RVV elementwise and scalar broadcast, non-single-rounding | Done | int8 add tests | Single-rounding path remains scalar/reference |
| ADD int16 | optimized int16 add | RVV elementwise | Done | int16 add test | Broadcast/specialized int16 variants need audit |
| SUB float | NEON elementwise | RVV elementwise | Done | float sub tests | Broadcast variants need explicit matrix row |
| SUB uint8/int8 | NEON quantized paths | RVV elementwise, non-single-rounding | Done | uint8/int8 sub tests | Single-rounding path remains scalar/reference |
| SUB int16 | optimized int16 path | RVV elementwise | Done | int16 sub test | Broadcast/specialized int16 variants need audit |
| MUL float | NEON elementwise and simple broadcast | RVV elementwise and simple broadcast | Done | float mul tests | None known for flat paths |
| MUL uint8/int8 | NEON quantized elementwise/simple broadcast | RVV elementwise/simple broadcast, non-single-rounding | Done | uint8/int8 mul tests | Single-rounding path remains scalar/reference |
| MUL int32 | NEON int32 elementwise | RVV int32 elementwise | Done | int32 mul test | None known for flat path |
| MUL int16 | Reference/F0 fixed-point path | No RVV replacement | Missing | Existing int16 reference comparison | Need decide whether to implement; previous analysis kept reference behavior |
| DIV float | NEON-like optimized arithmetic path | RVV float elementwise | Done | float div test | Broadcast variants need audit |
| DIV int32 | optimized int32 path | RVV int32 elementwise | Done | int32 div test | None known for flat path |
| Maximum/Minimum float | NEON float elementwise/broadcast | RVV elementwise and scalar broadcast | Done | maximum/minimum tests and kernel entry test | Complex broadcast shapes need model/perf audit |
| Maximum/Minimum int8 | NEON int8 elementwise/broadcast | RVV elementwise and scalar broadcast | Done | int8 maximum/minimum tests | Complex broadcast shapes need model/perf audit |
| SquaredDifference float/int32/int8 | optimized path with ARM coverage | RISC-V scalar and RVV flat paths | Done | squared difference tests | Broadcast/general shape coverage should be expanded |
| ReLU/ReluX float | NEON optimized activation | RVV vector clamp | Done | relu tests | Quantized variants covered separately |
| ReLU/ReluX int8/uint8/int16 | NEON-style quantized clamp | RVV vector clamp | Done | int8/uint8/int16 relu tests | None known for flat paths |
| LeakyRelu float | NEON optimized activation | RVV float | Done | float leaky relu test | None known |
| LeakyRelu int16 | integer optimized path | RVV int16 path in integer ops | Done | int16 leaky relu test | Single-rounding conditions need audit |
| PRelu float | optimized float path | RVV float scalar-broadcast and elementwise alpha | Done | PRelu tests | Quantized PRelu not counted yet |
| HardSwish float | NEON float path | RVV float | Done | hard swish float test | None known |
| HardSwish int8/uint8 | ARM NEON quantized path | RISC-V scalar and RVV quantized path | Done | quantized HardSwish tests | Depends on non-trivial fixed-point equivalence; keep high-priority regression coverage |
| Logistic float | Eigen/generic plus activation utility | RVV wrapper over vector load/store plus scalar math | Partial | logistic float test | Current RVV utility still computes scalar per lane; performance benefit limited |
| Tanh float | Eigen/generic plus activation utility | RVV wrapper over vector load/store plus scalar math | Partial | tanh float test | Current RVV utility still computes scalar per lane; performance benefit limited |
| Logistic/Tanh int16 | GEMMLOWP_NEON fixed-point paths | No direct RVV fixed-point replacement found | Missing | Indirect tensor utility tests | Need RVV fixed-point math or accept scalar fallback |
| ELU/GELU/Swish float | no clear NEON intrinsic path found | RVV/scalar utility coverage exists | Partial | ELU/GELU/Swish tests | These may not count toward NEON coverage but should stay in RVV feature matrix |
| Softmax float | optimized CPU path, some NEON helpers | RVV float softmax | Done | softmax float test | Need benchmark by real model shape |
| Softmax int8/uint8 LUT | ARM64 LUT path using table lookup helpers | uint8 LUT RVV and log-softmax quantized RVV; softmax LUT RVV needs audit | Partial | LUT tests, log-softmax quantized tests | Need separate TANH/LOGISTIC/Softmax LUT row-level confirmation |
| LogSoftmax float/int8/uint8 | optimized scalar plus ARM math assumptions | RISC-V scalar and RVV paths | Done | log-softmax tests | Uses math functions; real perf needs audit |
| Quantize int8/uint8/int16 | NEON affine quantize | RVV affine quantize | Done | affine quantize tests | Rounding mode must remain regression-tested across toolchains |
| Dequantize int8/uint8/int16 | NEON dequantize | RVV dequantize | Done | dequantize tests | None known |
| Requantize int8/uint8/int16/int32 combos | NEON requantize | RVV requantize, non-single-rounding | Done | requantize tests | Single-rounding path missing RVV |
| Conv2D int8 per-channel | optimized integer conv/GEMMLOWP_NEON | RVV dot-product path in `integer_ops/conv.h` plus scalar baseline | Partial | int8 conv per-channel test | GEMM/im2col/shape coverage and performance are not mature |
| Conv2D hybrid | optimized hybrid conv | RISC-V scalar and RVV hybrid conv | Partial | hybrid conv tests | Needs model-level performance and shape audit |
| Conv2D float | GEMM/backend based ARM path | RVV GEMM backend | Partial | float conv 1x1/GEMM tests | GEMM RVV is functional but not yet performance competitive/stable |
| FullyConnected int8 | NEON/gemmlowp/ruy paths | RVV int8 per-tensor/per-channel, plus GEMM backend | Partial | FC int8 tests | Sparse, 4-bit, shuffled FC not covered |
| FullyConnected float | CPU backend GEMM/ruy | RVV float GEMM backend | Partial | GEMM/FC tests | Needs real blocking/packing and VLEN tuning |
| ShuffledFullyConnected uint8->int16 | NEON-specific shuffled worker | No RVV shuffled worker found | Missing | None specific | Implement or document fallback if this path is still reachable |
| Sparse FullyConnected float/int8 | NEON tensor utils sparse paths | No direct RVV sparse tensor utility path found | Missing | None specific | Previously identified as lower priority but counts if NEON path is in scope |
| 4-bit FullyConnected | `optimized/4bit/neon_*` | No RVV 4-bit FC found | Missing | Existing 4-bit tests are NEON/reference oriented | Need decide priority; likely important for 90% coverage if enabled in build |
| DepthwiseConv float general | NEON row accumulation | RVV depth-multiplier-1 row accumulation | Partial | float depthwise tests | Only selected depth multiplier/shape path covered |
| DepthwiseConv uint8 general | NEON row accumulation and downquantize/store | RVV row accumulation, downquantize/store, scalar fallback | Partial | uint8 depthwise tests | Need more shapes, depth multipliers, activation/range coverage |
| DepthwiseConv uint8 3x3 filter | ARM64/NEON 3x3 filter kernels | RISC-V scalar and RVV 3x3 filter kernels | Done | 3x3 filter tests | Needs real performance on A210 |
| DepthwiseConv uint8 3x3 common | ARM64 common 3x3 fast path | RISC-V scalar and RVV common 3x3 path | Done | common 3x3 tests | Needs shape dispatch audit |
| DepthwiseConv int8 per-channel | integer_ops NEON depthwise | RISC-V scalar and RVV per-channel 3x3/general pieces | Partial | depthwise int8 tests | General per-channel shape matrix incomplete |
| DepthwiseConv hybrid 3x3 | ARM64 hybrid 3x3 filter files | No direct RVV hybrid 3x3 file-level equivalent found | Missing | None specific | Need inspect reachable hybrid depthwise dispatch and implement if used |
| AveragePool float/uint8/int8 | NEON/integer pool paths | RISC-V scalar and RVV average pool | Done | average pool tests | More window/stride/padding cases recommended |
| MaxPool float/uint8/int8 | NEON/integer pool paths | RVV max pool paths | Done | max pool tests | More window/stride/padding cases recommended |
| L2Pool float | optimized pool path | RVV coverage present | Done | L2 pool test | Need perf audit |
| Mean uint8/int8 HW | NEON integer mean | RVV mean HW paths | Done | uint8/int8 mean tests | More axes and reduction shapes needed |
| Reduce max/min/sum/mean float last-axis | NEON/reduce helpers | RVV last-axis reduction | Done | reduce tests | Non-last-axis generic reduce only partially covered |
| ArgMax/ArgMin float | NEON/ARM optimized compare loops | RVV value-first-index helper | Done | float argmax/argmin tests | Tie-breaking should stay covered |
| ArgMax/ArgMin int8 | NEON/ARM optimized compare loops | RISC-V scalar and RVV int8 helper | Done | int8 argmax/argmin tests | More axis shapes recommended |
| ResizeBilinear float | NEON resize kernels | RVV float resize branches | Done | resize bilinear tests | Quantized resize bilinear RVV needs audit |
| ResizeNearestNeighbor | NEON registration path | No explicit RVV registration/path found | Missing | none specific | Tensor copy style op; implement only if required by coverage count |
| Concatenation float | optimized copy path | RVV copy/tensor utility path in tests | Partial | concat float test | Kernel-level dispatch audit needed |
| Pad int8 | optimized copy/fill path | RVV copy/fill utility path in tests | Partial | pad int8 test | More dtypes and image-style paths need audit |
| Gather float | optimized copy path | RVV copy utility path in tests | Partial | gather float test | More dtypes/axis cases need audit |
| Table/LUT uint8/int8 | ARM64 LUT/table lookup helpers | RISC-V scalar and RVV LUT | Done | LUT tests | Need confirm LOGISTIC/TANH uint8 LUT builtin dispatch |
| Tensor utilities: dense matmul float | NEON tensor utils | RVV tests exist through RNN/SVDF helpers, but no broad rvv_tensor_utils dense API replacement | Partial | RNN/SVDF tests | Need one-to-one replacement for `neon_tensor_utils.h` entry points or document portable fallback |
| Tensor utilities: dense matmul int8/hybrid | NEON tensor utils | RISC-V/RVV RNN and hybrid conv pieces | Partial | RNN/SVDF/hybrid tests | Per-channel row-sum paths need audit |
| Tensor utilities: sparse matmul | NEON tensor utils sparse paths | No direct RVV sparse equivalent found | Missing | none specific | Required if sparse FC is counted |
| Tensor utilities: LSTM cwise mul/add/clip/sub/vector product | NEON tensor utils | RVV tests and helpers for LSTM cwise operations | Done | LSTM cwise tests | Confirm actual dispatch through `NEON_OR_PORTABLE` maps to RVV in all builds |
| Tensor utilities: quantize/normalization helpers | NEON tensor utils | Some RVV quantize/reduce coverage, no complete tensor_utils API mirror | Partial | quantize/reduce tests | Need API-level audit of every `NEON_OR_PORTABLE` wrapper |
| Conv3D / Conv3DTranspose | no clear NEON-specific source found | RISC-V scalar and RVV paths | Not counted | conv3d tests | Keep as extra RVV feature, not NEON coverage unless ARM fast path found |
| TransposeConv float/int | optimized GEMM/im2col path | Indirect RVV via GEMM only | Partial | no dedicated RVV coverage row found | Need direct test/benchmark if model uses it |
| NumericVerify/Cast | small NEON guarded paths | No RVV-specific equivalent found | Need audit | none specific | Likely low scoring impact but should be listed |

## Current Coverage Risk Summary

The project has broad RVV functional coverage for common elementwise,
activation, quantization, pooling, reduction, depthwise, convolution, fully
connected, recurrent, and tensor-manipulation paths. However, the RVSPOC 90%
requirement is stricter than "many RVV kernels exist": it asks for corresponding
RVV acceleration for ARM NEON/SVE optimized inference kernels.

High-risk gaps before claiming 90%:

1. `neon_tensor_utils.h` has many `NEON_OR_PORTABLE` entry points. Only some are
   clearly exercised by RVV tests. Sparse matrix-vector, quantization helper,
   and normalization wrappers need a one-to-one audit.
2. `optimized/4bit/neon_*` has no RVV equivalent.
3. Shuffled fully-connected and sparse fully-connected paths do not have clear
   RVV replacements.
4. Depthwise hybrid 3x3 and transitional depthwise paths need dispatch-level
   inspection.
5. Single-rounding quantized paths are often excluded from RVV by
   `!TFLITE_SINGLE_ROUNDING`.
6. Several tensor-copy style ops have RVV utility coverage but need kernel entry
   dispatch confirmation.
7. Performance evidence is still missing for most rows; functional coverage
   alone is not enough for scoring.

## Next Audit Steps

1. Turn this matrix into a mechanically checkable list keyed by source function.
2. Add source line references for each `Done` and `Missing` row.
3. Add a `coverage owner` column: direct RVV kernel, RVV through GEMM, RVV
   through tensor utility, or scalar fallback only.
4. Run the matrix against qemu `vlen=128/256/512` correctness tests.
5. Add A210 benchmark evidence for the rows used by MobileNetV1, MobileNetV2,
   and EfficientNet/EfficientDet-Lite0.
