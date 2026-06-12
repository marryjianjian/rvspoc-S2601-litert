# RVV Model Precision Summary

Date: 2026-06-12

This document records the current model-level RVV precision test method and the
latest real-image input dump results.

## Environment

- Source directory: `/Users/plack/code/LiteRT`
- Build directory: `/home/plack/build/riscv-clang-litert`
- Model directory: `/home/plack/rvspoc_model`
- Image directory on macOS: `/Users/plack/Downloads/rvv_input_images`
- Image directory on debian1: `/home/plack/rvspoc_model/images/bmp_224`
- Input dump directory: `/home/plack/rvspoc_model/input_dumps`
- Runner: `qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,vlen=256,elen=64`

The tested binaries are:

- RVV: `/home/plack/build/riscv-clang-litert/tflite_build/tensorflow-lite-rvv-model-precision-test`
- no-RVV/scalar: `/home/plack/build/riscv-clang-litert/tflite_build/tensorflow-lite-rvv-model-precision-test-norvv`

## Input Data

Six public COCO val2017 images were downloaded to
`/Users/plack/Downloads/rvv_input_images` and converted to 224x224 BMP files:

- `coco_000000000139`
- `coco_000000000285`
- `coco_000000000632`
- `coco_000000000724`
- `coco_000000000776`
- `coco_000000000785`

The BMP files were copied to debian1:

```sh
/home/plack/rvspoc_model/images/bmp_224
```

## Dump Generation

Dump generation is handled by:

```sh
tflite/kernels/internal/optimized/generate_rvv_input_dump.py
```

The script uses `flatc` and the TFLite schema to read each model input tensor's
metadata:

- input tensor type
- input tensor shape
- quantization scale
- quantization zero point
- optional min/max metadata

It then reads the 224x224 BMP images and writes raw input tensor bytes for each
sample. For FP32 inputs, image pixels are mapped into the model input real-value
range. For quantized inputs, the real values are quantized using the input
tensor's `scale` and `zero_point`.

The dump format is binary:

```text
char[8]  magic: "RVVINP01"
uint32   sample_count
repeat sample_count:
  uint32 input_count
  repeat input_count:
    int32    TfLite type enum value
    uint32   rank
    int32[]  shape
    uint64   byte_size
    uint8[]  raw tensor bytes
```

Generated dumps:

| Model | Dump | Samples | Input type |
|---|---|---:|---|
| EfficientNet Lite0 FP32 | `/home/plack/rvspoc_model/input_dumps/efficientnet-tflite-lite0-fp32.rvvinp` | 6 | FLOAT32 |
| EfficientNet Lite0 INT8 | `/home/plack/rvspoc_model/input_dumps/efficientnet-tflite-lite0-int8.rvvinp` | 6 | UINT8 |
| MobileNet V1 FP32 | `/home/plack/rvspoc_model/input_dumps/mobilenet_v1_1.0_224.rvvinp` | 6 | FLOAT32 |
| MobileNet V1 Quant | `/home/plack/rvspoc_model/input_dumps/mobilenet_v1_1.0_224_quant.rvvinp` | 6 | UINT8 |
| MobileNet V2 FP32 | `/home/plack/rvspoc_model/input_dumps/mobilenet_v2_1.0_224.rvvinp` | 6 | FLOAT32 |
| MobileNet V2 Quant | `/home/plack/rvspoc_model/input_dumps/mobilenet_v2_1.0_224_quant.rvvinp` | 6 | UINT8 |

## Precision Test Method

`rvv_model_precision_test` now supports:

```sh
--input_dump=/path/to/input.rvvinp
```

When this option is present:

1. The compare-mode parent passes the same dump path to reference, RVV, and
   scalar child runs.
2. Each run-mode child loads the dump.
3. For each sample, it checks that the dump input count matches the model input
   count.
4. For each input tensor, it checks `type`, `shape`, and `byte_size`.
5. If all checks pass, it copies the raw tensor bytes into the interpreter input
   buffer and invokes the model.

If `--input_dump` is not provided, the test keeps the older synthetic input path.

Current compare setup:

- reference binary: no-RVV binary
- RVV candidate: RVV binary
- scalar candidate: no-RVV binary
- both reference and candidate runs are executed under the same qemu RVV CPU
  settings

The scalar row is expected to match reference exactly because both use the same
no-RVV binary.

## Precision Criteria

The current model-level criteria are:

| Model kind | Top-1 criterion | Elementwise criterion |
|---|---:|---:|
| FP32 | top-1 mismatch <= 0.1% | max strict relative error <= `1e-5` |
| INT8/quantized | top-1 mismatch <= 1% | max LSB difference <= 1 |

Strict relative error is computed as:

```text
abs(actual - reference) / abs(reference)
```

If the reference value is zero, only an exact zero actual value has zero relative
error; otherwise the relative error is infinity.

## Current Results

The first complete real-image smoke run used `--samples=1`, so each model used
the first sample from its 6-sample dump.

Raw output directories:

- `/tmp/rvv_real_dump_precision_smoke_20260612_165024`
- `/tmp/rvv_real_dump_precision_remaining_20260612_165649`

| Model | Kind | RVV result | RVV top1 mismatch | RVV max abs | RVV max rel | RVV max LSB | Scalar result |
|---|---|---:|---:|---:|---:|---:|---:|
| EfficientNet Lite0 FP32 | FP32 | PASS | 0.000% | 0.000001401 | 0.000009676 | 0 | PASS |
| EfficientNet Lite0 INT8 | INT8 | PASS | 0.000% | 0.000000000 | 0.000000000 | 0 | PASS |
| MobileNet V1 FP32 | FP32 | FAIL | 0.000% | 0.000003695 | 0.000038881 | 0 | PASS |
| MobileNet V1 Quant | INT8 | PASS | 0.000% | 0.000000000 | 0.000000000 | 0 | PASS |
| MobileNet V2 FP32 | FP32 | FAIL | 0.000% | 0.000000454 | 0.000018661 | 0 | PASS |
| MobileNet V2 Quant | INT8 | PASS | 0.000% | 0.000000000 | 0.000000000 | 0 | PASS |

## Observations

- The real dump read path works for all six models.
- All quantized models passed with `max_lsb=0`.
- EfficientNet Lite0 FP32 passed the strict `1e-5` relative-error threshold.
- MobileNet V1 FP32 and MobileNet V2 FP32 preserved top-1 but exceeded the
  strict elementwise relative-error threshold.
- The scalar row passed for every model, as expected, because it compares the
  no-RVV binary against itself.
- A full 6-sample run was attempted first, but EfficientNet Lite0 FP32 no-RVV
  reference execution under qemu was too slow. It was stopped after several
  minutes while still in the first model's reference run. The current recorded
  data is therefore a 1-sample real-image smoke result, not the full 6-sample
  result.

## Next Steps

- Record the exact output index and element index for the FP32 maximum relative
  error so the failing MobileNet FP32 cases can be traced back to specific
  output values.
- Add an optional mode that stores only summary statistics instead of full output
  vectors, so 6-sample real-image runs are cheaper under qemu.
- Run the full 6-sample suite once the runtime cost is reduced or when real
  RISC-V hardware is available.
