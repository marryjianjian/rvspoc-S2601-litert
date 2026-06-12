# RVV Model Benchmark Summary

Date: 2026-06-12

Benchmark environment:

- Build directory: `/home/plack/build/riscv-clang-litert`
- Model directory: `/home/plack/rvspoc_model`
- Runner: `qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,vlen=256,elen=64`
- Benchmark arguments: `--runs=50 --warmup_runs=5 --num_threads=1`
- Raw JSON/log output: `/tmp/rvv_model_benchmark_all_20260611_142316`
- MobileNet V2 FP32 retest output: `/tmp/rvv_mobilenet_v2_fp32_retest_20260612_142941`

These numbers compare the RVV-enabled benchmark binary with the no-RVV benchmark binary under the same qemu user-mode environment. They are useful for relative comparison in this setup, but they are not real hardware absolute performance numbers.

## FP32 Models

| Model | RVV avg ms | no-RVV avg ms | Avg speedup | RVV p95 ms | no-RVV p95 ms | RVV std ms | no-RVV std ms | RVV throughput inf/s | no-RVV throughput inf/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet Lite0 FP32 | 4287.99 | 42680.90 | 9.95x | 4309.27 | 42864.70 | 61.15 | 94.95 | 0.233 | 0.023 |
| MobileNet V1 FP32 | 4660.40 | 66863.10 | 14.35x | 4670.35 | 88640.70 | 5.51 | 8397.87 | 0.215 | 0.015 |
| MobileNet V2 FP32 | 3481.54 | 33975.90 | 9.76x | 3504.75 | 34125.00 | 15.65 | 74.27 | 0.287 | 0.029 |

| Model | RVV RSS after inference KB | no-RVV RSS after inference KB | RVV peak RSS KB | no-RVV peak RSS KB |
|---|---:|---:|---:|---:|
| EfficientNet Lite0 FP32 | 44940 | 42100 | 55452 | 52528 |
| MobileNet V1 FP32 | 39960 | 39348 | 50232 | 46388 |
| MobileNet V2 FP32 | 38444 | 37576 | 48708 | 48096 |

## INT8 Models

| Model | RVV avg ms | no-RVV avg ms | Avg speedup | RVV p95 ms | no-RVV p95 ms | RVV std ms | no-RVV std ms | RVV throughput inf/s | no-RVV throughput inf/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet Lite0 INT8 | 8056.57 | 42814.30 | 5.31x | 8103.89 | 43107.70 | 24.90 | 201.72 | 0.124 | 0.023 |
| MobileNet V1 Quant | 9494.01 | 60974.50 | 6.42x | 9557.47 | 61233.80 | 46.26 | 362.46 | 0.105 | 0.016 |
| MobileNet V2 Quant | 10512.40 | 33840.50 | 3.22x | 10528.40 | 33987.60 | 10.85 | 93.60 | 0.095 | 0.030 |

| Model | RVV RSS after inference KB | no-RVV RSS after inference KB | RVV peak RSS KB | no-RVV peak RSS KB |
|---|---:|---:|---:|---:|
| EfficientNet Lite0 INT8 | 27272 | 24228 | 38356 | 35320 |
| MobileNet V1 Quant | 25812 | 23052 | 34200 | 33516 |
| MobileNet V2 Quant | 25604 | 22620 | 35936 | 33088 |

## Observations

- All six models completed successfully. The EfficientNet INT8 no-RVV run did not reproduce the earlier abort.
- In this qemu user-mode setup, all models are faster with RVV enabled. FP32 speedups range from 9.76x to 14.35x by average latency after the MobileNet V2 FP32 retest. INT8 speedups range from 3.22x to 6.42x.
- RVV runs have smaller latency variance for every model in the final recorded data.
- The first MobileNet V2 FP32 run had a no-RVV outlier: one timed run reached 1826988.93 ms, which inflated `latency_avg_ms` to 106509.00 ms and `latency_std_ms` to 245784.00 ms. A same-parameter retest did not reproduce it: no-RVV `latency_avg_ms` was 33975.90 ms, `latency_p95_ms` was 34125.00 ms, and `latency_std_ms` was 74.27 ms. The retest result replaces the outlier run in the table.
- RVV peak RSS is slightly higher in all six runs. The absolute increase is small relative to qemu execution time and likely not the main performance factor.
