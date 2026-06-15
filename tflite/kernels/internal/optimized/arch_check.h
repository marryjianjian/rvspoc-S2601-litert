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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_ARCH_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_ARCH_CHECK_H_

// Architecture detection for optimized kernels
// This header provides a unified way to detect and use SIMD/vector extensions

// ARM NEON detection
#if (defined(__ARM_NEON__) || defined(__ARM_NEON)) && \
    !defined(TFLITE_DISABLE_ARM_NEON)
#define TFLITE_USE_NEON
#include <arm_neon.h>
#endif

// x86 SSE detection
#if defined __GNUC__ && defined __SSE4_1__ && !defined TF_LITE_DISABLE_X86_NEON
#define TFLITE_USE_NEON
#include "NEON_2_SSE.h"
#endif

// RISC-V Vector Extension (RVV 1.0) detection
#if defined(__riscv) && defined(__riscv_vector) && defined(TFLITE_RISCV_RVV) && \
    !defined(TFLITE_DISABLE_RISCV_RVV)
#define TFLITE_USE_RVV
#include <riscv_vector.h>
#endif

// Architecture-specific optimization selection
// Use NEON_OR_PORTABLE for ARM/x86, RVV_OR_PORTABLE for RISC-V
#ifdef TFLITE_USE_NEON
#define NEON_OR_PORTABLE(funcname, ...) Neon##funcname(__VA_ARGS__)
#elif defined(TFLITE_USE_RVV)
#define NEON_OR_PORTABLE(funcname, ...) Rvv##funcname(__VA_ARGS__)
#else
#define NEON_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)
#endif

// RVV-specific macro for RISC-V only optimizations
#ifdef TFLITE_USE_RVV
#define RVV_OR_PORTABLE(funcname, ...) Rvv##funcname(__VA_ARGS__)
#else
#define RVV_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_ARCH_CHECK_H_
