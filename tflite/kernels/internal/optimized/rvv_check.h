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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_CHECK_H_

// RISC-V Vector Extension (RVV 1.0) detection
// Requires TFLITE_RISCV_RVV to be defined (set by CMake via -DTFLITE_RISCV_RVV)
// to be consistent with arch_check.h detection.
#if defined(__riscv) && defined(__riscv_vector) && defined(TFLITE_RISCV_RVV)
#define USE_RVV
#include <riscv_vector.h>  // IWYU pragma: export
#endif

// RVV_OR_PORTABLE(SomeFunc, args) calls RvvSomeFunc(args) if USE_RVV is
// defined, PortableSomeFunc(args) otherwise.
#ifdef USE_RVV
// Always use RVV code
#define RVV_OR_PORTABLE(funcname, ...) Rvv##funcname(__VA_ARGS__)

#else
// No RVV available: Use Portable code
#define RVV_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)

#endif  // defined(USE_RVV)

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_CHECK_H_
