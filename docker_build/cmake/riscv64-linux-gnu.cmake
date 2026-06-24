# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

set(RISCV64_MARCH "rv64gc" CACHE STRING "RISC-V -march value")
set(RISCV64_MABI "lp64d" CACHE STRING "RISC-V -mabi value")

set(CMAKE_C_FLAGS_INIT "-march=${RISCV64_MARCH} -mabi=${RISCV64_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV64_MARCH} -mabi=${RISCV64_MABI}")

# Debian/Ubuntu cross compilers already know their target sysroot. Keep this as
# a find root for headers/libraries while allowing host build tools from PATH.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(TFLITE_RISCV64 ON)
