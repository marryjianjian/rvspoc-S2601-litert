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

cmake_minimum_required(VERSION 3.20)
project(litert_riscv64_smoke C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "LiteRT repository root")
set(TFLITE_SOURCE_DIR "${LITERT_REPO_ROOT}/tflite" CACHE PATH
    "Path to the standalone tflite source directory")

add_subdirectory(
  "${TFLITE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

add_executable(litert_riscv64_smoke
  litert_riscv64_smoke.cc
)

target_link_libraries(litert_riscv64_smoke
  tensorflow-lite
)
