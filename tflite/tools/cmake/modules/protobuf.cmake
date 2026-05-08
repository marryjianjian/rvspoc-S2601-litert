#
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
  # Sync with tensorflow/third_party/protobuf/protobuf.patch
  GIT_TAG 90b73ac3f0b10320315c2ca0d03a5a9b095d2f66
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/protobuf"
)

set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
if(CMAKE_CROSSCOMPILING)
  # When cross-compiling, do not build protoc for the target architecture.
  # protoc is a host tool; a host-native protoc must be available.
  # It is set by litert/CMakeLists.txt or via -DProtobuf_PROTOC_EXECUTABLE=...
  if(NOT Protobuf_PROTOC_EXECUTABLE OR NOT EXISTS "${Protobuf_PROTOC_EXECUTABLE}")
    find_program(Protobuf_PROTOC_EXECUTABLE NAMES protoc)
  endif()
  if(NOT Protobuf_PROTOC_EXECUTABLE)
    message(FATAL_ERROR
      "protoc not found on the host system. When cross-compiling, a host-native"
      " protoc is required to generate .pb.cc/.pb.h files during the build."
      " Please install protobuf on the host, e.g.:\n"
      "  brew install protobuf          (macOS)\n"
      "  apt install protobuf-compiler  (Ubuntu/Debian)\n"
      "Alternatively, set -DProtobuf_PROTOC_EXECUTABLE=/path/to/host/protoc"
    )
  endif()
  set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
else()
  set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)
endif()

OverridableFetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
  OverridableFetchContent_Populate(protobuf)
endif()

# Export basic variables expected by consumers
set(Protobuf_INCLUDE_DIR "${protobuf_SOURCE_DIR}/src" CACHE INTERNAL "")
set(Protobuf_LIBRARIES protobuf::libprotobuf CACHE INTERNAL "")

# Avoid adding the same subdirectory twice when multiple find_package calls
# occur in a single configure (e.g., from different subprojects).
if(NOT TARGET protobuf::libprotobuf)
  add_subdirectory(${protobuf_SOURCE_DIR} ${protobuf_BINARY_DIR})
endif()

if(CMAKE_CROSSCOMPILING)
  # Already resolved above; log the path being used.
  message(STATUS "Cross-compiling: using host protoc: ${Protobuf_PROTOC_EXECUTABLE}")
else()
  set(Protobuf_PROTOC_EXECUTABLE protoc CACHE INTERNAL "")
endif()
