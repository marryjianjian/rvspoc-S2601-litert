#
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
find_package(Protobuf REQUIRED)

# Determine proto_path and output directory based on build context.
# Standalone build: proto files use 'tflite/...' import paths.
# TF-integrated build: proto files use 'tensorflow/lite/...' import paths.
string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" _is_tf_integrated)
if(NOT _is_tf_integrated EQUAL -1)
  # TF-integrated build: proto files are inside the TF source tree.
  set(_PROTO_PATH "${TENSORFLOW_SOURCE_DIR}")
  set(_PROTO_RELATIVE_DIR "tensorflow/lite/tools/benchmark/proto")
else()
  # Standalone build: proto files are in the LiteRT repository.
  # TFLITE_SOURCE_DIR points to the tflite/ directory; go up to its parent.
  get_filename_component(_PROTO_PATH "${TFLITE_SOURCE_DIR}/.." ABSOLUTE)
  set(_PROTO_RELATIVE_DIR "tflite/tools/benchmark/proto")
endif()

add_library(benchmark_result_proto benchmark_result.proto)

list(APPEND benchmark_result_generated_files
    ${CMAKE_BINARY_DIR}/${_PROTO_RELATIVE_DIR}/benchmark_result.pb.cc
    ${CMAKE_BINARY_DIR}/${_PROTO_RELATIVE_DIR}/benchmark_result.pb.h)

# Generate benchmark_result.pb.cc and benchmark_result.pb.h from
# benchmark_result.proto using protoc. Once the protobuf package version is
# upgraded, we can use protobuf_generate_cpp/protobuf_generate here directly.
add_custom_command(
    OUTPUT ${benchmark_result_generated_files}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${CMAKE_BINARY_DIR} --proto_path=${_PROTO_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_result.proto
    DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_result.proto
)

set_source_files_properties(${benchmark_result_generated_files} PROPERTIES GENERATED TRUE)
target_sources(benchmark_result_proto PRIVATE ${benchmark_result_generated_files})
target_link_libraries(benchmark_result_proto protobuf::libprotobuf)
target_include_directories(benchmark_result_proto PUBLIC ${CMAKE_BINARY_DIR})
