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
find_package(Protobuf REQUIRED)

# Determine proto_path and output directory based on build context.
# Standalone build: proto files use 'tflite/...' import paths.
# TF-integrated build: proto files use 'tensorflow/lite/...' import paths.
string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" _is_tf_integrated)
if(NOT _is_tf_integrated EQUAL -1)
  # TF-integrated build: proto files are inside the TF source tree.
  set(_PROTO_PATH "${TENSORFLOW_SOURCE_DIR}")
  set(_PROTO_RELATIVE_DIR "tensorflow/lite/profiling/proto")
else()
  # Standalone build: proto files are in the LiteRT repository.
  # TFLITE_SOURCE_DIR points to the tflite/ directory; go up to its parent.
  get_filename_component(_PROTO_PATH "${TFLITE_SOURCE_DIR}/.." ABSOLUTE)
  set(_PROTO_RELATIVE_DIR "tflite/profiling/proto")
endif()

add_library(profiling_info_proto profiling_info.proto)

list(APPEND profiling_info_generated_files
    ${CMAKE_BINARY_DIR}/${_PROTO_RELATIVE_DIR}/profiling_info.pb.cc
    ${CMAKE_BINARY_DIR}/${_PROTO_RELATIVE_DIR}/profiling_info.pb.h)

# Generate profiling_info.pb.cc and profiling_info.pb.h from
# profiling_info.proto using protoc. Once the protobuf package version is
# upgraded, we can use protobuf_generate_cpp/protobuf_generate here directly.
add_custom_command(
    OUTPUT ${profiling_info_generated_files}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${CMAKE_BINARY_DIR} --proto_path=${_PROTO_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/profiling_info.proto
    DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling_info.proto
)

set_source_files_properties(${profiling_info_generated_files} PROPERTIES GENERATED TRUE)
target_sources(profiling_info_proto PRIVATE ${profiling_info_generated_files})
target_link_libraries(profiling_info_proto protobuf::libprotobuf)
target_include_directories(profiling_info_proto PUBLIC ${CMAKE_BINARY_DIR})

add_library(model_runtime_info_proto model_runtime_info.proto)
list(APPEND model_runtime_info_generated_files
    ${CMAKE_BINARY_DIR}/${_PROTO_RELATIVE_DIR}/model_runtime_info.pb.cc
    ${CMAKE_BINARY_DIR}/${_PROTO_RELATIVE_DIR}/model_runtime_info.pb.h
)

# Generate model_runtime_info.pb.cc and model_runtime_info.pb.h from
# model_runtime_info.proto using protoc. Once the protobuf package version is
# upgraded, we can use protobuf_generate_cpp/protobuf_generate here directly.
add_custom_command(
    OUTPUT ${model_runtime_info_generated_files}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${CMAKE_BINARY_DIR} --proto_path=${_PROTO_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/model_runtime_info.proto
    DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/model_runtime_info.proto ${profiling_info_generated_files}
)

set_source_files_properties(${model_runtime_info_generated_files} PROPERTIES GENERATED TRUE)
target_sources(model_runtime_info_proto PRIVATE ${model_runtime_info_generated_files})
target_link_libraries(model_runtime_info_proto protobuf::libprotobuf)
target_include_directories(model_runtime_info_proto PUBLIC ${CMAKE_BINARY_DIR})
