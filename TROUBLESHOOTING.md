# Troubleshooting

## Proto compilation fails with "File does not reside within any path specified using --proto_path"

### Symptom

Running CMake build for tflite proto targets (`profiling_info_proto`, `model_runtime_info_proto`, `benchmark_result_proto`) fails with protoc error:

```
File does not reside within any path specified using --proto_path (or -I).
```

### Cause

The proto files in the standalone LiteRT repo use `tflite/...` import paths (e.g., `import "tflite/profiling/proto/profiling_info.proto"`), but the `--proto_path` passed to protoc did not point to a directory that could resolve these paths. Additionally, the generated output file paths were hardcoded as `tensorflow/lite/...`, which does not match what protoc actually produces when the proto files are under `tflite/...`.

This originated from the proto CMakeLists.txt files being carried over from the TensorFlow monorepo, where the directory structure was `tensorflow/lite/profiling/proto/` and `--proto_path` pointed to `TENSORFLOW_SOURCE_DIR`. In the standalone LiteRT repo the directory structure changed to `tflite/profiling/proto/`, so the old path configuration no longer worked.

### Affected files

- `tflite/profiling/proto/CMakeLists.txt`
- `tflite/tools/benchmark/proto/CMakeLists.txt`

### Solution

Replace the hardcoded `--proto_path` and output directory paths with build-mode auto-detection:

```cmake
string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" _is_tf_integrated)
if(NOT _is_tf_integrated EQUAL -1)
  # TF-integrated build
  set(_PROTO_PATH "${TENSORFLOW_SOURCE_DIR}")
  set(_PROTO_RELATIVE_DIR "tensorflow/lite/profiling/proto")
else()
  # Standalone build
  get_filename_component(_PROTO_PATH "${TFLITE_SOURCE_DIR}/.." ABSOLUTE)
  set(_PROTO_RELATIVE_DIR "tflite/profiling/proto")
endif()
```

| Build mode | `--proto_path` | Output path |
|---|---|---|
| Standalone | Parent of `TFLITE_SOURCE_DIR` (LiteRT repo root) | `${CMAKE_BINARY_DIR}/tflite/...` |
| TF-integrated | `TENSORFLOW_SOURCE_DIR` (TF repo root) | `${CMAKE_BINARY_DIR}/tensorflow/lite/...` |

Key points:

- Use `TFLITE_SOURCE_DIR` (set by the parent `tflite/CMakeLists.txt`) instead of `CMAKE_SOURCE_DIR/..` to locate the proto import root — this works regardless of which CMakeLists.txt is the top-level entry point.
- The detection checks whether `CMAKE_CURRENT_SOURCE_DIR` is under `${TENSORFLOW_SOURCE_DIR}/tensorflow/lite`. If yes, we are inside the TF source tree; otherwise, it is a standalone build.
- The generated output path must match the proto file's path relative to `--proto_path`, otherwise CMake's custom command will never find its declared outputs.


## Segmentation fault at litert/vendors/samsung/ai_litecore_manager.h:83:27 when using riscv64-linux-gnu-g++ as compiler

I am not sure why but it's alright when using clang++ as compiler.
```
83 | struct AiLiteCoreManager::PublicApi {
   |                           ^~~~~~~~~
0x279df5f internal_error(char const*, ...)
	???:0
0xbb179c decl_namespace_context(tree_node*)
	???:0
0xbb17cb decl_anon_ns_mem_p(tree_node*)
	???:0
0xa654ff constrain_class_visibility(tree_node*)
	???:0
0x9f009b finish_struct_1(tree_node*)
	???:0
0x9f1383 finish_struct(tree_node*, tree_node*)
	???:0
0xb4273f c_parse_file()
	???:0
0xc38cf7 c_common_parse_file()
	???:0
Please submit a full bug report, with preprocessed source (by using -freport-bug).
Please include the complete backtrace with any bug report.
```

I just keep moving right now.
