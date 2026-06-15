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

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tflite/core/c/common.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/kernels/register.h"
#include "tflite/core/model.h"
#include "tflite/core/model_builder.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/kernel_util.h"

namespace {

struct Options {
  std::string model_path;
  std::string mode = "compare";
  std::string label;
  std::string result_file;
  std::string json_output;
  std::string input_dump_path;
  std::string tensor_dump_path;
  std::string reference_binary;
  std::string rvv_binary;
  std::string scalar_binary;
  std::string candidate_runner;
  std::string reference_runner;
  std::vector<std::string> candidate_runner_args;
  std::vector<std::string> reference_runner_args;
  int samples = 1;
  int num_threads = 1;
  double fp32_top1_tolerance_percent = 0.1;
  double int8_top1_tolerance_percent = 1.0;
  double fp32_relative_tolerance = 1e-5;
  int int8_lsb_tolerance = 1;
};

struct TensorResult {
  int type = kTfLiteNoType;
  int top1 = -1;
  std::vector<double> values;
};

struct SampleResult {
  std::vector<TensorResult> outputs;
};

struct RunResult {
  std::string label;
  bool rvv_compiled = false;
  std::vector<SampleResult> samples;
};

struct CompareStats {
  std::string label;
  int samples = 0;
  int top1_mismatches = 0;
  double top1_mismatch_percent = 0.0;
  double max_abs_diff = 0.0;
  double max_relative_error = 0.0;
  int64_t max_lsb_diff = 0;
  bool top1_pass = false;
  bool element_pass = false;
  bool pass = false;
};

struct InputDumpTensor {
  int type = kTfLiteNoType;
  std::vector<int> shape;
  std::vector<uint8_t> data;
};

struct InputDumpSample {
  std::vector<InputDumpTensor> inputs;
};

struct InputDump {
  std::vector<InputDumpSample> samples;
};

bool IsRvvCompiled() {
#ifdef USE_RVV
  return true;
#else
  return false;
#endif
}

std::string BaseName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return path;
  }
  return path.substr(slash + 1);
}

std::string DirName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  if (slash == 0) {
    return "/";
  }
  return path.substr(0, slash);
}

std::string MakeTempFile(const std::string& suffix) {
  std::string pattern = "/tmp/rvv_model_precision_XXXXXX";
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  const int fd = mkstemp(buffer.data());
  if (fd >= 0) {
    close(fd);
  }
  std::string path(buffer.data());
  std::rename(path.c_str(), (path + suffix).c_str());
  return path + suffix;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --model=/path/model.tflite [options]\n\n"
      << "Options:\n"
      << "  --mode=compare|run\n"
      << "  --model=PATH\n"
      << "  --samples=N                         Default: 1\n"
      << "  --num_threads=N                     Default: 1\n"
      << "  --input_dump=PATH                   Real input tensor dump\n"
      << "  --reference_binary=PATH             Default: this binary\n"
      << "  --rvv_binary=PATH                   Candidate RVV binary\n"
      << "  --scalar_binary=PATH                Candidate scalar binary\n"
      << "  --candidate_runner=PATH             e.g. qemu-riscv64\n"
      << "  --candidate_runner_arg=ARG          Repeatable\n"
      << "  --reference_runner=PATH             Optional reference runner\n"
      << "  --reference_runner_arg=ARG          Repeatable\n"
      << "  --json_output=PATH\n"
      << "  --result_file=PATH                  Internal run output\n"
      << "  --tensor_dump=PATH                  Internal tensor dump output\n"
      << "  --label=NAME                        Internal run label\n";
}

bool ConsumeArgValue(int* index, int argc, char** argv, const std::string& arg,
                     const std::string& name, std::string* value) {
  const std::string prefix = "--" + name + "=";
  if (arg.rfind(prefix, 0) == 0) {
    *value = arg.substr(prefix.size());
    return true;
  }
  if (arg == "--" + name) {
    if (*index + 1 >= argc) {
      std::cerr << "Missing value for --" << name << "\n";
      return false;
    }
    *value = argv[++(*index)];
    return true;
  }
  return false;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (ConsumeArgValue(&i, argc, argv, arg, "model", &value)) {
      options->model_path = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "mode", &value)) {
      options->mode = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "label", &value)) {
      options->label = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "result_file", &value)) {
      options->result_file = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "json_output", &value)) {
      options->json_output = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "input_dump", &value)) {
      options->input_dump_path = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "tensor_dump", &value)) {
      options->tensor_dump_path = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "reference_binary",
                               &value)) {
      options->reference_binary = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "rvv_binary", &value)) {
      options->rvv_binary = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "scalar_binary", &value)) {
      options->scalar_binary = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "candidate_runner",
                               &value)) {
      options->candidate_runner = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "candidate_runner_arg",
                               &value)) {
      options->candidate_runner_args.push_back(value);
    } else if (ConsumeArgValue(&i, argc, argv, arg, "reference_runner",
                               &value)) {
      options->reference_runner = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "reference_runner_arg",
                               &value)) {
      options->reference_runner_args.push_back(value);
    } else if (ConsumeArgValue(&i, argc, argv, arg, "samples", &value)) {
      options->samples = std::stoi(value);
    } else if (ConsumeArgValue(&i, argc, argv, arg, "num_threads", &value)) {
      options->num_threads = std::stoi(value);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (options->model_path.empty()) {
    std::cerr << "--model is required.\n";
    return false;
  }
  if (options->samples <= 0 || options->num_threads <= 0) {
    std::cerr << "samples and num_threads must be positive.\n";
    return false;
  }
  return true;
}

template <typename T>
bool ReadBinary(std::istream& in, T* value) {
  return static_cast<bool>(
      in.read(reinterpret_cast<char*>(value), sizeof(T)));
}

template <typename T>
bool WriteBinary(std::ostream& out, const T& value) {
  return static_cast<bool>(
      out.write(reinterpret_cast<const char*>(&value), sizeof(T)));
}

bool ReadBytes(std::istream& in, std::vector<uint8_t>* data) {
  if (data->empty()) {
    return true;
  }
  return static_cast<bool>(
      in.read(reinterpret_cast<char*>(data->data()), data->size()));
}

bool LoadInputDump(const std::string& path, InputDump* dump) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open input dump: " << path << "\n";
    return false;
  }

  char magic[8] = {};
  if (!in.read(magic, sizeof(magic)) ||
      std::memcmp(magic, "RVVINP01", sizeof(magic)) != 0) {
    std::cerr << "Invalid input dump magic: " << path << "\n";
    return false;
  }

  uint32_t sample_count = 0;
  if (!ReadBinary(in, &sample_count)) {
    std::cerr << "Failed to read input dump sample count: " << path << "\n";
    return false;
  }
  dump->samples.clear();
  dump->samples.resize(sample_count);

  for (uint32_t sample = 0; sample < sample_count; ++sample) {
    uint32_t input_count = 0;
    if (!ReadBinary(in, &input_count)) {
      std::cerr << "Failed to read input count for sample " << sample << "\n";
      return false;
    }
    dump->samples[sample].inputs.resize(input_count);
    for (uint32_t input = 0; input < input_count; ++input) {
      InputDumpTensor& tensor = dump->samples[sample].inputs[input];
      int32_t type = kTfLiteNoType;
      uint32_t rank = 0;
      if (!ReadBinary(in, &type) || !ReadBinary(in, &rank)) {
        std::cerr << "Failed to read tensor header for sample " << sample
                  << ", input " << input << "\n";
        return false;
      }
      tensor.type = type;
      tensor.shape.resize(rank);
      for (uint32_t dim = 0; dim < rank; ++dim) {
        int32_t value = 0;
        if (!ReadBinary(in, &value)) {
          std::cerr << "Failed to read tensor shape for sample " << sample
                    << ", input " << input << "\n";
          return false;
        }
        tensor.shape[dim] = value;
      }
      uint64_t byte_size = 0;
      if (!ReadBinary(in, &byte_size) ||
          byte_size >
              static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        std::cerr << "Invalid tensor byte size for sample " << sample
                  << ", input " << input << "\n";
        return false;
      }
      tensor.data.resize(static_cast<size_t>(byte_size));
      if (!ReadBytes(in, &tensor.data)) {
        std::cerr << "Failed to read tensor bytes for sample " << sample
                  << ", input " << input << "\n";
        return false;
      }
    }
  }

  char trailing = 0;
  if (in.read(&trailing, 1)) {
    std::cerr << "Input dump has trailing bytes: " << path << "\n";
    return false;
  }
  return true;
}

bool TensorShapeMatches(const TfLiteTensor* tensor,
                        const std::vector<int>& dump_shape) {
  if (tensor == nullptr || tensor->dims == nullptr ||
      tensor->dims->size != static_cast<int>(dump_shape.size())) {
    return false;
  }
  for (int i = 0; i < tensor->dims->size; ++i) {
    if (tensor->dims->data[i] != dump_shape[i]) {
      return false;
    }
  }
  return true;
}

bool FillInputFromDump(TfLiteTensor* tensor, const InputDumpTensor& input,
                       int sample, int input_index) {
  if (tensor == nullptr || tensor->data.raw == nullptr) {
    std::cerr << "Missing input tensor storage for sample " << sample
              << ", input " << input_index << "\n";
    return false;
  }
  if (tensor->type != input.type) {
    std::cerr << "Input dump type mismatch for sample " << sample
              << ", input " << input_index << ": model=" << tensor->type
              << ", dump=" << input.type << "\n";
    return false;
  }
  if (!TensorShapeMatches(tensor, input.shape)) {
    std::cerr << "Input dump shape mismatch for sample " << sample
              << ", input " << input_index << "\n";
    return false;
  }
  if (tensor->bytes != input.data.size()) {
    std::cerr << "Input dump byte size mismatch for sample " << sample
              << ", input " << input_index << ": model=" << tensor->bytes
              << ", dump=" << input.data.size() << "\n";
    return false;
  }
  std::memcpy(tensor->data.raw, input.data.data(), input.data.size());
  return true;
}

bool WriteTensorDumpHeader(std::ostream& out, int samples) {
  const char magic[8] = {'R', 'V', 'V', 'T', 'E', 'N', '0', '1'};
  const uint32_t sample_count = static_cast<uint32_t>(samples);
  return static_cast<bool>(out.write(magic, sizeof(magic))) &&
         WriteBinary(out, sample_count);
}

bool WriteTensorDumpSample(std::ostream& out,
                           const tflite::Interpreter& interpreter) {
  const uint32_t tensor_count =
      static_cast<uint32_t>(interpreter.tensors_size());
  if (!WriteBinary(out, tensor_count)) {
    return false;
  }
  for (uint32_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
    const TfLiteTensor* tensor =
        interpreter.tensor(static_cast<int>(tensor_index));
    const int32_t type = tensor == nullptr ? kTfLiteNoType : tensor->type;
    const uint32_t rank =
        tensor == nullptr || tensor->dims == nullptr
            ? 0
            : static_cast<uint32_t>(tensor->dims->size);
    const uint64_t byte_size =
        tensor == nullptr || tensor->data.raw == nullptr
            ? 0
            : static_cast<uint64_t>(tensor->bytes);
    const char* raw_name =
        tensor == nullptr || tensor->name == nullptr ? "" : tensor->name;
    const std::string name(raw_name);
    const uint32_t name_size = static_cast<uint32_t>(name.size());

    if (!WriteBinary(out, tensor_index) || !WriteBinary(out, type) ||
        !WriteBinary(out, rank)) {
      return false;
    }
    for (uint32_t dim = 0; dim < rank; ++dim) {
      const int32_t value = tensor->dims->data[dim];
      if (!WriteBinary(out, value)) {
        return false;
      }
    }
    if (!WriteBinary(out, byte_size) || !WriteBinary(out, name_size) ||
        !out.write(name.data(), name.size())) {
      return false;
    }
    if (byte_size != 0 &&
        !out.write(tensor->data.raw, static_cast<std::streamsize>(byte_size))) {
      return false;
    }
  }
  return true;
}

void FillTensor(TfLiteTensor* tensor, int sample) {
  if (tensor == nullptr || tensor->data.raw == nullptr || tensor->bytes == 0) {
    return;
  }
  const size_t elements =
      tensor->bytes /
      std::max<size_t>(1, tflite::TfLiteTypeGetSize(tensor->type));
  const size_t sample_offset = static_cast<size_t>(sample) * 997;
  switch (tensor->type) {
    case kTfLiteFloat32: {
      float* data = tensor->data.f;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = std::sin(static_cast<float>(i + sample_offset) * 0.013f);
      }
      break;
    }
    case kTfLiteInt8: {
      int8_t* data = tensor->data.int8;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<int8_t>(
            static_cast<int>(((i + sample_offset) * 13 + 7) % 256) - 128);
      }
      break;
    }
    case kTfLiteUInt8: {
      uint8_t* data = tensor->data.uint8;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<uint8_t>(((i + sample_offset) * 13 + 7) % 256);
      }
      break;
    }
    default:
      std::memset(tensor->data.raw, 0, tensor->bytes);
      break;
  }
}

int Top1(const std::vector<double>& values) {
  if (values.empty()) {
    return -1;
  }
  return static_cast<int>(std::max_element(values.begin(), values.end()) -
                          values.begin());
}

std::vector<double> TensorValues(const TfLiteTensor* tensor) {
  const size_t elements =
      tensor->bytes /
      std::max<size_t>(1, tflite::TfLiteTypeGetSize(tensor->type));
  std::vector<double> values(elements);
  switch (tensor->type) {
    case kTfLiteFloat32:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.f[i];
      break;
    case kTfLiteFloat64:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.f64[i];
      break;
    case kTfLiteInt8:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.int8[i];
      break;
    case kTfLiteUInt8:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.uint8[i];
      break;
    case kTfLiteInt16:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.i16[i];
      break;
    case kTfLiteUInt16:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.ui16[i];
      break;
    case kTfLiteInt32:
      for (size_t i = 0; i < elements; ++i) values[i] = tensor->data.i32[i];
      break;
    case kTfLiteInt64:
      for (size_t i = 0; i < elements; ++i) {
        values[i] = static_cast<double>(tensor->data.i64[i]);
      }
      break;
    default:
      std::fill(values.begin(), values.end(), 0.0);
      break;
  }
  return values;
}

bool RunModel(const Options& options, RunResult* result) {
  result->label = options.label.empty() ? (IsRvvCompiled() ? "rvv" : "unknown")
                                        : options.label;
  result->rvv_compiled = IsRvvCompiled();

  auto model =
      tflite::FlatBufferModel::BuildFromFile(options.model_path.c_str());
  if (!model) {
    std::cerr << "Failed to load model: " << options.model_path << "\n";
    return false;
  }

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions interpreter_options;
  if (!options.tensor_dump_path.empty()) {
    interpreter_options.SetPreserveAllTensors(true);
  }
  tflite::InterpreterBuilder builder(*model, resolver, &interpreter_options);
  if (builder.SetNumThreads(options.num_threads) != kTfLiteOk) {
    std::cerr << "Failed to set num_threads=" << options.num_threads << "\n";
    return false;
  }

  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  if (!interpreter) {
    std::cerr << "Failed to create interpreter.\n";
    return false;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "AllocateTensors failed.\n";
    return false;
  }

  InputDump input_dump;
  const InputDump* input_dump_ptr = nullptr;
  if (!options.input_dump_path.empty()) {
    if (!LoadInputDump(options.input_dump_path, &input_dump)) {
      return false;
    }
    if (input_dump.samples.size() < static_cast<size_t>(options.samples)) {
      std::cerr << "Input dump has " << input_dump.samples.size()
                << " samples, but --samples=" << options.samples << "\n";
      return false;
    }
    input_dump_ptr = &input_dump;
  }

  std::ofstream tensor_dump;
  if (!options.tensor_dump_path.empty()) {
    tensor_dump.open(options.tensor_dump_path, std::ios::binary);
    if (!tensor_dump) {
      std::cerr << "Failed to open tensor dump: "
                << options.tensor_dump_path << "\n";
      return false;
    }
    if (!WriteTensorDumpHeader(tensor_dump, options.samples)) {
      std::cerr << "Failed to write tensor dump header: "
                << options.tensor_dump_path << "\n";
      return false;
    }
  }

  result->samples.clear();
  result->samples.reserve(options.samples);
  for (int sample = 0; sample < options.samples; ++sample) {
    const std::vector<int>& inputs = interpreter->inputs();
    if (input_dump_ptr != nullptr &&
        input_dump_ptr->samples[sample].inputs.size() != inputs.size()) {
      std::cerr << "Input dump input count mismatch for sample " << sample
                << ": model=" << inputs.size() << ", dump="
                << input_dump_ptr->samples[sample].inputs.size() << "\n";
      return false;
    }
    for (size_t input = 0; input < inputs.size(); ++input) {
      TfLiteTensor* tensor = interpreter->tensor(inputs[input]);
      if (input_dump_ptr != nullptr) {
        if (!FillInputFromDump(tensor,
                               input_dump_ptr->samples[sample].inputs[input],
                               sample, static_cast<int>(input))) {
          return false;
        }
      } else {
        FillTensor(tensor, sample);
      }
    }
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Invoke failed at sample " << sample << ".\n";
      return false;
    }
    if (tensor_dump.is_open() &&
        !WriteTensorDumpSample(tensor_dump, *interpreter)) {
      std::cerr << "Failed to write tensor dump sample " << sample << ": "
                << options.tensor_dump_path << "\n";
      return false;
    }
    SampleResult sample_result;
    for (int output_index : interpreter->outputs()) {
      const TfLiteTensor* tensor = interpreter->tensor(output_index);
      TensorResult tensor_result;
      tensor_result.type = tensor->type;
      tensor_result.values = TensorValues(tensor);
      tensor_result.top1 = Top1(tensor_result.values);
      sample_result.outputs.push_back(std::move(tensor_result));
    }
    result->samples.push_back(std::move(sample_result));
  }
  return true;
}

bool WriteRunFile(const std::string& path, const RunResult& result) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to open result file: " << path << "\n";
    return false;
  }
  out << "label=" << result.label << "\n";
  out << "rvv_compiled=" << (result.rvv_compiled ? 1 : 0) << "\n";
  out << "samples=" << result.samples.size() << "\n";
  for (size_t sample = 0; sample < result.samples.size(); ++sample) {
    out << "sample=" << sample << "\n";
    out << "outputs=" << result.samples[sample].outputs.size() << "\n";
    for (size_t tensor = 0; tensor < result.samples[sample].outputs.size();
         ++tensor) {
      const TensorResult& output = result.samples[sample].outputs[tensor];
      out << "tensor=" << tensor << "\n";
      out << "type=" << output.type << "\n";
      out << "size=" << output.values.size() << "\n";
      out << "top1=" << output.top1 << "\n";
      out << "values=" << std::setprecision(17);
      for (double value : output.values) {
        out << ' ' << value;
      }
      out << "\n";
    }
  }
  return true;
}

bool ReadExpectedLine(std::istream& in, const char* key, std::string* value) {
  std::string line;
  if (!std::getline(in, line)) {
    return false;
  }
  const std::string prefix = std::string(key) + "=";
  if (line.rfind(prefix, 0) != 0) {
    std::cerr << "Expected key " << key << ", got: " << line << "\n";
    return false;
  }
  *value = line.substr(prefix.size());
  return true;
}

bool LoadRunFile(const std::string& path, RunResult* result) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Failed to read result file: " << path << "\n";
    return false;
  }
  std::string value;
  if (!ReadExpectedLine(in, "label", &result->label)) return false;
  if (!ReadExpectedLine(in, "rvv_compiled", &value)) return false;
  result->rvv_compiled = std::stoi(value) != 0;
  if (!ReadExpectedLine(in, "samples", &value)) return false;
  const int sample_count = std::stoi(value);
  result->samples.resize(sample_count);
  for (int sample = 0; sample < sample_count; ++sample) {
    if (!ReadExpectedLine(in, "sample", &value)) return false;
    if (!ReadExpectedLine(in, "outputs", &value)) return false;
    const int output_count = std::stoi(value);
    result->samples[sample].outputs.resize(output_count);
    for (int tensor = 0; tensor < output_count; ++tensor) {
      TensorResult& output = result->samples[sample].outputs[tensor];
      if (!ReadExpectedLine(in, "tensor", &value)) return false;
      if (!ReadExpectedLine(in, "type", &value)) return false;
      output.type = std::stoi(value);
      if (!ReadExpectedLine(in, "size", &value)) return false;
      const int size = std::stoi(value);
      if (!ReadExpectedLine(in, "top1", &value)) return false;
      output.top1 = std::stoi(value);
      if (!ReadExpectedLine(in, "values", &value)) return false;
      std::istringstream values_stream(value);
      output.values.resize(size);
      for (int i = 0; i < size; ++i) {
        values_stream >> output.values[i];
      }
    }
  }
  return true;
}

bool IsFloatType(int type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat64;
}

bool IsQuantizedModel(const RunResult& result) {
  return !result.samples.empty() && !result.samples[0].outputs.empty() &&
         !IsFloatType(result.samples[0].outputs[0].type);
}

double StrictRelativeError(double reference, double actual) {
  const double abs_diff = std::abs(actual - reference);
  const double ref_abs = std::abs(reference);
  if (ref_abs == 0.0) {
    return abs_diff == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return abs_diff / ref_abs;
}

CompareStats CompareToReference(const RunResult& reference,
                                const RunResult& candidate,
                                const Options& options) {
  CompareStats stats;
  stats.label = candidate.label;
  stats.samples = static_cast<int>(
      std::min(reference.samples.size(), candidate.samples.size()));
  const bool quantized = IsQuantizedModel(reference);
  const double top1_tolerance = quantized ? options.int8_top1_tolerance_percent
                                          : options.fp32_top1_tolerance_percent;

  bool element_pass = reference.samples.size() == candidate.samples.size();
  int64_t compared_values = 0;
  for (int sample = 0; sample < stats.samples; ++sample) {
    if (!reference.samples[sample].outputs.empty() &&
        !candidate.samples[sample].outputs.empty() &&
        reference.samples[sample].outputs[0].top1 !=
            candidate.samples[sample].outputs[0].top1) {
      ++stats.top1_mismatches;
    }
    if (reference.samples[sample].outputs.size() !=
        candidate.samples[sample].outputs.size()) {
      element_pass = false;
    }
    const size_t output_count =
        std::min(reference.samples[sample].outputs.size(),
                 candidate.samples[sample].outputs.size());
    for (size_t tensor = 0; tensor < output_count; ++tensor) {
      const TensorResult& ref = reference.samples[sample].outputs[tensor];
      const TensorResult& actual = candidate.samples[sample].outputs[tensor];
      if (ref.type != actual.type ||
          ref.values.size() != actual.values.size()) {
        element_pass = false;
      }
      const size_t size = std::min(ref.values.size(), actual.values.size());
      for (size_t i = 0; i < size; ++i) {
        const double abs_diff = std::abs(actual.values[i] - ref.values[i]);
        stats.max_abs_diff = std::max(stats.max_abs_diff, abs_diff);
        ++compared_values;
        if (IsFloatType(ref.type)) {
          const double rel =
              StrictRelativeError(ref.values[i], actual.values[i]);
          stats.max_relative_error = std::max(stats.max_relative_error, rel);
          if (rel > options.fp32_relative_tolerance) {
            element_pass = false;
          }
        } else {
          const int64_t lsb =
              std::llabs(static_cast<int64_t>(std::llround(actual.values[i])) -
                         static_cast<int64_t>(std::llround(ref.values[i])));
          stats.max_lsb_diff = std::max(stats.max_lsb_diff, lsb);
          if (lsb > options.int8_lsb_tolerance) {
            element_pass = false;
          }
        }
      }
    }
  }
  if (compared_values == 0) {
    element_pass = false;
  }
  stats.top1_mismatch_percent =
      stats.samples == 0 ? 100.0
                         : 100.0 * static_cast<double>(stats.top1_mismatches) /
                               static_cast<double>(stats.samples);
  stats.top1_pass = stats.top1_mismatch_percent <= top1_tolerance;
  stats.element_pass = element_pass;
  stats.pass = stats.top1_pass && stats.element_pass;
  return stats;
}

std::vector<std::string> ChildArgs(const Options& options,
                                   const std::string& label,
                                   const std::string& result_file) {
  std::vector<std::string> args = {
      "--mode=run",
      "--label=" + label,
      "--model=" + options.model_path,
      "--samples=" + std::to_string(options.samples),
      "--num_threads=" + std::to_string(options.num_threads),
      "--result_file=" + result_file,
  };
  if (!options.input_dump_path.empty()) {
    args.push_back("--input_dump=" + options.input_dump_path);
  }
  return args;
}

int RunProcess(const std::string& program, const std::string& runner,
               const std::vector<std::string>& runner_args,
               const std::vector<std::string>& args) {
  const std::string executable = runner.empty() ? program : runner;
  std::vector<std::string> argv_storage;
  argv_storage.push_back(executable);
  if (!runner.empty()) {
    argv_storage.insert(argv_storage.end(), runner_args.begin(),
                        runner_args.end());
    argv_storage.push_back(program);
  }
  argv_storage.insert(argv_storage.end(), args.begin(), args.end());

  std::vector<char*> argv;
  argv.reserve(argv_storage.size() + 1);
  for (std::string& arg : argv_storage) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "fork failed: " << std::strerror(errno) << "\n";
    return 1;
  }
  if (pid == 0) {
    execvp(executable.c_str(), argv.data());
    std::cerr << "exec failed for " << executable << ": "
              << std::strerror(errno) << "\n";
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::cerr << "waitpid failed: " << std::strerror(errno) << "\n";
    return 1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    std::cerr << program << " terminated by signal " << WTERMSIG(status)
              << "\n";
  }
  return 1;
}

void PrintStats(const Options& options, const RunResult& reference,
                const CompareStats& rvv, const CompareStats& scalar) {
  const bool quantized = IsQuantizedModel(reference);
  std::cout << "\nRVV model precision comparison\n";
  std::cout << "model: " << options.model_path << "\n";
  std::cout << "samples: " << options.samples
            << ", model_kind: " << (quantized ? "int8/quantized" : "fp32")
            << "\n";
  std::cout << "thresholds: top1_mismatch <= "
            << (quantized ? options.int8_top1_tolerance_percent
                          : options.fp32_top1_tolerance_percent)
            << "%, ";
  if (quantized) {
    std::cout << "max_lsb <= " << options.int8_lsb_tolerance << "\n\n";
  } else {
    std::cout << "max_relative_error <= " << options.fp32_relative_tolerance
              << "\n\n";
  }

  auto print = [&](const CompareStats& stats) {
    std::cout << std::left << std::setw(10) << stats.label << std::right
              << " top1_mismatch=" << std::fixed << std::setprecision(3)
              << stats.top1_mismatch_percent << "%"
              << " max_abs=" << std::setprecision(9) << stats.max_abs_diff
              << " max_rel=" << stats.max_relative_error
              << " max_lsb=" << stats.max_lsb_diff
              << " result=" << (stats.pass ? "PASS" : "FAIL") << "\n";
  };
  print(rvv);
  print(scalar);
}

bool WriteJson(const std::string& path, const Options& options,
               const RunResult& reference, const CompareStats& rvv,
               const CompareStats& scalar) {
  if (path.empty()) {
    return true;
  }
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to write JSON output: " << path << "\n";
    return false;
  }
  auto write_double = [&](double value) {
    if (std::isfinite(value)) {
      out << value;
    } else if (value > 0) {
      out << "\"Infinity\"";
    } else {
      out << "\"NaN\"";
    }
  };
  auto write_stats = [&](const char* name, const CompareStats& stats) {
    out << "  \"" << name << "\": {\n";
    out << "    \"top1_mismatch_percent\": " << stats.top1_mismatch_percent
        << ",\n";
    out << "    \"max_abs_diff\": ";
    write_double(stats.max_abs_diff);
    out << ",\n";
    out << "    \"max_relative_error\": ";
    write_double(stats.max_relative_error);
    out << ",\n";
    out << "    \"max_lsb_diff\": " << stats.max_lsb_diff << ",\n";
    out << "    \"pass\": " << (stats.pass ? "true" : "false") << "\n";
    out << "  }";
  };
  out << "{\n";
  out << "  \"model\": \"" << options.model_path << "\",\n";
  out << "  \"samples\": " << options.samples << ",\n";
  out << "  \"model_kind\": \""
      << (IsQuantizedModel(reference) ? "int8/quantized" : "fp32") << "\",\n";
  write_stats("rvv", rvv);
  out << ",\n";
  write_stats("scalar", scalar);
  out << "\n}\n";
  return true;
}

int RunCompare(int argc, char** argv, Options options) {
  const std::string self = argv[0];
  if (options.reference_binary.empty()) {
    options.reference_binary = self;
  }
  if (options.rvv_binary.empty()) {
    options.rvv_binary = self;
  }
  if (options.scalar_binary.empty()) {
    options.scalar_binary = DirName(self) + "/" + BaseName(self) + "-norvv";
  }

  const std::string reference_file = MakeTempFile("_reference.txt");
  const std::string rvv_file = MakeTempFile("_rvv.txt");
  const std::string scalar_file = MakeTempFile("_scalar.txt");

  if (RunProcess(options.reference_binary, options.reference_runner,
                 options.reference_runner_args,
                 ChildArgs(options, "reference", reference_file)) != 0) {
    return 1;
  }
  if (RunProcess(options.rvv_binary, options.candidate_runner,
                 options.candidate_runner_args,
                 ChildArgs(options, "rvv", rvv_file)) != 0) {
    return 1;
  }
  if (RunProcess(options.scalar_binary, options.candidate_runner,
                 options.candidate_runner_args,
                 ChildArgs(options, "scalar", scalar_file)) != 0) {
    return 1;
  }

  RunResult reference;
  RunResult rvv;
  RunResult scalar;
  if (!LoadRunFile(reference_file, &reference) ||
      !LoadRunFile(rvv_file, &rvv) || !LoadRunFile(scalar_file, &scalar)) {
    return 1;
  }

  std::remove(reference_file.c_str());
  std::remove(rvv_file.c_str());
  std::remove(scalar_file.c_str());

  const CompareStats rvv_stats = CompareToReference(reference, rvv, options);
  const CompareStats scalar_stats =
      CompareToReference(reference, scalar, options);
  PrintStats(options, reference, rvv_stats, scalar_stats);
  if (!WriteJson(options.json_output, options, reference, rvv_stats,
                 scalar_stats)) {
    return 1;
  }
  return (rvv_stats.pass && scalar_stats.pass) ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (options.mode == "run") {
    RunResult result;
    if (!RunModel(options, &result)) {
      return 1;
    }
    if (!options.result_file.empty() &&
        !WriteRunFile(options.result_file, result)) {
      return 1;
    }
    return 0;
  }
  if (options.mode == "compare") {
    return RunCompare(argc, argv, options);
  }

  std::cerr << "Unsupported --mode=" << options.mode << "\n";
  return 1;
}
