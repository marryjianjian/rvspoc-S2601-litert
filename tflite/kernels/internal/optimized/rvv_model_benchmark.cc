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

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
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

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model_path;
  std::string mode = "compare";
  std::string label;
  std::string result_file;
  std::string json_output;
  std::string runner;
  std::vector<std::string> runner_args;
  int warmup_runs = 5;
  int runs = 50;
  int num_threads = 1;
};

struct Metrics {
  std::string label;
  bool rvv_compiled = false;
  double startup_ms = 0.0;
  double first_invoke_ms = 0.0;
  double total_invoke_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_avg_ms = 0.0;
  double latency_p50_ms = 0.0;
  double latency_p90_ms = 0.0;
  double latency_p95_ms = 0.0;
  double latency_p99_ms = 0.0;
  double latency_max_ms = 0.0;
  double throughput_inferences_per_second = 0.0;
  int warmup_runs = 0;
  int runs = 0;
  int num_threads = 0;
  int64_t rss_before_kb = 0;
  int64_t rss_after_startup_kb = 0;
  int64_t rss_after_inference_kb = 0;
  int64_t peak_rss_kb = 0;
  int64_t model_size_bytes = 0;
};

bool IsRvvCompiled() {
#ifdef USE_RVV
  return true;
#else
  return false;
#endif
}

double MsSince(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int64_t GetPeakRssKb() {
  rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }
#if defined(__APPLE__)
  return static_cast<int64_t>(usage.ru_maxrss / 1024);
#else
  return static_cast<int64_t>(usage.ru_maxrss);
#endif
}

int64_t GetCurrentRssKb() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  int64_t total_pages = 0;
  int64_t resident_pages = 0;
  statm >> total_pages >> resident_pages;
  if (!statm) {
    return 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return 0;
  }
  return resident_pages * page_size / 1024;
#else
  return GetPeakRssKb();
#endif
}

int64_t GetFileSize(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return 0;
  }
  return static_cast<int64_t>(file.tellg());
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

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --model=/path/model.tflite [options]\n\n"
      << "Options:\n"
      << "  --model=PATH           TFLite model to benchmark.\n"
      << "  --runs=N              Timed inference iterations. Default: 50.\n"
      << "  --warmup_runs=N       Warmup iterations. Default: 5.\n"
      << "  --num_threads=N       TFLite CPU threads. Default: 1.\n"
      << "  --json_output=PATH    Also write comparison JSON.\n"
      << "  --runner=PATH         Optional child process runner, e.g. qemu-riscv64.\n"
      << "  --runner_arg=ARG      Optional repeated runner argument.\n"
      << "  --mode=compare|run    Internal. Default: compare.\n"
      << "  --label=NAME          Internal child-run label.\n"
      << "  --result_file=PATH    Internal child-run metrics file.\n";
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
    } else if (ConsumeArgValue(&i, argc, argv, arg, "runner", &value)) {
      options->runner = value;
    } else if (ConsumeArgValue(&i, argc, argv, arg, "runner_arg", &value)) {
      options->runner_args.push_back(value);
    } else if (ConsumeArgValue(&i, argc, argv, arg, "runs", &value)) {
      options->runs = std::stoi(value);
    } else if (ConsumeArgValue(&i, argc, argv, arg, "warmup_runs", &value)) {
      options->warmup_runs = std::stoi(value);
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
  if (options->runs <= 0 || options->warmup_runs < 0 ||
      options->num_threads <= 0) {
    std::cerr
        << "runs and num_threads must be positive; warmup_runs must be >= 0.\n";
    return false;
  }
  return true;
}

void FillTensor(TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->data.raw == nullptr || tensor->bytes == 0) {
    return;
  }
  const size_t elements =
      tensor->bytes /
      std::max<size_t>(1, tflite::TfLiteTypeGetSize(tensor->type));
  switch (tensor->type) {
    case kTfLiteFloat32: {
      float* data = tensor->data.f;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.013f);
      }
      break;
    }
    case kTfLiteFloat64: {
      double* data = tensor->data.f64;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = std::sin(static_cast<double>(i) * 0.013);
      }
      break;
    }
    case kTfLiteInt8: {
      int8_t* data = tensor->data.int8;
      for (size_t i = 0; i < elements; ++i) {
        data[i] =
            static_cast<int8_t>((static_cast<int>(i * 13 + 7) % 127) - 63);
      }
      break;
    }
    case kTfLiteUInt8: {
      uint8_t* data = tensor->data.uint8;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<uint8_t>((i * 13 + 7) % 255);
      }
      break;
    }
    case kTfLiteInt16: {
      int16_t* data = tensor->data.i16;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<int16_t>((static_cast<int>(i * 7) % 2048) - 1024);
      }
      break;
    }
    case kTfLiteUInt16: {
      uint16_t* data = tensor->data.ui16;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<uint16_t>((i * 7) % 2048);
      }
      break;
    }
    case kTfLiteInt32: {
      int32_t* data = tensor->data.i32;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<int32_t>((i * 17) % 4096) - 2048;
      }
      break;
    }
    case kTfLiteInt64: {
      int64_t* data = tensor->data.i64;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<int64_t>((i * 19) % 4096) - 2048;
      }
      break;
    }
    case kTfLiteBool: {
      bool* data = tensor->data.b;
      for (size_t i = 0; i < elements; ++i) {
        data[i] = (i % 2) == 0;
      }
      break;
    }
    default:
      std::memset(tensor->data.raw, 0, tensor->bytes);
      break;
  }
}

double Percentile(std::vector<double> sorted_values, double percentile) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  std::sort(sorted_values.begin(), sorted_values.end());
  const double index =
      percentile * static_cast<double>(sorted_values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(index));
  const size_t upper = static_cast<size_t>(std::ceil(index));
  if (lower == upper) {
    return sorted_values[lower];
  }
  const double fraction = index - static_cast<double>(lower);
  return sorted_values[lower] * (1.0 - fraction) +
         sorted_values[upper] * fraction;
}

bool RunBenchmark(const Options& options, Metrics* metrics) {
  metrics->label = options.label.empty() ? (IsRvvCompiled() ? "rvv" : "no_rvv")
                                         : options.label;
  metrics->rvv_compiled = IsRvvCompiled();
  metrics->warmup_runs = options.warmup_runs;
  metrics->runs = options.runs;
  metrics->num_threads = options.num_threads;
  metrics->model_size_bytes = GetFileSize(options.model_path);
  metrics->rss_before_kb = GetCurrentRssKb();

  const auto startup_begin = Clock::now();
  auto model =
      tflite::FlatBufferModel::BuildFromFile(options.model_path.c_str());
  if (!model) {
    std::cerr << "Failed to load model: " << options.model_path << "\n";
    return false;
  }

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions interpreter_options;
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
  for (int input_index : interpreter->inputs()) {
    FillTensor(interpreter->tensor(input_index));
  }
  metrics->startup_ms = MsSince(startup_begin, Clock::now());
  metrics->rss_after_startup_kb = GetCurrentRssKb();

  const auto first_begin = Clock::now();
  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "First Invoke failed.\n";
    return false;
  }
  metrics->first_invoke_ms = MsSince(first_begin, Clock::now());

  for (int i = 0; i < options.warmup_runs; ++i) {
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Warmup Invoke failed at iteration " << i << ".\n";
      return false;
    }
  }

  std::vector<double> latencies_ms;
  latencies_ms.reserve(options.runs);
  const auto timed_begin = Clock::now();
  for (int i = 0; i < options.runs; ++i) {
    const auto begin = Clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Timed Invoke failed at iteration " << i << ".\n";
      return false;
    }
    latencies_ms.push_back(MsSince(begin, Clock::now()));
  }
  metrics->total_invoke_ms = MsSince(timed_begin, Clock::now());
  metrics->rss_after_inference_kb = GetCurrentRssKb();
  metrics->peak_rss_kb = GetPeakRssKb();

  const auto minmax =
      std::minmax_element(latencies_ms.begin(), latencies_ms.end());
  metrics->latency_min_ms = *minmax.first;
  metrics->latency_max_ms = *minmax.second;
  metrics->latency_avg_ms =
      std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
      static_cast<double>(latencies_ms.size());
  metrics->latency_p50_ms = Percentile(latencies_ms, 0.50);
  metrics->latency_p90_ms = Percentile(latencies_ms, 0.90);
  metrics->latency_p95_ms = Percentile(latencies_ms, 0.95);
  metrics->latency_p99_ms = Percentile(latencies_ms, 0.99);
  metrics->throughput_inferences_per_second =
      1000.0 * static_cast<double>(options.runs) / metrics->total_invoke_ms;
  return true;
}

void WriteMetric(std::ostream& os, const char* key, const std::string& value) {
  os << key << "=" << value << "\n";
}

void WriteMetric(std::ostream& os, const char* key, double value) {
  os << key << "=" << std::fixed << std::setprecision(6) << value << "\n";
}

void WriteMetric(std::ostream& os, const char* key, int64_t value) {
  os << key << "=" << value << "\n";
}

void WriteMetric(std::ostream& os, const char* key, int value) {
  WriteMetric(os, key, static_cast<int64_t>(value));
}

bool WriteMetricsFile(const std::string& path, const Metrics& metrics) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to open result file: " << path << "\n";
    return false;
  }
  WriteMetric(out, "label", metrics.label);
  WriteMetric(out, "rvv_compiled",
              metrics.rvv_compiled ? int64_t{1} : int64_t{0});
  WriteMetric(out, "startup_ms", metrics.startup_ms);
  WriteMetric(out, "first_invoke_ms", metrics.first_invoke_ms);
  WriteMetric(out, "total_invoke_ms", metrics.total_invoke_ms);
  WriteMetric(out, "latency_min_ms", metrics.latency_min_ms);
  WriteMetric(out, "latency_avg_ms", metrics.latency_avg_ms);
  WriteMetric(out, "latency_p50_ms", metrics.latency_p50_ms);
  WriteMetric(out, "latency_p90_ms", metrics.latency_p90_ms);
  WriteMetric(out, "latency_p95_ms", metrics.latency_p95_ms);
  WriteMetric(out, "latency_p99_ms", metrics.latency_p99_ms);
  WriteMetric(out, "latency_max_ms", metrics.latency_max_ms);
  WriteMetric(out, "throughput_inferences_per_second",
              metrics.throughput_inferences_per_second);
  WriteMetric(out, "warmup_runs", metrics.warmup_runs);
  WriteMetric(out, "runs", metrics.runs);
  WriteMetric(out, "num_threads", metrics.num_threads);
  WriteMetric(out, "rss_before_kb", metrics.rss_before_kb);
  WriteMetric(out, "rss_after_startup_kb", metrics.rss_after_startup_kb);
  WriteMetric(out, "rss_after_inference_kb", metrics.rss_after_inference_kb);
  WriteMetric(out, "peak_rss_kb", metrics.peak_rss_kb);
  WriteMetric(out, "model_size_bytes", metrics.model_size_bytes);
  return true;
}

bool LoadMetricsFile(const std::string& path, Metrics* metrics) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Failed to read result file: " << path << "\n";
    return false;
  }
  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(in, line)) {
    const size_t equals = line.find('=');
    if (equals == std::string::npos) {
      continue;
    }
    kv[line.substr(0, equals)] = line.substr(equals + 1);
  }
  auto get_double = [&](const char* key) { return std::stod(kv[key]); };
  auto get_i64 = [&](const char* key) { return std::stoll(kv[key]); };

  metrics->label = kv["label"];
  metrics->rvv_compiled = get_i64("rvv_compiled") != 0;
  metrics->startup_ms = get_double("startup_ms");
  metrics->first_invoke_ms = get_double("first_invoke_ms");
  metrics->total_invoke_ms = get_double("total_invoke_ms");
  metrics->latency_min_ms = get_double("latency_min_ms");
  metrics->latency_avg_ms = get_double("latency_avg_ms");
  metrics->latency_p50_ms = get_double("latency_p50_ms");
  metrics->latency_p90_ms = get_double("latency_p90_ms");
  metrics->latency_p95_ms = get_double("latency_p95_ms");
  metrics->latency_p99_ms = get_double("latency_p99_ms");
  metrics->latency_max_ms = get_double("latency_max_ms");
  metrics->throughput_inferences_per_second =
      get_double("throughput_inferences_per_second");
  metrics->warmup_runs = static_cast<int>(get_i64("warmup_runs"));
  metrics->runs = static_cast<int>(get_i64("runs"));
  metrics->num_threads = static_cast<int>(get_i64("num_threads"));
  metrics->rss_before_kb = get_i64("rss_before_kb");
  metrics->rss_after_startup_kb = get_i64("rss_after_startup_kb");
  metrics->rss_after_inference_kb = get_i64("rss_after_inference_kb");
  metrics->peak_rss_kb = get_i64("peak_rss_kb");
  metrics->model_size_bytes = get_i64("model_size_bytes");
  return true;
}

std::string MakeTempFile(const std::string& suffix) {
  std::string pattern = "/tmp/rvv_model_benchmark_XXXXXX";
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

int RunProcess(const Options& options, const std::string& program,
               const std::vector<std::string>& args) {
  const std::string executable =
      options.runner.empty() ? program : options.runner;
  std::vector<std::string> argv_storage;
  argv_storage.reserve(args.size() + options.runner_args.size() + 3);
  argv_storage.push_back(executable);
  if (!options.runner.empty()) {
    argv_storage.insert(argv_storage.end(), options.runner_args.begin(),
                        options.runner_args.end());
    argv_storage.push_back(program);
  }
  for (const std::string& arg : args) {
    argv_storage.push_back(arg);
  }

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

std::vector<std::string> ChildArgs(const Options& options,
                                   const std::string& label,
                                   const std::string& result_file) {
  return {
      "--mode=run",
      "--label=" + label,
      "--model=" + options.model_path,
      "--runs=" + std::to_string(options.runs),
      "--warmup_runs=" + std::to_string(options.warmup_runs),
      "--num_threads=" + std::to_string(options.num_threads),
      "--result_file=" + result_file,
  };
}

void PrintRow(const char* metric, double rvv, double no_rvv, const char* unit,
              bool higher_is_better) {
  const double ratio = higher_is_better ? rvv / no_rvv : no_rvv / rvv;
  std::cout << std::left << std::setw(28) << metric << std::right
            << std::setw(14) << std::fixed << std::setprecision(3) << rvv
            << std::setw(14) << no_rvv << std::setw(10) << unit << std::setw(12)
            << ratio << "x\n";
}

void PrintComparison(const Options& options, const Metrics& rvv,
                     const Metrics& no_rvv) {
  std::cout << "\nRVV model benchmark comparison\n";
  std::cout << "model: " << options.model_path << "\n";
  std::cout << "runs: " << options.runs
            << ", warmup_runs: " << options.warmup_runs
            << ", num_threads: " << options.num_threads << "\n\n";
  std::cout << std::left << std::setw(28) << "metric" << std::right
            << std::setw(14) << "rvv" << std::setw(14) << "no_rvv"
            << std::setw(10) << "unit" << std::setw(12) << "speedup\n";
  std::cout << std::string(78, '-') << "\n";
  PrintRow("startup", rvv.startup_ms, no_rvv.startup_ms, "ms", false);
  PrintRow("first_invoke", rvv.first_invoke_ms, no_rvv.first_invoke_ms, "ms",
           false);
  PrintRow("latency_avg", rvv.latency_avg_ms, no_rvv.latency_avg_ms, "ms",
           false);
  PrintRow("latency_p50", rvv.latency_p50_ms, no_rvv.latency_p50_ms, "ms",
           false);
  PrintRow("latency_p90", rvv.latency_p90_ms, no_rvv.latency_p90_ms, "ms",
           false);
  PrintRow("latency_p95", rvv.latency_p95_ms, no_rvv.latency_p95_ms, "ms",
           false);
  PrintRow("latency_p99", rvv.latency_p99_ms, no_rvv.latency_p99_ms, "ms",
           false);
  PrintRow("latency_min", rvv.latency_min_ms, no_rvv.latency_min_ms, "ms",
           false);
  PrintRow("latency_max", rvv.latency_max_ms, no_rvv.latency_max_ms, "ms",
           false);
  PrintRow("throughput", rvv.throughput_inferences_per_second,
           no_rvv.throughput_inferences_per_second, "inf/s", true);
  PrintRow("rss_after_startup", rvv.rss_after_startup_kb,
           no_rvv.rss_after_startup_kb, "KB", false);
  PrintRow("rss_after_inference", rvv.rss_after_inference_kb,
           no_rvv.rss_after_inference_kb, "KB", false);
  PrintRow("peak_rss", rvv.peak_rss_kb, no_rvv.peak_rss_kb, "KB", false);
}

void WriteJsonMetrics(std::ostream& os, const char* name,
                      const Metrics& metrics) {
  os << "  \"" << name << "\": {\n";
  os << "    \"label\": \"" << metrics.label << "\",\n";
  os << "    \"rvv_compiled\": " << (metrics.rvv_compiled ? "true" : "false")
     << ",\n";
  os << "    \"startup_ms\": " << metrics.startup_ms << ",\n";
  os << "    \"first_invoke_ms\": " << metrics.first_invoke_ms << ",\n";
  os << "    \"latency_avg_ms\": " << metrics.latency_avg_ms << ",\n";
  os << "    \"latency_p50_ms\": " << metrics.latency_p50_ms << ",\n";
  os << "    \"latency_p90_ms\": " << metrics.latency_p90_ms << ",\n";
  os << "    \"latency_p95_ms\": " << metrics.latency_p95_ms << ",\n";
  os << "    \"latency_p99_ms\": " << metrics.latency_p99_ms << ",\n";
  os << "    \"throughput_inferences_per_second\": "
     << metrics.throughput_inferences_per_second << ",\n";
  os << "    \"rss_after_startup_kb\": " << metrics.rss_after_startup_kb
     << ",\n";
  os << "    \"rss_after_inference_kb\": " << metrics.rss_after_inference_kb
     << ",\n";
  os << "    \"peak_rss_kb\": " << metrics.peak_rss_kb << "\n";
  os << "  }";
}

bool WriteComparisonJson(const std::string& path, const Options& options,
                         const Metrics& rvv, const Metrics& no_rvv) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to write JSON output: " << path << "\n";
    return false;
  }
  out << "{\n";
  out << "  \"model\": \"" << options.model_path << "\",\n";
  out << "  \"runs\": " << options.runs << ",\n";
  out << "  \"warmup_runs\": " << options.warmup_runs << ",\n";
  out << "  \"num_threads\": " << options.num_threads << ",\n";
  WriteJsonMetrics(out, "rvv", rvv);
  out << ",\n";
  WriteJsonMetrics(out, "no_rvv", no_rvv);
  out << ",\n";
  out << "  \"speedup\": {\n";
  out << "    \"latency_avg\": " << no_rvv.latency_avg_ms / rvv.latency_avg_ms
      << ",\n";
  out << "    \"throughput\": "
      << rvv.throughput_inferences_per_second /
             no_rvv.throughput_inferences_per_second
      << "\n";
  out << "  }\n";
  out << "}\n";
  return true;
}

int RunCompare(int argc, char** argv, const Options& options) {
  const std::string self = argv[0];
  const std::string no_rvv_binary =
      DirName(self) + "/" + BaseName(self) + "-norvv";
  const std::string rvv_result = MakeTempFile("_rvv.txt");
  const std::string no_rvv_result = MakeTempFile("_norvv.txt");

  int exit_code =
      RunProcess(options, self, ChildArgs(options, "rvv", rvv_result));
  if (exit_code != 0) {
    return exit_code;
  }
  exit_code = RunProcess(options, no_rvv_binary,
                         ChildArgs(options, "no_rvv", no_rvv_result));
  if (exit_code != 0) {
    return exit_code;
  }

  Metrics rvv;
  Metrics no_rvv;
  if (!LoadMetricsFile(rvv_result, &rvv) ||
      !LoadMetricsFile(no_rvv_result, &no_rvv)) {
    return 1;
  }
  std::remove(rvv_result.c_str());
  std::remove(no_rvv_result.c_str());

  PrintComparison(options, rvv, no_rvv);
  if (!options.json_output.empty() &&
      !WriteComparisonJson(options.json_output, options, rvv, no_rvv)) {
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (options.mode == "compare") {
    return RunCompare(argc, argv, options);
  }
  if (options.mode == "run") {
    Metrics metrics;
    if (!RunBenchmark(options, &metrics)) {
      return 1;
    }
    if (!options.result_file.empty() &&
        !WriteMetricsFile(options.result_file, metrics)) {
      return 1;
    }
    return 0;
  }

  std::cerr << "Unsupported --mode=" << options.mode << "\n";
  return 1;
}
