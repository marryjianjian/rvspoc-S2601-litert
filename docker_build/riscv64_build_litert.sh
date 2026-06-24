#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${LITERT_RISCV64_LITERT_BUILD_DIR:-${REPO_ROOT}/build-riscv64-litert}"
TOOLCHAIN_FILE="${REPO_ROOT}/docker_build/cmake/riscv64-linux-gnu.cmake"
TARGET="${LITERT_RISCV64_LITERT_TARGET:-run_model}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${LITERT_BUILD_JOBS:-$(nproc)}"
RUN_QEMU=1
QEMU_CPU="${LITERT_RISCV64_QEMU_CPU:-rv64}"
RISCV64_MARCH="${RISCV64_MARCH:-rv64gc}"
RISCV64_MABI="${RISCV64_MABI:-lp64d}"
TFLITE_ENABLE_RVV="${TFLITE_ENABLE_RVV:-OFF}"
CMAKE_EXTRA_ARGS=()

HOST_TOOLS_DIR="${TFLITE_HOST_TOOLS_DIR:-}"
if [[ -z "${HOST_TOOLS_DIR}" ]]; then
  if [[ -x /opt/flatbuffers-host/bin/flatc ]]; then
    HOST_TOOLS_DIR="/opt/flatbuffers-host/bin"
  else
    HOST_TOOLS_DIR="/usr/bin"
  fi
fi

usage() {
  cat <<'EOF'
Usage: docker_build/riscv64_build_litert.sh [options]

Options:
  --target NAME          CMake target to build. Default: run_model
  --build-dir PATH      Build directory. Default: ./build-riscv64-litert
  --clean               Remove the build directory before configuring.
  --no-qemu             Build and inspect the target, but skip QEMU execution.
  --cmake-arg ARG       Extra argument forwarded to cmake configure. Repeatable.
  -h, --help            Show this help.

Environment:
  LITERT_RISCV64_QEMU_CPU   QEMU CPU string. Default: rv64
  LITERT_BUILD_JOBS         Parallel build jobs. Default: nproc
  CMAKE_BUILD_TYPE          CMake build type. Default: Release
  RISCV64_MARCH             RISC-V -march value. Default: rv64gc
  RISCV64_MABI              RISC-V -mabi value. Default: lp64d
  TFLITE_ENABLE_RVV         Enable TFLite RVV code paths. Default: OFF
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --clean)
      rm -rf "${BUILD_DIR}"
      shift
      ;;
    --no-qemu)
      RUN_QEMU=0
      shift
      ;;
    --cmake-arg)
      CMAKE_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[riscv64-litert] Repository: ${REPO_ROOT}"
echo "[riscv64-litert] Build dir:  ${BUILD_DIR}"
echo "[riscv64-litert] Target:     ${TARGET}"
echo "[riscv64-litert] Host tools: ${HOST_TOOLS_DIR}"
echo "[riscv64-litert] ISA:        ${RISCV64_MARCH}/${RISCV64_MABI}"
echo "[riscv64-litert] RVV:        ${TFLITE_ENABLE_RVV}"

TENSORFLOW_SOURCE_ARG=()
if [[ -d "${REPO_ROOT}/build-riscv64/tensorflow-lite/tensorflow-src" ]]; then
  TENSORFLOW_SOURCE_ARG=(
    "-DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=${REPO_ROOT}/build-riscv64/tensorflow-lite/tensorflow-src"
    "-DTENSORFLOW_SOURCE_DIR=${REPO_ROOT}/build-riscv64/tensorflow-lite/tensorflow-src"
  )
fi

cmake -S "${REPO_ROOT}/litert" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DTFLITE_HOST_TOOLS_DIR="${HOST_TOOLS_DIR}" \
  -DProtobuf_PROTOC_EXECUTABLE="${Protobuf_PROTOC_EXECUTABLE:-/usr/bin/protoc}" \
  -DRISCV64_MARCH="${RISCV64_MARCH}" \
  -DRISCV64_MABI="${RISCV64_MABI}" \
  -DCMAKE_C_FLAGS="-march=${RISCV64_MARCH} -mabi=${RISCV64_MABI}" \
  -DCMAKE_CXX_FLAGS="-march=${RISCV64_MARCH} -mabi=${RISCV64_MABI}" \
  -DTFLITE_ENABLE_RVV="${TFLITE_ENABLE_RVV}" \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_NNAPI=OFF \
  -DTFLITE_ENABLE_EXTERNAL_DELEGATE=OFF \
  -DLITERT_ENABLE_GPU=OFF \
  -DLITERT_ENABLE_NPU=OFF \
  -DLITERT_DISABLE_KLEIDIAI=ON \
  "${TENSORFLOW_SOURCE_ARG[@]}" \
  "${CMAKE_EXTRA_ARGS[@]}"

cmake --build "${BUILD_DIR}" --target "${TARGET}" -j "${JOBS}"

ARTIFACT=""
if [[ -x "${BUILD_DIR}/${TARGET}" ]]; then
  ARTIFACT="${BUILD_DIR}/${TARGET}"
else
  ARTIFACT="$(find "${BUILD_DIR}" -type f -perm -111 -name "${TARGET}" | head -n 1 || true)"
fi

if [[ -z "${ARTIFACT}" || ! -f "${ARTIFACT}" ]]; then
  echo "[riscv64-litert] Built target '${TARGET}', but no executable artifact was found for QEMU." >&2
  echo "[riscv64-litert] Build output remains in ${BUILD_DIR}" >&2
  exit 0
fi

echo "[riscv64-litert] Artifact: ${ARTIFACT}"
file "${ARTIFACT}"
riscv64-linux-gnu-readelf -h "${ARTIFACT}" | sed -n '/Class:/p;/Machine:/p;/Flags:/p'

if ! file "${ARTIFACT}" | grep -q "RISC-V"; then
  echo "[riscv64-litert] Artifact is not a RISC-V executable: ${ARTIFACT}" >&2
  exit 1
fi

if [[ "${RUN_QEMU}" -eq 1 ]]; then
  echo "[riscv64-litert] Running QEMU check..."
  set +e
  qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu "${QEMU_CPU}" "${ARTIFACT}" --help >/tmp/litert_riscv64_run_model_help.txt 2>&1
  qemu_rc=$?
  set -e
  head -n 20 /tmp/litert_riscv64_run_model_help.txt
  if ! grep -q "Flags from" /tmp/litert_riscv64_run_model_help.txt; then
    echo "[riscv64-litert] QEMU check failed with exit code ${qemu_rc}." >&2
    exit 1
  fi
fi

echo "[riscv64-litert] Build completed."
