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
IMAGE="${LITERT_RISCV64_IMAGE:-litert_riscv64_build_env}"
SKIP_IMAGE_BUILD=0
MODE="litert"
DOCKER_BUILD_ARGS=()
CONTAINER_ARGS=()

usage() {
  cat <<'EOF'
Usage: docker_build/build_riscv64_with_docker.sh [host options] [build options]

Host options:
  --mode MODE             Build mode: litert or smoke. Default: litert.
  --use_existing_image     Skip docker build and reuse litert_riscv64_build_env.
  --docker-build-arg ARG   Extra argument forwarded to docker build. Repeatable.
  -h, --help               Show this help.

Build options are forwarded to the selected in-container build script, e.g.:
  --clean
  --target run_model
  --no-qemu
  --cmake-arg -DTFLITE_ENABLE_RVV=OFF

Examples:
  docker_build/build_riscv64_with_docker.sh --clean
  docker_build/build_riscv64_with_docker.sh --mode smoke --clean
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --use_existing_image)
      SKIP_IMAGE_BUILD=1
      shift
      ;;
    --docker-build-arg)
      DOCKER_BUILD_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      CONTAINER_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${MODE}" in
  litert)
    CONTAINER_SCRIPT="./docker_build/riscv64_build_litert.sh"
    ;;
  smoke)
    CONTAINER_SCRIPT="./docker_build/riscv64_build_and_test.sh"
    ;;
  *)
    echo "Error: unsupported --mode '${MODE}'. Expected 'litert' or 'smoke'." >&2
    usage >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed or not in PATH" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running" >&2
  exit 1
fi

if [[ "${SKIP_IMAGE_BUILD}" -eq 0 ]]; then
  if [[ "${#DOCKER_BUILD_ARGS[@]}" -gt 0 ]]; then
    docker build \
      -t "${IMAGE}" \
      -f "${REPO_ROOT}/docker_build/riscv64_build.Dockerfile" \
      "${DOCKER_BUILD_ARGS[@]}" \
      "${REPO_ROOT}"
  else
    docker build \
      -t "${IMAGE}" \
      -f "${REPO_ROOT}/docker_build/riscv64_build.Dockerfile" \
      "${REPO_ROOT}"
  fi
else
  echo "Using existing Docker image '${IMAGE}'"
fi

DOCKER_RUN_CMD=(
  docker run --rm
  --security-opt seccomp=unconfined
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp/litert_docker_home
  -e USER="$(id -un)"
  -v "${REPO_ROOT}:/litert_build"
  -w /litert_build
  "${IMAGE}"
  "${CONTAINER_SCRIPT}"
)

if [[ "${#CONTAINER_ARGS[@]}" -gt 0 ]]; then
  "${DOCKER_RUN_CMD[@]}" "${CONTAINER_ARGS[@]}"
else
  "${DOCKER_RUN_CMD[@]}"
fi
