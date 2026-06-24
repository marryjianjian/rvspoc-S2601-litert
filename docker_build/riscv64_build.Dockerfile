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

# Docker image for reproducible riscv64 Linux cross-compilation and QEMU smoke
# testing. This is intentionally separate from hermetic_build.Dockerfile, which
# is the Android/Bazel build image.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils \
    binutils-riscv64-linux-gnu \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    file \
    g++-riscv64-linux-gnu \
    gcc-riscv64-linux-gnu \
    git \
    libc6-dev-riscv64-cross \
    ninja-build \
    pkg-config \
    protobuf-compiler \
    python3 \
    qemu-user \
    unzip \
    wget \
    zip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build a recent host flatc. Ubuntu 24.04's flatbuffers-compiler is too old
# for the TFLite schema syntax used by this tree.
ENV FLATBUFFERS_VERSION=25.9.23
RUN wget -q https://github.com/google/flatbuffers/archive/refs/tags/v${FLATBUFFERS_VERSION}.tar.gz -O /tmp/flatbuffers.tar.gz && \
    mkdir -p /tmp/flatbuffers-src /tmp/flatbuffers-build /opt/flatbuffers-host/bin && \
    tar -xzf /tmp/flatbuffers.tar.gz -C /tmp/flatbuffers-src --strip-components=1 && \
    cmake -S /tmp/flatbuffers-src -B /tmp/flatbuffers-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DFLATBUFFERS_BUILD_TESTS=OFF \
      -DFLATBUFFERS_INSTALL=OFF && \
    cmake --build /tmp/flatbuffers-build --target flatc -j "$(nproc)" && \
    cp /tmp/flatbuffers-build/flatc /opt/flatbuffers-host/bin/flatc && \
    rm -rf /tmp/flatbuffers.tar.gz /tmp/flatbuffers-src /tmp/flatbuffers-build

ENV TFLITE_HOST_TOOLS_DIR=/opt/flatbuffers-host/bin
ENV Protobuf_PROTOC_EXECUTABLE=/usr/bin/protoc
ENV PATH=/opt/flatbuffers-host/bin:${PATH}

WORKDIR /litert_build

CMD ["./docker_build/riscv64_build_and_test.sh"]
