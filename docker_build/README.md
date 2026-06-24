# Building with Docker

This repository provides a Docker-based hermetic build environment that
automatically configures all necessary dependencies for building the project
without requiring manual configuration or setup. It handles both git submodule
initialization and project configuration in a single step.

## Prerequisites

- Docker installed on your machine
- Docker Compose (optional, for using docker-compose.yml)

## Building with the Docker Script

1. Clone this repository
2. Run the build script:
   ```
   ./build_with_docker.sh
   ```

This will:

- Build a Docker image with all necessary dependencies
- Run the container, mounting the current litert checkout directory
- Generate the configuration file (.litert_configure.bazelrc)
- Build a target. We use `//litert/runtime:compiled_model` as an example

## Building with Docker Compose

Alternatively, you can use Docker Compose:

```
docker-compose up
```

## Customizing the Build

To build different targets, you can either:

1. Modify the `hermetic_build.Dockerfile` and change the CMD line
2. Modify the command in `docker-compose.yml`
3. Pass a custom command when running Docker:
   ```
   # Run this from the repository root.
   docker run --rm --user $(id -u):$(id -g) -v $(pwd):/litert_build litert_build_env bash -c "bazel build //litert/your_custom:target"
   ```

## Accessing Build Artifacts

Copy artifacts out of the container:
```
docker cp <container>:/litert_build/bazel-bin/<path> .
```
(`litert_build_container` is the name used by `build_with_docker.sh`. Use
`docker ps -a` to find the name for Docker Compose.)

To browse outputs from inside a container shell, run (from the repo root):
```
docker run --rm -it --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) -v $(pwd):/litert_build litert_build_env bash
```

## Building RISC-V 64-bit Targets

The Android/Bazel image in `hermetic_build.Dockerfile` is separate from the
RISC-V cross-compilation image. By default, the RISC-V wrapper builds the
current LiteRT CMake project under `litert/`:

```
./docker_build/build_riscv64_with_docker.sh --clean
```

This command:

- Builds the `litert_riscv64_build_env` Docker image from
  `docker_build/riscv64_build.Dockerfile`
- Configures CMake with `docker_build/cmake/riscv64-linux-gnu.cmake`
- Configures `litert/` for `riscv64` Linux with GPU/NPU/XNNPACK/RVV disabled by
  default
- Cross-compiles the `run_model` target
- Verifies the executable with `file` and `riscv64-linux-gnu-readelf`
- Runs `run_model --help` under `qemu-riscv64`

Useful variants:

```
# Reuse an existing image and rebuild the default LiteRT target.
./docker_build/build_riscv64_with_docker.sh --use_existing_image

# Build a different LiteRT CMake target.
./docker_build/build_riscv64_with_docker.sh --use_existing_image --target analyze_model

# Build without running QEMU.
./docker_build/build_riscv64_with_docker.sh --use_existing_image --no-qemu

# Build and QEMU-test the lightweight smoke target instead of litert/run_model.
./docker_build/build_riscv64_with_docker.sh --mode smoke --use_existing_image

# Try the RVV path explicitly. This requires compiler intrinsics compatible
# with the RVV code in the tree.
TFLITE_ENABLE_RVV=ON RISCV64_MARCH=rv64gcv \
  LITERT_RISCV64_QEMU_CPU=rv64,v=true,vlen=256,elen=64 \
  ./docker_build/build_riscv64_with_docker.sh --mode smoke --use_existing_image
```

LiteRT project build outputs are left under `build-riscv64-litert/` on the
host. Smoke build outputs are left under `build-riscv64/`.

## How It Works

The Docker environment:
1. Sets up a Ubuntu 24.04 build environment (with newer libc/libc++)
2. Installs Bazel 7.4.1 and necessary build tools
3. Configures Android SDK and NDK with the correct versions
4. Automatically initializes and updates git submodules
5. Automatically generates the .litert_configure.bazelrc file
6. Provides a hermetic build environment independent of your local setup

## Troubleshooting

If you encounter build errors:

1. Check that your Docker daemon has sufficient RAM and CPU allocated
2. Ensure you have proper permissions to mount the current directory
3. Check the Docker logs for any specific error messages

You can run a shell in the container for debugging (from the repo root):
```
docker run --rm -it --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) -v $(pwd):/litert_build litert_build_env bash
```
