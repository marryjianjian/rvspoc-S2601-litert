# RISC-V 64-bit cross-compilation toolchain file for LiteRT (using Clang)
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=path/to/riscv64-linux-clang.cmake
#
# Prerequisites:
#   - Clang with RISC-V target support (usually available in recent versions)
#   - RISC-V sysroot (adjust CMAKE_SYSROOT below)
#
# Note: This toolchain uses Clang for cross-compilation instead of GCC.
#       Clang's integrated assembler and linker should work well for RISC-V.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Specify the cross-compiler (Clang)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Use LLVM's lld as the linker (host system's ld does not support RISC-V)
# Install lld: apt install lld  OR  brew install lld
#
# Important: We do NOT use CMAKE_SYSROOT because lld has a known issue where it
# prepends the sysroot prefix to ALL absolute paths in linker scripts, causing
# doubled paths (e.g. /sysroot/sysroot/lib/libc.so.6). Instead:
#   - Compiler gets --sysroot=<path> via CMAKE_C/CXX_FLAGS (for header search)
#   - Linker gets --sysroot=/ via CMAKE_EXE_LINKER_FLAGS (no-op, absolute paths stay as-is)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld --sysroot=/ -L/usr/lib/gcc-cross/riscv64-linux-gnu/14 -L/usr/riscv64-linux-gnu/lib")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld --sysroot=/ -L/usr/lib/gcc-cross/riscv64-linux-gnu/14 -L/usr/riscv64-linux-gnu/lib")

# Specify the target triple for RISC-V 64-bit Linux
set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

# Specify the search paths for libraries and headers (CMake find commands)
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RISC-V specific compiler flags
# Enable RV64GC with Vector extension (RVV 1.0)
# -B path: GCC cross-compilation startup files (crtbeginS.o, crtendS.o, libgcc.a)
# --sysroot: for header search only (linker uses --sysroot=/ from LINKER_FLAGS)
set(CMAKE_C_FLAGS_INIT "-march=rv64gcv -mabi=lp64d -B/usr/lib/gcc-cross/riscv64-linux-gnu/14 --sysroot=/usr/riscv64-linux-gnu")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gcv -mabi=lp64d -B/usr/lib/gcc-cross/riscv64-linux-gnu/14 --sysroot=/usr/riscv64-linux-gnu -I/usr/riscv64-linux-gnu/include/c++/14 -I/usr/riscv64-linux-gnu/include/c++/14/riscv64-linux-gnu")

# Set the target architecture for LiteRT
set(TFLITE_RISCV64 ON)
set(TFLITE_ENABLE_RVV ON)
