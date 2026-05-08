# RISC-V 64-bit cross-compilation toolchain file for LiteRT
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=path/to/riscv64-linux-gnu.cmake

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Specify the cross-compiler
set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# Specify the sysroot (adjust path as needed)
set(CMAKE_SYSROOT /opt/riscv/sysroot)

# Specify the search paths for libraries and headers
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RISC-V specific compiler flags
# Enable RV64GC with Vector extension (RVV 1.0)
set(CMAKE_C_FLAGS_INIT "-march=rv64gcv -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gcv -mabi=lp64d")

# Set the target architecture for LiteRT
set(TFLITE_RISCV64 ON)
set(TFLITE_ENABLE_RVV ON)
