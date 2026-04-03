# --------------------------------------------------------------------------
# Tensio — Tensor I/O Library
# Single-header C99 library for Safetensors, GGUF, and TQ formats
# --------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)
project(tensio VERSION 0.1.0 LANGUAGES C)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(TENSIO_ENABLE_VTABLES   "Enable SQLite virtual tables"              ON)
option(TENSIO_ENABLE_BUNDLE    "Enable tar bundle support"                 ON)
option(TENSIO_ENABLE_IO_URING  "Enable io_uring async backend (Linux)"     OFF)
option(TENSIO_ENABLE_STB_IMAGE "Enable stb_image_write support"            ON)
option(TENSIO_ENABLE_LZ4       "Enable LZ4 compression for TQ format"      ON)
option(TENSIO_ENABLE_NEON      "Enable ARM NEON optimizations for TQ"      ON)
option(TENSIO_BUILD_TESTS      "Build test suite"                          OFF)
option(TENSIO_BUILD_BENCHMARKS "Build benchmarks"                          OFF)
option(TENSIO_BUILD_EXAMPLES   "Build examples"                            OFF)

# --------------------------------------------------------------------------
# FetchContent — external dependencies
# --------------------------------------------------------------------------
include(FetchContent)

# yyjson (always required — used by safetensors JSON parser)
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG        0.9.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(yyjson)

# SQLite3 (system package, only when vtables are enabled)
if(TENSIO_ENABLE_VTABLES)
    find_package(SQLite3 REQUIRED)
endif()

# LZ4 (fetched from source, only when LZ4 compression is enabled)
if(TENSIO_ENABLE_LZ4)
    FetchContent_Declare(
        lz4
        GIT_REPOSITORY https://github.com/lz4/lz4.git
        GIT_TAG        v1.10.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_Populate(lz4)

    # lz4 does not provide a top-level CMakeLists.txt, so build manually
    # Include both the core lz4 and the frame API
    add_library(lz4 STATIC
        ${lz4_SOURCE_DIR}/lib/lz4.c
        ${lz4_SOURCE_DIR}/lib/lz4.h
        ${lz4_SOURCE_DIR}/lib/lz4frame.c
        ${lz4_SOURCE_DIR}/lib/lz4frame.h
        ${lz4_SOURCE_DIR}/lib/lz4hc.c
        ${lz4_SOURCE_DIR}/lib/lz4hc.h
        ${lz4_SOURCE_DIR}/lib/xxhash.c
        ${lz4_SOURCE_DIR}/lib/xxhash.h
    )
    target_include_directories(lz4 PUBLIC
        $<BUILD_INTERFACE:${lz4_SOURCE_DIR}/lib>
        $<INSTALL_INTERFACE:include>
    )
endif()

# --------------------------------------------------------------------------
# Core INTERFACE library target
# --------------------------------------------------------------------------
add_library(tensio INTERFACE)

target_include_directories(tensio INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler warnings (GNU / Clang) — propagated to all consumers
# -Werror is NOT propagated via INTERFACE; each target opts in explicitly
target_compile_options(tensio INTERFACE
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-std=c99 -pedantic -Wall -Wextra -Wno-gcc-install-dir-libstdcxx>
)

# Always link yyjson
target_link_libraries(tensio INTERFACE yyjson)

# Conditional compile definitions and link libraries
if(TENSIO_ENABLE_VTABLES)
    target_compile_definitions(tensio INTERFACE TENSIO_ENABLE_VTABLES)
    target_link_libraries(tensio INTERFACE SQLite::SQLite3)
endif()

if(TENSIO_ENABLE_BUNDLE)
    target_compile_definitions(tensio INTERFACE TENSIO_ENABLE_BUNDLE)
endif()

if(TENSIO_ENABLE_IO_URING)
    target_compile_definitions(tensio INTERFACE TENSIO_ENABLE_IO_URING)
endif()

if(TENSIO_ENABLE_STB_IMAGE)
    target_compile_definitions(tensio INTERFACE TENSIO_ENABLE_STB_IMAGE)
    target_include_directories(tensio INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    )
endif()

if(TENSIO_ENABLE_LZ4)
    target_compile_definitions(tensio INTERFACE TQ_WITH_LZ4)
    target_link_libraries(tensio INTERFACE lz4)
endif()

if(TENSIO_ENABLE_NEON)
    target_compile_definitions(tensio INTERFACE TQ_WITH_NEON)
    target_link_libraries(tensio INTERFACE m)
endif()

# --------------------------------------------------------------------------
# CLI executable
# --------------------------------------------------------------------------
add_executable(tensio_cli src/main.c)

target_compile_definitions(tensio_cli PRIVATE
    SAFETENSORS_IMPLEMENTATION
    GGUF_IMPLEMENTATION
    TQ_IMPLEMENTATION
    CONVERT_IMPLEMENTATION
    BUNDLE_IMPLEMENTATION
    _GNU_SOURCE
)

target_compile_options(tensio_cli PRIVATE -Werror)
# -Wno-unused-function must come after INTERFACE -Wall; use source property to append last
set_source_files_properties(src/main.c PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")
target_link_libraries(tensio_cli PRIVATE tensio yyjson)

if(TENSIO_ENABLE_VTABLES)
    target_compile_definitions(tensio_cli PRIVATE
        VTABLE_SAFETENSORS_IMPLEMENTATION
        VTABLE_GGUF_IMPLEMENTATION
        VTABLE_TQ_IMPLEMENTATION
    )
    target_link_libraries(tensio_cli PRIVATE SQLite::SQLite3)
endif()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(TENSIO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
if(TENSIO_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
if(TENSIO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
install(
    DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(TARGETS tensio EXPORT tensioTargets)

if(TENSIO_ENABLE_LZ4)
    install(TARGETS lz4 EXPORT tensioTargets)
endif()

install(EXPORT tensioTargets
    FILE        tensioTargets.cmake
    NAMESPACE   tensio::
    DESTINATION lib/cmake/tensio
)
