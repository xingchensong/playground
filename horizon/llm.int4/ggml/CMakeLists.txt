cmake_minimum_required(VERSION 3.3)
project(ggml VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS "on")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)
find_package(Threads REQUIRED)
add_compile_definitions(_GNU_SOURCE)

# ggml
add_subdirectory(src)

# mnist
set(TEST_TARGET mnist)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

# mnist-quantize
set(TEST_TARGET mnist-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
