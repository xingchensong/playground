cmake_minimum_required(VERSION 3.3)
project(ggml VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS "on")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c")
find_package(Threads REQUIRED)
add_compile_definitions(_GNU_SOURCE)

# mnist
set(TEST_TARGET mnist)
add_executable(${TEST_TARGET} main.cpp)
