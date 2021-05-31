cmake_minimum_required(VERSION 3.14)
project(template)

add_compile_definitions(DEBUG)
add_compile_definitions(_GLIBCXX_DEBUG)
set(CMAKE_CXX_STANDARD 17)

# Add -O0 to remove optimizations when using gcc
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O2")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

add_executable(template main.cpp)