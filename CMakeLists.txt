cmake_minimum_required(VERSION 3.14)
project(template)

add_compile_definitions(DEBUG)
add_compile_definitions(_GLIBCXX_DEBUG)
set(CMAKE_CXX_STANDARD 17)

add_executable(template main.cpp)