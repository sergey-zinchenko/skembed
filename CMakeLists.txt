cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")

project(skembed)

find_package(faiss CONFIG REQUIRED)
find_package(llama CONFIG REQUIRED)

add_executable(skembed main.cpp)

target_link_libraries(skembed PRIVATE faiss_avx2 llama_common)