cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")

project(skembed)

find_package(faiss CONFIG REQUIRED)
find_package(llama CONFIG REQUIRED)

add_executable(skembed
        include/model_initialization_holder.h
        include/index.h
        include/abstract/abstract_index.h
        include/abstract/abstract_model_initialization_holder.h
        include/abstract/abstract_model.h
        include/model.h
        src/main.cpp
        src/model_initialization_holder.cpp
        src/index.cpp
        src/model.cpp
)

target_include_directories(skembed PRIVATE include)

target_link_libraries(skembed PRIVATE faiss_avx2 llama_common)