cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")

project(skembed)

find_package(faiss CONFIG REQUIRED)
find_package(Llama CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(zxorm CONFIG REQUIRED)
find_package(csv-parser CONFIG REQUIRED)
find_path(BEXT_DI_INCLUDE_DIRS "boost/di.hpp")

add_executable(skembed
        include/abstract/abstract_index.h
        include/abstract/abstract_model_backend.h
        include/abstract/abstract_model.h
        include/model_backend.h
        include/model.h
        include/flat_embed.h
        include/index_of_embeddings.h
        include/nearest_neighbor_index.h
        src/main.cpp
        src/model_backend.cpp
        src/index_of_embeddings.cpp
        src/model.cpp
        src/nearest_neighbor_index.cpp
        src/flat_embed.cpp
)

target_include_directories(skembed PRIVATE include ${BEXT_DI_INCLUDE_DIRS})

target_link_libraries(skembed PRIVATE faiss_avx2 llama_common spdlog::spdlog zxorm::zxorm  csv-parser)