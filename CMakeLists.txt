if(MSVC)
    cmake_minimum_required(VERSION 3.4)
else()
    cmake_minimum_required(VERSION 2.8)
endif()
project(scalagbm)
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

find_package(CUDA REQUIRED QUIET)
find_package(OpenMP REQUIRED QUIET)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# CUDA 11
if(CUDA_VERSION VERSION_LESS "11.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo --expt-extended-lambda --default-stream per-thread")
if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++14 -lineinfo --expt-extended-lambda --default-stream per-thread -arch=compute_86")
endif()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# for easylogging++ configuration
add_definitions("-DELPP_FEATURE_PERFORMANCE_TRACKING")
add_definitions("-DELPP_THREAD_SAFE")
add_definitions("-DELPP_STL_LOGGING")
add_definitions("-DELPP_NO_LOG_TO_FILE") 
# includes
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cub)
include_directories(${COMMON_INCLUDES})

add_subdirectory(src/thundergbm)


# configuration file
set(DATASET_DIR ${PROJECT_SOURCE_DIR}/datasets/)
configure_file(include/thundergbm/config.h.in config.h)

