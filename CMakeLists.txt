cmake_minimum_required(VERSION 3.18)
project(FlashICP LANGUAGES CXX)

option(USE_CUDA "Build the CUDA voxel-downsample path" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SOURCES src/main.cpp)

# CUDA only if a compiler is present; the CPU-only build works without it.
include(CheckLanguage)
check_language(CUDA)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  list(APPEND SOURCES src/voxel_gpu.cu)
  message(STATUS "FlashICP: CUDA enabled")
else()
  message(STATUS "FlashICP: CUDA disabled (CPU baseline only)")
endif()

add_executable(flashicp ${SOURCES})
target_include_directories(flashicp PRIVATE src)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  target_compile_definitions(flashicp PRIVATE FLASHICP_CUDA)
  # 87 = Jetson AGX Orin (Ampere). Change per target GPU.
  set_target_properties(flashicp PROPERTIES CUDA_ARCHITECTURES "87")
endif()
