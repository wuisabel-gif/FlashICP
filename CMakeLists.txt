cmake_minimum_required(VERSION 3.18)
project(FlashICP LANGUAGES CXX)

option(USE_CUDA "Build the CUDA voxel and correspondence paths" ON)
option(BUILD_TESTING "Build FlashICP tests" ON)
set(FLASHICP_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures to build (for example 87, 86; empty uses compiler default)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SOURCES src/registration.cpp)
set(FLASHICP_HAVE_CUDA OFF)
include(CheckLanguage)
check_language(CUDA)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(FLASHICP_HAVE_CUDA ON)
  list(APPEND SOURCES src/voxel_gpu.cu src/corr_gpu.cu)
  message(STATUS "FlashICP: CUDA enabled")
else()
  message(STATUS "FlashICP: CUDA disabled (CPU baseline only)")
endif()

add_library(flashicp_core STATIC ${SOURCES})
target_include_directories(flashicp_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(flashicp_core PUBLIC cxx_std_17)

if(FLASHICP_HAVE_CUDA)
  target_compile_definitions(flashicp_core PUBLIC FLASHICP_CUDA)
  set_target_properties(flashicp_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${FLASHICP_CUDA_ARCHITECTURES}")
endif()

add_executable(flashicp src/main.cpp)
target_link_libraries(flashicp PRIVATE flashicp_core)
target_include_directories(flashicp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(CTest)
if(BUILD_TESTING)
  function(flashicp_enable_test_assertions target)
    if(MSVC)
      target_compile_options(${target} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${target} PRIVATE -UNDEBUG)
    endif()
  endfunction()

  add_executable(test_corr tools/test_corr.cpp)
  target_link_libraries(test_corr PRIVATE flashicp_core)
  target_include_directories(test_corr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  flashicp_enable_test_assertions(test_corr)
  add_test(NAME flashicp_corr_cpu COMMAND test_corr)

  add_executable(test_primitives tests/test_primitives.cpp)
  target_link_libraries(test_primitives PRIVATE flashicp_core)
  target_include_directories(test_primitives PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  flashicp_enable_test_assertions(test_primitives)
  add_test(NAME flashicp_primitives_cpu COMMAND test_primitives)
endif()
