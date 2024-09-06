cmake_minimum_required(VERSION 3.14)
SITE_NAME(HOSTNAME)

project(sparce_llm_cache_library CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(COMMON_FLAGS " -O0 -g ")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(COMMON_FLAGS " -O3 -g ")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
# no-as-needed is necessary because gurobi95 is not directly used by moe_cache
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  ${COMMON_FLAGS} -fPIC -Wall -fopenmp -march=native    -D_GLIBCXX_USE_CXX11_ABI=0 -DUNIT_TEST -DPSTL_USE_PARALLEL_POLICIES=0 -Wl,--no-as-needed")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_FLAGS} --compiler-options '-fopenmp -fPIC '  -D_GLIBCXX_USE_CXX11_ABI=0  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80")
message(DEBUG "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(DEBUG "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
include_directories(
  ${PROJECT_SOURCE_DIR}/src/cpp_worker
  /usr/include/python3.10
  /usr/local/lib/python3.10/dist-packages/torch/include
  /usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${PROJECT_SOURCE_DIR}/3rdparty/json/single_include
)

# source scripts already adds necessary path to LD path
link_directories(
  # $ENV{GUROBI_HOME}/lib
  # $ENV{CONDA_PREFIX}/lib
  # /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8
  /usr/local/cuda/lib64
)


file(GLOB_RECURSE CACHE_SRC
  src/cpp_worker/*.cc
  src/cpp_worker/*.cpp
  src/cpp_worker/*.cu)

# build samgraph library
add_library(moe_cache
  SHARED
  ${CACHE_SRC}
)
target_link_libraries(moe_cache
  PUBLIC
  cuda
  cudart
  nvToolsExt
  python3.10
  )
