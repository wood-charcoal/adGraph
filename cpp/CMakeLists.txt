#=============================================================================
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(NV_GRAPH VERSION 0.4.0 LANGUAGES C CXX HIP)

enable_language(HIP)

find_package(HIP REQUIRED)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")  # -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for.
   Default is to compile for a pre-determined st of gpu-architectures based
   on CTK version.")

if("${GPU_ARCHS}" STREQUAL "")
  # Default ROCm GPU architectures
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a")
else()
  foreach(arch ${GPU_ARCHS})
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=${arch}")
  endforeach()
endif()

# set warnings as errors
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall")  # -Werror")

# set default build type
set(CMAKE_BUILD_TYPE "Release")

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")  # -Werror")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "nvGraph: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "nvGraph: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(NOT DEFINED NVGRAPH_LIGHT)
    set(NVGRAPH_LIGHT True)
endif(NOT DEFINED NVGRAPH_LIGHT)

###################################################################################################
# - lapack modules ---------------------------------------------------------------------------------

set(LAPACK_INSTALL_ROOT "$ENV{LAPACK_INSTALL_ROOT}" CACHE PATH "Path to LAPACK installation root")

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

# if(BUILD_TESTS)
#     include(CTest)
#     include(ConfigureGoogleTest)

#     if(GTEST_FOUND)
#         message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
#         include_directories(${GTEST_INCLUDE_DIR})
#         add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
#     else()
#         message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
#     endif(GTEST_FOUND)
# endif(BUILD_TESTS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include"
                    "${CMAKE_SOURCE_DIR}/../external"
                    # "${CMAKE_SOURCE_DIR}/../external/cusp"
                    "$ENV{DTK_ROOT}/include"
                    "$ENV{DTK_ROOT}/include/hip"
                    "${LAPACK_INSTALL_ROOT}/include"
                   )

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("$ENV{DTK_ROOT}/lib"
                 "$ENV{DTK_ROOT}/lib64"
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}"
                 "${LAPACK_INSTALL_ROOT}/lib64"
                 )

###################################################################################################
# - library targets -------------------------------------------------------------------------------
add_library(nvgraph SHARED
            thirdparty/cnmem/src/cnmem.cpp
            # src/arnoldi.cu
            # src/bfs.cu
            # src/bfs2d.cu
            # src/bfs_kernels.cu
            # src/convert.cu
            # src/csrmv.cu
            src/csr_graph.cpp
            # src/csrmv_cub.cu
            # src/graph_extractor.cpp
            # src/jaccard_gpu.cu
            # src/kmeans.cu
            # src/lanczos.cu
            # src/lobpcg.cu
            src/matrix.cpp
            # src/modularity_maximization.cu
            src/nvgraph.cpp
            src/nvgraph_hipblas.cpp
            src/nvgraph_hipsparse.cpp
            src/nvgraph_error.cpp
            src/nvgraph_lapack.cpp
            src/nvgraph_vector_kernels.cpp
            # src/size2_selector.cpp
            # src/sssp.cpp
            src/triangles_counting_kernels.cpp
            src/triangles_counting.cpp
            src/pagerank.cpp
            src/pagerank_kernels.cpp
            # src/partition.cu
            src/valued_csr_graph.cpp
            # src/widest_path.cu         
)

target_compile_definitions(nvgraph PRIVATE
    __HIP_PLATFORM_AMD__=1 
)

target_compile_options(nvgraph PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++14> 
)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

if(NVGRAPH_LIGHT MATCHES True)
    add_definitions( -DNVGRAPH_LIGHT=${NVGRAPH_LIGHT} )
endif(NVGRAPH_LIGHT MATCHES True)


###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(nvgraph 
    PRIVATE
    -Wl,--start-group 
    hipblas hipsparse hiprand hipsolver amdhip64 hip::host cblas blas lapack
    -Wl,--end-group
)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS nvgraph
        DESTINATION lib64)

install(FILES
         ${CMAKE_CURRENT_SOURCE_DIR}/include/nvgraph.h
        DESTINATION include/nvgraph)
