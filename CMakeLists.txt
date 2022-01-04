# Copyright (C) 1883 Thomas Edison - All Rights Reserved
# You may use, distribute and modify this code under the
# terms of the GPLv3 license, which unfortunately won't be
# written for another century.
#
# You should have received a copy of the LICENSE file with
# this file. 

if (UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -ftree-vectorize -march=native")
endif ()


cmake_minimum_required(VERSION 3.5)

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)
include(CableBuildType)
include(CableToolchains)

cable_configure_toolchain(DEFAULT cxx17)


project(eaminer)
set(PROJECT_VERSION 1.3.15) # Update version

cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release RelWithDebInfo)

option(ETHASHCL "Build with OpenCL mining" OFF)
option(ETHASHCUDA "Build with CUDA mining" OFF)
option(ETHASHSYCL "Build with SYCL mining" OFF)
option(ETHASHCPU "Build with CPU mining (only for development)" OFF)
option(APICORE "Build with API Server support" ON)
option(DEVBUILD "Log developer metrics" OFF)

# propagates CMake configuration options to the compiler
function(configureProject)
    if (ETHASHCL)
        add_definitions(-DETH_ETHASHCL)
    endif ()
    if (ETHASHCUDA)
        add_definitions(-DETH_ETHASHCUDA)
    endif ()
    if (ETHASHSYCL)
        add_definitions(-DETH_ETHASHSYCL)
        set(DPCPP_FLAGS "-fsycl-targets=spir64,nvptx64-nvidia-cuda -Wno-unknown-cuda-version -fsycl-id-queries-fit-in-int -Xcuda-ptxas -v" CACHE STRING "SYCL Flags (for LLVM/SYCL and Intel(R) DPC++)")
    endif ()
    if (ETHASHCPU)
        add_definitions(-DETH_ETHASHCPU)
    endif ()
    if (APICORE)
        add_definitions(-DAPI_CORE)
    endif ()
    if (DEVBUILD)
        add_definitions(-DDEV_BUILD)
    endif ()
endfunction()

#hunter_add_package(Boost COMPONENTS system filesystem thread program_options)
find_package(Boost COMPONENTS system filesystem thread program_options REQUIRED)

find_package(OpenSSL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_directories(${JSONCPP_LIBRARY_DIRS})
#find_library(jsoncpp REQUIRED)

configureProject()

message("----------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("----------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                      ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                        ${ETHASHCUDA}")
message("-- ETHASHSYCL       Build SYCL components                        ${ETHASHSYCL}")
message("-- ETHASHCPU        Build CPU components (only for development)  ${ETHASHCPU}")
message("-- APICORE          Build API Server components                  ${APICORE}")
message("-- DEVBUILD         Build with dev logging                       ${DEVBUILD}")
message("----------------------------------------------------------------------------")
message("")

include(EthCompilerSettings)

cable_add_buildinfo_library(PROJECT_NAME ${PROJECT_NAME})

add_subdirectory(libdev)
add_subdirectory(libeth)
add_subdirectory(libhwmon)
add_subdirectory(libpool)
add_subdirectory(ethash)

if (ETHASHCL)
    add_subdirectory(libcl)
endif ()
if (ETHASHCUDA)
    add_subdirectory(libcuda)
endif ()
if (ETHASHSYCL)
    add_subdirectory(libsycl)
endif ()
if (ETHASHCPU)
    add_subdirectory(libcpu)
endif ()
if (APICORE)
    add_subdirectory(libapi)
endif ()

add_subdirectory(eaminer)

if (WIN32)
    set(CPACK_GENERATOR ZIP)
else ()
    set(CPACK_GENERATOR TGZ)
endif ()
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)
