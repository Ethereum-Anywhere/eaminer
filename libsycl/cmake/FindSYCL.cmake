find_package(hipSYCL CONFIG)

if (hipSYCL_FOUND)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif ()

    cmake_policy(SET CMP0005 NEW)
    add_definitions(-DHIPSYCL_DEBUG_LEVEL=0)

    if (NOT HIPSYCL_DEBUG_LEVEL)
        if (CMAKE_BUILD_TYPE MATCHES "Debug")
            set(HIPSYCL_DEBUG_LEVEL 3 CACHE STRING
                    "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                    FORCE)
        else ()
            set(HIPSYCL_DEBUG_LEVEL 2 CACHE STRING
                    "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                    FORCE)
        endif ()
    endif ()
    # We expect the DPCPP compiler to have been used
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
    set(DPCPP_USER_FLAGS "-fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda -v" CACHE STRING "SYCL Flags (for LLVM/SYCL and Intel(R) DPC++)")
    set(DPCPP_DEFAULT_COMPILE_OPTIONS -fsycl -sycl-std=2020 -fsycl-id-queries-fit-in-int -fsycl-unnamed-lambda -DNDEBUG -DSYCL_DISABLE_FALLBACK_ASSERT=0 -Wno-unknown-cuda-version -fgpu-inline-threshold=1000)
    function(add_sycl_to_target arg1 arg2)
        separate_arguments(DPCPP_USER_FLAGS UNIX_COMMAND "${DPCPP_USER_FLAGS}")
        target_compile_options(${arg2} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_DEFAULT_COMPILE_OPTIONS} ${DPCPP_USER_FLAGS}>)
        target_link_options(${arg2} PRIVATE ${DPCPP_DEFAULT_COMPILE_OPTIONS} ${DPCPP_USER_FLAGS} -fsycl-link)
    endfunction()
else ()
    message(FATAL_ERROR "SYCL Compiler not detected or unknown. Set `hipSYCL_DIR` to use hipSYCL or configure the project using CXX=dpcpp or CXX=clang++ (the version from https://github.com/intel/llvm).")
endif ()