# Set necessary compile and link flags

include(EthCheckCXXFlags)


# C++11 check and activation

message("Compiler is ${CMAKE_CXX_COMPILER_ID}")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=parentheses")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING")

    eth_add_cxx_compiler_flag_if_supported(-ffunction-sections)
    eth_add_cxx_compiler_flag_if_supported(-fdata-sections)
    eth_add_cxx_compiler_flag_if_supported(-flto)
    eth_add_cxx_linker_flag_if_supported(-pthread)
    eth_add_cxx_linker_flag_if_supported(-static-libstdc++)
    eth_add_cxx_linker_flag_if_supported(-Wl,--gc-sections)
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -Wl,-Map=eaminer.map")

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra -Wno-unknown-warning-option")

    if (ETHASHSYCL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wno-linker-warnings")
    endif ()

    eth_add_cxx_compiler_flag_if_supported(-ffunction-sections)
    eth_add_cxx_compiler_flag_if_supported(-fcolor-diagnostics)
    eth_add_cxx_compiler_flag_if_supported(-fdata-sections)
    eth_add_cxx_linker_flag_if_supported(-Wl,--gc-sections)
    eth_add_cxx_linker_flag_if_supported(-pthread)

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

    # declare Windows Vista requirement
    # undefine windows.h MAX & MIN macros because they conflict with std::min & std::max functions
    # disable unsafe CRT Library functions warnings
    add_definitions(/D_WIN32_WINNT=0x0600 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /D_SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING)

    # enable parallel compilation
    # specify Exception Handling Model
    # enable LTCG for faster builds
    # disable unknown pragma warnings (C4068)
    # disable conversion from 'size_t' to 'type', possible loss of data (C4267)
    # disable C++ exception specification ignored except to indicate a function is not __declspec(nothrow) (C4290)
    add_compile_options(/MP /EHsc /GL /wd4068 /wd4267 /wd4290 /std:c++17)

    # enable LTCG for faster builds
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG")

    # enable LTCG for faster builds
    # enable unused references removal
    # enable RELEASE so that the executable file has its checksum set
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG /OPT:REF /OPT:ICF /RELEASE")
else ()
    message(WARNING "Your compiler is not tested, if you run into any issues, we'd welcome any patches.")
endif ()

set(SANITIZE NO CACHE STRING "Instrument build with provided sanitizer")
if (SANITIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${SANITIZE}")
endif ()
