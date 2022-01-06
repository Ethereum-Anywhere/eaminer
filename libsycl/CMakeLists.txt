include(cmake/FindSYCL.cmake)

add_library(sycl_backend SHARED ethash_sycl_miner_kernel.cpp ethash_sycl_miner_kernel.h keccak.hpp sycl_helpers.hpp)
add_sycl_to_target(TARGET sycl_backend SOURCES ethash_sycl_miner_kernel.cpp)

add_library(sycl STATIC SYCLMiner.h SYCLMiner.cpp ethash_sycl_miner_kernel.h)
add_sycl_to_target(TARGET sycl SOURCES SYCLMiner.cpp)
target_link_libraries(sycl PUBLIC sycl_backend Boost::thread ethash)
target_include_directories(sycl PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
