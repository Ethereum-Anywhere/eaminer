include(cmake/FindSYCL.cmake)

add_library(sycl SHARED SYCLMiner.cpp SYCLMiner.h ethash_sycl_miner_kernel.cpp ethash_sycl_miner_kernel.h keccak.hpp sycl_helpers.hpp)
add_sycl_to_target(TARGET sycl SOURCES ethash_sycl_miner_kernel.cpp)
target_link_libraries(sycl PUBLIC Boost::thread ethash)
target_include_directories(sycl PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
