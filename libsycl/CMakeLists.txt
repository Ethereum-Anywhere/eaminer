include(cmake/FindSYCL.cmake)

file(GLOB sources "*.cpp")
file(GLOB headers "*.h" "*.hpp")

add_library(sycl STATIC ${sources} ${headers})
add_sycl_to_target(TARGET sycl SOURCES ${sources})
target_link_libraries(sycl PUBLIC eth Boost::thread ethash)
target_include_directories(sycl PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
