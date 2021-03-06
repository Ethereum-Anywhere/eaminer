# Copyright (C) 1883 Thomas Edison - All Rights Reserved
# You may use, distribute and modify this code under the
# terms of the GPLv3 license, which unfortunately won't be
# written for another century.
#
# You should have received a copy of the LICENSE file with
# this file. 

cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE eaminer)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

find_package(Boost REQUIRED COMPONENTS program_options)

target_link_libraries(eaminer PRIVATE eth pool dev eaminer-buildinfo Boost::system Boost::thread Boost::program_options ethash OpenSSL::SSL jsoncpp_static)
if (ETHDBUS)
    find_package(PkgConfig)
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/lib/x86_64-linux-gnu/pkgconfig")
    pkg_check_modules(DBUS dbus-1)
    include_directories(${DBUS_INCLUDE_DIRS})
    link_directories(${DBUS_LIBRARY_DIRS})
    target_link_libraries(eaminer PRIVATE ${DBUS_LIBRARIES})
endif ()

if (APPLE AND ETHASHSYCL)
    include(../libsycl/cmake/FindSYCL.cmake)
    add_sycl_to_target(TARGET ${EXECUTABLE})
endif ()

if (APICORE)
    target_link_libraries(eaminer PRIVATE api)
endif ()

include(GNUInstallDirs)
install(TARGETS eaminer DESTINATION ${CMAKE_INSTALL_BINDIR})
if (MSVC)
    install(FILES $<TARGET_PDB_FILE:eaminer> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
endif ()
