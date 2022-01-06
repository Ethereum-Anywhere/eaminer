# Building from source

## Table of Contents

* [Requirements](#requirements)
    * [Common](#common)
    * [Linux](#linux)
        * [OpenCL support on Linux](#opencl-support-on-linux)
    * [Windows](#windows)
* [CMake configuration options](#cmake-configuration-options)
* [Disable Hunter](#disable-hunter)
* [Instructions](#instructions)
    * [Windows-specific script](#windows-specific-script)


# Requirements

This project uses [CMake].

## Common

1. [CMake] >= 3.5
2. [Git](https://git-scm.com/downloads)
3. [OpenSSL] (libssl-dev)
4. [JsonCPP] (libjsoncpp-dev)
5. [boost] 



## SYCL on Linux

### Using the open source Intel/LLVM
Build a SYCL compiler using instructions from https://github.com/intel/llvm.

Then configure cmake with `CXX=clang++ CC=clang cmake path_to_eaminer -DETHASHSYCL=ON`.

Set `DPCPP_FLAGS` to configure the target before building:
* CUDA: `-fsycl-targets=nvptx64-nvidia-cuda`, you can specify an arch with `-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75` and print ptxas info with `-Xcuda-ptxas -v`
* HIP:  `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfxXXX` offload arch must be specified.
* OpenCL: with JIT: `-fsycl-targets=spir64` or with AOT: `-fsycl-targets=spir64_x86_64`. Eventually pass info to the AOT compiler with something like: `-Xsycl-target-backend=spir64_x86_64 --march=avx`. See `opencl-aot --help` for available options.
* FPGA: `-fsycl-targets=spir64-fpga`


### Using Intel(R) oneAPI/DPC++
Install the compiler from Intel and bring run `source /opt/intel/oneapi/setvars.sh`. 

Configure the cmake project with `CXX=dpcpp CC=icx cmake path_to_eaminer -DETHASHSYCL=ON`.

Set `DPCPP_FLAGS` as presented before.


### Using hipSYCL
Install hipSYCL and set the environment variable `hipSYCL_DIR` to the path of the installation and configure the wmake project with:
`cmake path_to_eaminer -DETHASHSYCL=ON -DHIPSYCL_TARGETS=...` where `DHIPSYCL_TARGETS` is set accordingly to hipSYCL doc (for OMP and CUDA it would be: `omp;cuda:sm_XX`). 


## Linux

1. GCC version >= 4.8
2. DBUS development libs if building with `-DETHDBUS`. E.g. on Ubuntu run:

```shell
sudo apt install libdbus-1-dev
```

### OpenCL support on Linux

If you're planning to use [OpenCL on Linux](https://github.com/ruslo/hunter/wiki/pkg.opencl#pitfalls)
you have to install the OpenGL libraries. E.g. on Ubuntu run:

```shell
sudo apt-get install mesa-common-dev
```

These are sufficient for Ubuntu LTS releases. Other packages may be needed depending on your distrubution.

## Windows

1. [Visual Studio 2017](https://www.visualstudio.com/downloads/); Community Edition works fine. **Make sure you install MSVC 2015 toolkit (v140).**

# Instructions

1. Make sure git submodules are up to date:

    ```shell
    git submodule update --init --recursive
    ```

2. Create a build directory:

    ```shell
    mkdir build
    cd build
    ```

3. Configure the project with CMake. Check out the additional [configuration options](#cmake-configuration-options).

    ```shell
    cmake ..
    ```

    **Note:** On Windows, it's possible to have issues with VS 2017 default compilers, due to CUDA expecting a specific
    toolset version; in that case, use:

    ```shell
    cmake .. -G "Visual Studio 15 2017 Win64"
    # or this if you have build errors in the CUDA step
    cmake .. -G "Visual Studio 15 2017 Win64" -T v140
    ```

4. Build the project using [CMake Build Tool Mode]. This is a portable variant of `make`.

    ```shell
    cmake --build .
    ```

    Note: On Windows, it is possible to have compiler issues if you don't specify the build config. In that case use:

    ```shell
    cmake --build . --config Release
    ```

5. _(Optional, Linux only)_ Install the built executable:

    ```shell
    sudo make install
    ```

### Windows-specific script

Complete sample Windows batch file - **adapt it to your system**. Assumes that:

* it's placed one folder up from the eaminer source folder
* you have CMake installed
* you have Perl installed

```bat
@echo off
setlocal

rem add MSVC in PATH
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"

rem add Perl in PATH; it's needed for OpenSSL build
set "PERL_PATH=C:\Perl\perl\bin"
set "PATH=%PERL_PATH%;%PATH%"

rem switch to eaminer's source folder
cd "%~dp0\eaminer\"

if not exist "build\" mkdir "build\"

rem For CUDA 9.x pass also `-T v140`
cmake -G "Visual Studio 15 2017 Win64" -H. -Bbuild -DETHASHCL=ON -DETHASHCUDA=ON -DAPICORE=ON ..
cd "build\"
cmake --build . --config Release --target package

endlocal
pause
```

# CMake configuration options

Pass these options to CMake configuration command, e.g.

```shell
cmake .. -DETHASHCUDA=ON -DETHASHCL=OFF
```

* `-DETHASHCL=ON` - enable OpenCL mining, `ON` by default.
* `-DETHASHCUDA=ON` - enable CUDA mining, `ON` by default.
* `-DAPICORE=ON` - enable API Server, `ON` by default.
* `-DBINKERN=ON` - install AMD binary kernels, `OFF` by default.
* `-DETHDBUS=ON` - enable D-Bus support, `OFF` by default.

## Disable Hunter

Hunter was removed.


[CMake]: https://cmake.org/
[CMake Build Tool Mode]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-tool-mode

