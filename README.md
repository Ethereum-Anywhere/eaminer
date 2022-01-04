# eaminer -- Ethereum Anywhere miner

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)][Gitter]

> Ethereum (ethash) miner with OpenCL, CUDA and stratum support

**eaminer** is an Ethash Heterogeneous mining application: you can/should be able to mine on pretty much any device any coin that relies on an Ethash Proof of Work.

This project is a fork from [nsfminer](https://github.com/no-fee-ethereum-mining/nsfminer) that was archived. This project adds a SYCL Heterogeneous backend. SYCL is a heterogeneous programming model developed by
Khronos (sort of OpenCL successor). Several SYCL implementations exist and this project was tested with hipSYCl and Intel/LLVM. These two implementations allows targeting pretty much any hardware today without having to
use through OpenCL.

## New features

* SYCL mining (tested on AMD/HIP, NVIDIA/CUDA, Intel/L0, OpenMP and OpenCL)

## Features

* OpenCL mining
* Nvidia CUDA mining
* realistic benchmarking
* stratum mining without proxy
* Automatic devices configuration
* farm failover

## Table of Contents

* [Usage](#usage)
    * [Examples connecting to pools](#examples-connecting-to-pools)
* [Build](#build)
    * [Building from source](#building-from-source)
* [API](#api)
* [Contribute](#contribute)

## Usage

**eaminer** is a command line program. This means you launch it either from a Windows command prompt or Linux console, or create shortcuts to predefined command lines using a Linux Bash script or Windows batch/cmd file.
For a full list of available command, please run:

```sh
eaminer --help
```
Complete list of command options [here](docs/Options.md).

### Examples connecting to pools

Check our [samples](docs/POOL_EXAMPLES_ETH.md) to see how to connect to different pools.

### Building from source

[Instructions](docs/BUILD.md)

## API

[Specifications](docs/API_DOCUMENTATION.md)

## Contribute

All contributions are welcome, but please format your code before!

## License

Licensed under the [GNU General Public License, Version 3](LICENSE).
