/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include "include_sycl.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>


// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
inline constexpr int MAX_SEARCH_RESULTS = 4;

inline constexpr int ACCESSES = 64;
inline constexpr int THREADS_PER_HASH = (128 / 16);

inline constexpr int ETHASH_DATASET_PARENTS = 256;
inline constexpr int NODE_WORDS = (64 / 4);

inline constexpr int PARALLEL_HASH = 8;

inline static double target_batch_time = 0.5;

struct Search_results {
    uint32_t solCount;
    uint32_t hashCount;
    uint32_t done;
    uint32_t gid[MAX_SEARCH_RESULTS];
};

typedef struct {
    sycl::uint4 uint4s[32U / sizeof(sycl::uint4)];
} hash32_t;

typedef union {
    uint32_t words[128U / sizeof(uint32_t)];
    sycl::uint2 uint2s[128U / sizeof(sycl::uint2)];
    sycl::uint4 uint4s[128U / sizeof(sycl::uint4)];
} hash128_t;

typedef union {
    uint32_t words[64U / sizeof(uint32_t)];
    sycl::uint2 uint2s[64U / sizeof(sycl::uint2)];
    sycl::uint4 uint4s[64U / sizeof(sycl::uint4)];
} hash64_t;

struct sycl_device_task {
    Search_results* res = nullptr;
    sycl::event e{};
    inline Search_results get_result(sycl::queue& q) {
        Search_results local{};
        q.memcpy(&local, res, sizeof(Search_results), e).wait();
        e = sycl::event{};
        return local;
    }
};


struct sycl_ethash_search_kernel_tag {};

struct sycl_ethash_calculate_dag_item_kernel_tag {};

