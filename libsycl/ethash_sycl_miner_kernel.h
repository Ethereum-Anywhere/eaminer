/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "include_sycl.h"


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

struct Search_results {
    uint32_t solCount;
    uint32_t hashCount;
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


//template<int threads_per_hash_, int parallel_hash_>
struct sycl_ethash_search_kernel_tag {};

//template<int threads_per_hash_ = THREADS_PER_HASH, int parallel_hash_ = PARALLEL_HASH>
[[nodiscard]] sycl_device_task run_ethash_search(        //
        uint32_t work_groups,                            //
        uint32_t work_items,                             //
        sycl::queue q,                                   //
        sycl_device_task task,                           //
        uint64_t start_nonce,                            //
        uint64_t d_dag_num_items,                        //
        const hash128_t* __restrict const d_dag,         //
        hash32_t d_header,                               //
        uint64_t d_target,                               //
        const uint32_t* __restrict d_kill_signal_host,   //
        uint32_t* __restrict d_kill_signal_device);

size_t get_ethash_search_kernel_max_work_items(sycl::queue& q);


//extern template std::future<Search_results> run_ethash_search<THREADS_PER_HASH, PARALLEL_HASH>(   //
//        uint32_t work_groups, uint32_t work_items, sycl::queue q, uint64_t start_nonce, uint64_t d_dag_num_items, const hash128_t* d_dag, const hash32_t& d_header,
//        uint64_t d_target, const sycl::event& e = {});

struct sycl_ethash_calculate_dag_item_kernel_tag {};

[[nodiscard]] std::vector<sycl::event> ethash_generate_dag(   //
        uint64_t dag_size, uint32_t work_groups, uint32_t work_items, sycl::queue q, uint32_t d_dag_num_items, uint32_t d_light_num_items, hash128_t* d_dag,
        const hash64_t* d_light, const sycl::event& e = {});

size_t get_ethash_generate_kernel_max_work_items(sycl::queue& q);
