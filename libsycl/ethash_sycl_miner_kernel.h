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
#include <future>

#if defined(__has_include)
#    if __has_include(<sycl/sycl.hpp>)
#        include <sycl/sycl.hpp>
#    else
#        include <CL/sycl.hpp>
#    endif
#else
#    include <sycl/sycl.hpp>
#endif

#if !defined(SYCL_LANGUAGE_VERSION)
#    error "You use a SYCL compiler"
#endif

#if defined(SYCL_IMPLEMENTATION_HIPSYCL)
#    define SYCL_ATOMIC_REF sycl::atomic_ref
#    define OPT_CONSTEXPR
#elif defined(SYCL_IMPLEMENTATION_ONEAPI) || defined(SYCL_IMPLEMENTATION_INTEL)
#    define OPT_CONSTEXPR constexpr
#    if defined(__INTEL_LLVM_COMPILER)
#        define SYCL_ATOMIC_REF sycl::ext::oneapi::atomic_ref
#    else
#        define SYCL_ATOMIC_REF sycl::atomic_ref
#    endif
#else
#    error "Untested SYCL implementation"
#endif


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
    uint32_t done;
    uint32_t gid[MAX_SEARCH_RESULTS];
};

typedef struct {
    sycl::uint4 uint4s[32 / sizeof(sycl::uint4)];
} hash32_t;

typedef union {
    uint32_t words[128 / sizeof(uint32_t)];
    sycl::uint2 uint2s[128 / sizeof(sycl::uint2)];
    sycl::uint4 uint4s[128 / sizeof(sycl::uint4)];
} hash128_t;

typedef union {
    uint32_t words[64 / sizeof(uint32_t)];
    sycl::uint2 uint2s[64 / sizeof(sycl::uint2)];
    sycl::uint4 uint4s[64 / sizeof(sycl::uint4)];
} hash64_t;


[[nodiscard]] std::future<Search_results> run_ethash_search(   //
        uint32_t work_groups, uint32_t work_items, sycl::queue q, uint64_t start_nonce, uint64_t d_dag_num_items, const hash128_t* d_dag, const hash32_t& d_header,
        uint64_t d_target, sycl::event e = {});

[[nodiscard]] std::vector<sycl::event> ethash_generate_dag(   //
        uint64_t dag_size, uint32_t work_groups, uint32_t work_items, sycl::queue q, uint32_t d_dag_num_items, uint32_t d_light_num_items, hash128_t* d_dag,
        const hash64_t* d_light, sycl::event e = {});
