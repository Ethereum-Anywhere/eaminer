/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include "ethash_miner_common.h"

#include "dagger_shuffled.hpp"
#include "keccak.hpp"


static constexpr Search_results empty_res{};


[[nodiscard]] sycl_device_task run_ethash_search(   //
        uint32_t work_groups,                       //
        uint32_t work_items,                        //
        sycl::queue q,                              //
        sycl_device_task task,                      //
        uint64_t start_nonce,                       //
        uint64_t d_dag_num_items,                   //
        const hash128_t* __restrict const d_dag,    //
        hash32_t d_header,                          //
        uint64_t d_target,                          //
        uint32_t* __restrict d_kill_signal_host) {

    constexpr bool avoid_excessive_atomic_loads = true;
    using uint_atomic_ref_t = SYCL_ATOMIC_REF<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::global_space>;
    using uint_atomic_ref_host_t = SYCL_ATOMIC_REF<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>;
    auto init_evt = q.memcpy(task.res, &empty_res, sizeof(Search_results), task.e);
    task.e = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(init_evt);
        cgh.parallel_for<sycl_ethash_search_kernel_tag>(                   //
                sycl::nd_range<1>(work_groups * work_items, work_items),   //
                [=, output_buffer = task.res](sycl::nd_item<1> item) /* [[sycl::reqd_sub_group_size(32)]] [[sycl::work_group_size_hint(128)]] */ {
                    auto done_ref = uint_atomic_ref_t(output_buffer->done);
                    /* We don't want to do RMA to host memory too often, so every 2048 work-items only we will cache the data in device memory. */
                    if (item.get_global_linear_id() % 2048U == 0 && !done_ref.load()) {
                        auto host_signal_ref = uint_atomic_ref_host_t(*d_kill_signal_host);
                        done_ref.store(host_signal_ref.load() != 0);
                    }

                    if constexpr (avoid_excessive_atomic_loads) {
                        uint32_t run_cancelled = false;
                        if (item.get_sub_group().get_local_linear_id() == 0U) { run_cancelled = done_ref.load(); }
                        if (sycl::group_broadcast(item.get_sub_group(), run_cancelled, 0U) != 0U) { return; }
                    } else {
                        if (done_ref.load()) { return; }
                    }

                    bool r = compute_hash<THREADS_PER_HASH, PARALLEL_HASH>(item, start_nonce + item.get_global_linear_id(), d_dag_num_items, d_dag, d_header, d_target);
                    if (item.get_local_linear_id() == 0U) { uint_atomic_ref_t(output_buffer->hashCount).fetch_add(1U); }
                    if (r) { return; }

                    if (uint32_t index = uint_atomic_ref_t(output_buffer->solCount).fetch_add(1U); index < MAX_SEARCH_RESULTS) {
                        output_buffer->gid[index] = item.get_global_linear_id();
                        uint_atomic_ref_t(output_buffer->done).store(1);
                    }
                });
    });
    return task;
}


static inline void ethash_calculate_dag_item(   //
        const sycl::nd_item<1>& item,           //
        uint32_t start,                         //
        uint32_t d_dag_num_items,               //
        uint32_t d_light_num_items,             //
        hash128_t* __restrict d_dag,            //
        const hash64_t* __restrict const d_light) noexcept {

    const uint32_t node_index = start + item.get_global_linear_id();
    if (((node_index >> 1) & (~1)) >= d_dag_num_items) return;

    union un {
        hash128_t dag_node;
        sycl::uint2 sha3_buf[25]{};
    } u{};

    copy<4>(u.dag_node.uint4s, d_light[node_index % d_light_num_items].uint4s);
    u.dag_node.words[0] ^= node_index;
    SHA3_512(u.sha3_buf);

    const int thread_id = (int) (item.get_local_linear_id() & 3U);
#pragma unroll
    for (int i = 0; i != ETHASH_DATASET_PARENTS; ++i) {
        uint32_t parent_index = fnv(node_index ^ i, u.dag_node.words[i % NODE_WORDS]) % d_light_num_items;
#pragma unroll
        for (int t = 0; t < 4; t++) {
            uint32_t shuffle_index = shuffle_sync<4>(item.get_sub_group(), parent_index, t);
            sycl::uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
#pragma unroll
            for (int w = 0; w < 4; w++) {
                sycl::uint4 s4 = shuffle_sync<4>(item.get_sub_group(), p4, w);
                if (t == thread_id) { u.dag_node.uint4s[w] = fnv(u.dag_node.uint4s[w], s4); }
            }
        }
    }

    SHA3_512(u.sha3_buf);
    auto* dag_nodes = reinterpret_cast<hash64_t*>(d_dag);
    copy<4>(dag_nodes[node_index].uint4s, u.dag_node.uint4s);
}

[[nodiscard]] std::vector<sycl::event> ethash_generate_dag(   //
        uint64_t dag_size,                                    //
        uint32_t work_groups,                                 //
        uint32_t work_items,                                  //
        sycl::queue q,                                        //
        uint32_t d_dag_num_items,                             //
        uint32_t d_light_num_items,                           //
        hash128_t* __restrict d_dag,                          //
        const hash64_t* __restrict const d_light,             //
        const sycl::event& evt) {

    /**
     * Kernels need to be launched from a single place otherwise it generates
     * duplicate code, (and breaks with named kernels!)
     */
    const auto dag_generation_kernel_launcher = [&](const uint32_t& launch_work_groups, const uint32_t& base) {
        return q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(evt);
            cgh.parallel_for<sycl_ethash_calculate_dag_item_kernel_tag>(              //
                    sycl::nd_range<1>(launch_work_groups * work_items, work_items),   //
                    [=](sycl::nd_item<1> item) /* [[sycl::reqd_sub_group_size(32)]] [[sycl::work_group_size_hint(128)]] */ {
                        ethash_calculate_dag_item(item, base, d_dag_num_items, d_light_num_items, d_dag, d_light);
                    });
        });
    };

    const auto work = (int32_t) (dag_size / sizeof(hash64_t));
    const int64_t run = work_groups * work_items;
    std::vector<sycl::event> events{};
    events.reserve((work + run - 1U) / run);   // Upper bound

    int64_t base;
    for (base = 0; base <= work - run; base += run) { events.emplace_back(dag_generation_kernel_launcher(work_groups, base)); }
    if (base < work) {
        uint32_t lastGrid = work - base;
        lastGrid = (lastGrid + work_items - 1) / work_items;
        events.emplace_back(dag_generation_kernel_launcher(lastGrid, base));
    }
    return events;
}
