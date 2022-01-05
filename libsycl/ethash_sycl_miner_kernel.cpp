/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#include "ethash_sycl_miner_kernel.h"
#include "dagger_shuffled.hpp"
#include "keccak.hpp"

template<int threads_per_hash_ = THREADS_PER_HASH, int parallel_hash_ = PARALLEL_HASH, int accesses_ = ACCESSES>
static inline void ethash_search_kernel(                                                                                                     //
        const sycl::nd_item<1>& item, sycl::accessor<Search_results, 1, sycl::access::mode::write, sycl::access::target::device> g_output,   //
        uint64_t start_nonce,                                                                                                                //
        uint64_t d_dag_num_items,                                                                                                            //
        const hash128_t* const d_dag,                                                                                                        //
        const sycl::accessor<hash32_t, 1, sycl::access::mode::read, sycl::access::target::constant_buffer>& d_header,                        //
        uint64_t d_target) noexcept {

    if (g_output[0].done) { return; }
    using atomic_ref_t = SYCL_ATOMIC_REF<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::global_space>;
    auto done_ref = atomic_ref_t(g_output[0].done);
    if (done_ref.load()) { return; }
    uint32_t const gid = item.get_global_linear_id();
    bool r = compute_hash<threads_per_hash_, parallel_hash_, accesses_>(item, start_nonce + gid, d_dag_num_items, d_dag, d_header.get_pointer(), d_target);
    if (item.get_local_linear_id() == 0U) { atomic_ref_t(g_output[0].hashCount).fetch_add(1U); }
    if (r) { return; }
    uint32_t index = atomic_ref_t(g_output[0].solCount).fetch_add(1U);
    if (index >= MAX_SEARCH_RESULTS) { return; }
    g_output[0].gid[index] = gid;
    done_ref = 1;
}


[[nodiscard]] std::future<Search_results> run_ethash_search(   //
        uint32_t work_groups,                                  //
        uint32_t work_items,                                   //
        sycl::queue q,                                         //
        uint64_t start_nonce,                                  //
        uint64_t d_dag_num_items,                              //
        const hash128_t* const d_dag,                          //
        const hash32_t& d_header,                              //
        uint64_t d_target,                                     //
        sycl::event in_evt) {

    return std::async(std::launch::async, [=]() mutable {
        Search_results g_output{};
        {
            auto header_buf = sycl::buffer<hash32_t>(&d_header, 1U);
            auto output_buf = sycl::buffer<Search_results>(&g_output, 1U);
            header_buf.set_write_back(false);
            header_buf.set_final_data(nullptr);
            q.submit([&](sycl::handler& cgh) {
                auto acc_header = header_buf.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
                auto output_header = output_buf.get_access<sycl::access::mode::write, sycl::access::target::device>(cgh);
                cgh.depends_on(in_evt);
                cgh.parallel_for(                                                  //
                        sycl::nd_range<1>(work_groups * work_items, work_items),   //
                        [=](sycl::nd_item<1> item) /* [[sycl::reqd_sub_group_size(32)]] [[sycl::work_group_size_hint(128)]] */ {
                            ethash_search_kernel(item, output_header, start_nonce, d_dag_num_items, d_dag, acc_header, d_target);
                        });
            });
        }
        return g_output;
    });
}


template<int ethash_dataset_parents_ = ETHASH_DATASET_PARENTS, int node_words_ = NODE_WORDS>
static inline void ethash_calculate_dag_item(   //
        const sycl::nd_item<1>& item,           //
        uint32_t start,                         //
        uint64_t d_dag_num_items,               //
        uint32_t d_light_num_items,             //
        hash128_t* d_dag,                       //
        const hash64_t* const d_light) noexcept {

    const uint32_t node_index = start + item.get_global_linear_id();
    if (((node_index >> 1) & (~1)) >= d_dag_num_items) return;

    union un {
        hash128_t dag_node;
        sycl::uint2 sha3_buf[25]{};
    } u{};

    copy<4>(u.dag_node.uint4s, d_light[node_index % d_light_num_items].uint4s);
    u.dag_node.words[0] ^= node_index;
    SHA3_512(u.sha3_buf);

    const int thread_id = (int) (item.get_local_id() & 3U);
    for (int i = 0; i != ethash_dataset_parents_; ++i) {
        uint32_t parent_index = fnv(node_index ^ i, u.dag_node.words[i % node_words_]) % d_light_num_items;
        for (int t = 0; t < 4; t++) {
            uint32_t shuffle_index = shuffle_sync<4>(item.get_sub_group(), parent_index, t);
            sycl::uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
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
        hash128_t* d_dag,                                     //
        const hash64_t* const d_light, sycl::event evt) {

    const auto work = (int32_t) (dag_size / sizeof(hash64_t));
    const int64_t run = work_groups * work_items;
    std::vector<sycl::event> events{};
    events.reserve((work + run - 1U) / run);   // Upper bound

    int64_t base;
    for (base = 0; base <= work - run; base += run) {
        sycl::event submission_evt = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(evt);
            cgh.parallel_for(                             //
                    sycl::nd_range<1>(run, work_items),   //
                    [=](sycl::nd_item<1> item) /* [[sycl::reqd_sub_group_size(32)]] [[sycl::work_group_size_hint(128)]] */ {
                        ethash_calculate_dag_item(item, base, d_dag_num_items, d_light_num_items, d_dag, d_light);
                    });
        });
        events.emplace_back(submission_evt);
    }
    if (base < work) {
        uint32_t lastGrid = work - base;
        lastGrid = (lastGrid + work_items - 1) / work_items;
        sycl::event submission_evt = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(evt);
            cgh.parallel_for(                                               //
                    sycl::nd_range<1>(lastGrid * work_items, work_items),   //
                    [=](sycl::nd_item<1> item) /* [[sycl::reqd_sub_group_size(32)]] [[sycl::work_group_size_hint(128)]] */ {
                        ethash_calculate_dag_item(item, base, d_dag_num_items, d_light_num_items, d_dag, d_light);
                    });
        });
        events.emplace_back(submission_evt);
    }
    return events;
}
