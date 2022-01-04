/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include "keccak.hpp"
#include "sycl_helpers.hpp"

template<typename T> OPT_CONSTEXPR static inline T fnv(const T& a, const T& b) { return (a * 0x01000193U) ^ b; }

constexpr static inline uint32_t fnv_reduce(const sycl::uint4& v) { return fnv(fnv(fnv(v.x(), v.y()), v.z()), v.w()); }


template<int threads_per_hash, int parallel_hash, int accesses>
static inline bool compute_hash(const sycl::nd_item<1>& item, const uint64_t& nonce, const uint64_t& d_dag_size, const hash128_t* const d_dag, const hash32_t& d_header,
                                const uint64_t& d_target) noexcept {
    // sha3_512(header .. nonce)
    std::array<sycl::uint2, 12> state{};
    state[4] = vectorize(nonce);

    keccak_f1600_init(state, d_header);

    // Threads work together in this phase in groups of 8.
    const int thread_id = (int) (item.get_local_id() & (threads_per_hash - 1U));
    const int mix_idx = thread_id & 3;


    for (int i = 0; i < threads_per_hash; i += parallel_hash) {
        std::array<sycl::uint4, parallel_hash> mix{};
        std::array<uint32_t, parallel_hash> init0{};

// share init among threads
#pragma unroll
        for (int p = 0; p < parallel_hash; p++) {
            std::array<sycl::uint2, 8> shuffle{};
#pragma unroll
            for (int j = 0; j < 8; j++) { shuffle[j] = shuffle_sync<threads_per_hash>(item.get_sub_group(), state[j], i + p); }
            switch (mix_idx) {
                case 0: mix[p] = vectorize2(shuffle[0], shuffle[1]); break;
                case 1: mix[p] = vectorize2(shuffle[2], shuffle[3]); break;
                case 2: mix[p] = vectorize2(shuffle[4], shuffle[5]); break;
                case 3: mix[p] = vectorize2(shuffle[6], shuffle[7]); break;
                default: __builtin_unreachable();
            }
            init0[p] = shuffle_sync<threads_per_hash>(item.get_sub_group(), shuffle[0].x(), 0);
        }

#ifndef DAGGER_VARIANT
        for (int a = 0; a < accesses; a += 4) {
            const auto t = (int) bfe<2, 3>(a);

#pragma unroll
            for (int b = 0; b < 4; b++) {
                std::array<uint32_t, parallel_hash> offset{};
#    pragma unroll
                for (int p = 0; p < parallel_hash; p++) {
                    offset[p] = fnv(init0[p] ^ (a + b), (mix[p][b])) % d_dag_size;
                    offset[p] = shuffle_sync<threads_per_hash>(item.get_sub_group(), offset[p], t);
                }
#    pragma unroll
                for (int p = 0; p < parallel_hash; p++) { mix[p] = fnv(mix[p], d_dag[offset[p]].uint4s[thread_id]); }
            }
        }
#else
#    pragma unroll
        for (int p = 0; p < parallel_hash; p++) {

            for (int a = 0; a < accesses; a += 4) {
#    pragma unroll
                for (int b = 0; b < 4; b++) {
                    const auto t = (int) bfe<2, 3>(a);
                    uint32_t offset = fnv(init0[p] ^ (a + b), (mix[p][b])) % d_dag_size;
                    offset = shuffle_sync<threads_per_hash>(item.get_sub_group(), offset, t);
                    mix[p] = fnv(mix[p], d_dag[offset].uint4s[thread_id]);
                }
            }
        }
#endif


#pragma unroll
        for (int p = 0; p < parallel_hash; p++) {
            std::array<sycl::uint2, 4> shuffle{};
            const uint32_t thread_mix = fnv_reduce(mix[p]);

            // update mix across threads
            shuffle[0].x() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 0);
            shuffle[0].y() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 1);
            shuffle[1].x() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 2);
            shuffle[1].y() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 3);
            shuffle[2].x() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 4);
            shuffle[2].y() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 5);
            shuffle[3].x() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 6);
            shuffle[3].y() = shuffle_sync<threads_per_hash>(item.get_sub_group(), thread_mix, 7);

            if ((i + p) == thread_id) {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }


    // keccak_256(keccak_512(header..nonce) .. mix);
    if (cuda_swab64(keccak_f1600_final(state)) > d_target) return true;

    return false;
}
