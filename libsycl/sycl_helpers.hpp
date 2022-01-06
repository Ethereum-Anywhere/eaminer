/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once


#include "ethash_sycl_miner_kernel.h"

/**
 * Missing functions in HIPSYCL
 */
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
namespace sycl {
template<typename T> constexpr static inline T rotate(const T& x, uint32_t n) { return (x << n) | (x >> ((sizeof(T) * 8U) - n)); }
constexpr static inline uint64_t upsample(const uint32_t& hi, const uint32_t& lo) { return (uint64_t{hi} << 32) + lo; }
}   // namespace sycl
#endif

OPT_CONSTEXPR static inline sycl::uint2 xor5(const sycl::uint2& a, const sycl::uint2& b, const sycl::uint2& c, const sycl::uint2& d, const sycl::uint2& e) {
    return a ^ b ^ c ^ d ^ e;
}

OPT_CONSTEXPR static inline sycl::uint2 xor3(const sycl::uint2& a, const sycl::uint2& b, const sycl::uint2& c) { return a ^ b ^ c; }

OPT_CONSTEXPR static inline sycl::uint2 chi(const sycl::uint2& a, const sycl::uint2& b, const sycl::uint2& c) { return a ^ (~b) & c; }

template<int N, typename T> static constexpr void copy(T* dst, const T* src) {
#pragma unroll
    for (int i = 0; i != N; ++i) { (dst)[i] = (src)[i]; }
}


/**
 * Swap bytes order within a 64 bit word
 * @param x
 * @return
 */
constexpr static inline uint64_t SWAB64(const uint64_t x) {
    return ((x & 0xff00000000000000ULL) >> 56) | ((x & 0x00ff000000000000ULL) >> 40) | ((x & 0x0000ff0000000000ULL) >> 24) | ((x & 0x000000ff00000000ULL) >> 8) |
           ((x & 0x00000000ff000000ULL) << 8) | ((x & 0x0000000000ff0000ULL) << 24) | ((x & 0x000000000000ff00ULL) << 40) | ((x & 0x00000000000000ffULL) << 56);
}

/**
 * Returns 64 unsigned where HI is x.y and LO is x.x
 */
constexpr static inline uint64_t devectorize(const sycl::uint2& x) { return sycl::upsample(x.y(), x.x()); }

/**
 * returns "sycl::uint2{ lo, hi }"
 * @param x
 * @return
 */
OPT_CONSTEXPR static inline sycl::uint2 vectorize(const uint64_t x) { return sycl::uint2{x & 0xffffffffU, x >> 32}; }   // LO,HI

/**
 *
 * @param in
 * @param x
 * @param y
 * @return
 */
constexpr static inline void devectorize2(const sycl::uint4& in, sycl::uint2& x, sycl::uint2& y) {
    x.x() = in.x();
    x.y() = in.y();
    y.x() = in.z();
    y.y() = in.w();
}

/**
 * Concatenates two sycl::uint2
 * @param x
 * @param y
 * @return
 */
OPT_CONSTEXPR static inline sycl::uint4 vectorize2(const sycl::uint2& x, const sycl::uint2& y) { return {x.x(), x.y(), y.x(), y.y()}; }

OPT_CONSTEXPR static inline sycl::uint2 ROR8(const sycl::uint2& a) {
    uint64_t word = devectorize(a);
    word = sycl::rotate<uint64_t>(word, 64U - 8U);
    return vectorize(word);
}

OPT_CONSTEXPR static inline sycl::uint2 ROL8(const sycl::uint2& a) {
    uint64_t word = devectorize(a);
    word = sycl::rotate<uint64_t>(word, 8U);
    return vectorize(word);
}

/**
 * OK
 */
OPT_CONSTEXPR static inline sycl::uint2 ROL2(const sycl::uint2& a, const int offset) {
    sycl::uint2 result{};
    if (offset <= 32) {
        result.y() = (a.y() << (offset)) + (a.x() >> (32 - offset));
        result.x() = (a.x() << (offset)) + (a.y() >> (32 - offset));
    } else {
        result.y() = (a.x() << (offset - 32)) + (a.y() >> (64 - offset));
        result.x() = (a.y() << (offset - 32)) + (a.x() >> (64 - offset));
    }
    return result;
}

/**
 * Generates the proper PTX on cuda at least: https://cuda.godbolt.org/z/4rds7od19
 */
template<int bit, int numBits> static inline uint32_t bfe(uint32_t x) {
    static_assert(bit >= 0 && bit < 32);
    static_assert(numBits > 0 && numBits <= 32);
    constexpr uint32_t mask = (numBits == 8U * sizeof(uint32_t)) ? ~(static_cast<uint32_t>(0U)) : (static_cast<uint32_t>(1U) << numBits) - static_cast<uint32_t>(1);
    return (x >> bit) & mask;
}


template<int width, typename T> static inline T shuffle_sync(const sycl::sub_group& sg, const T& val, int srcLane) {
    if constexpr (width == 1) {
        return val;
    } else {
        static_assert(width == 2 || width == 4 || width == 8 || width == 16 || width == 32);
        int32_t offset = (static_cast<int32_t>(sg.get_local_linear_id()) / width) * width;
        return sycl::select_from_group(sg, val, (srcLane % width) + offset);
    }
}

/**
 * Returns the maximum admissible work group size
 * @tparam KernelName
 * @param q
 * @return
 */
template<typename KernelName> static inline size_t sycl_max_work_items(sycl::queue& q) {
    size_t max_items = std::max<size_t>(1U, std::min<size_t>(2048U, static_cast<uint32_t>(q.get_device().get_info<sycl::info::device::max_work_group_size>())));
#if defined(SYCL_IMPLEMENTATION_INTEL) || defined(SYCL_IMPLEMENTATION_ONEAPI)
    try {
        sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
        auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context()).get_kernel(id);
        //size_t register_count = kernel.get_info<sycl::info::kernel_device_specific::ext_codeplay_num_regs>(q.get_device());
        max_items = std::min(max_items, kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device()));
        if (q.get_device().is_gpu()) { max_items = std::min<size_t>(max_items, 128U); }
    } catch (std::exception& e) {
        std::cout << "Couldn't read kernel properties for device: " << q.get_device().get_info<sycl::info::device::name>() << " got exception: " << e.what() << std::endl;
    }
#endif
    return max_items;
}
