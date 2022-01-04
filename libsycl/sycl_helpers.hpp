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
template<typename T> constexpr static inline T rotate(const T& x, int n) { return (x << n) | (x >> ((sizeof(T) * 8) - n)); }
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
 *
 * @param LO
 * @param HI
 * @return
 */
constexpr static inline uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI) { return sycl::upsample(HI, LO); }

/**
 * Swap bytes order within a 32 bit word
 * @param x
 * @return
 */
constexpr static inline uint32_t cuda_swab32(const uint32_t x) {
    return (((x) << 24) & 0xff000000U) | (((x) << 8) & 0x00ff0000U) | (((x) >> 8) & 0x0000ff00U) | (((x) >> 24) & 0x000000ffU);
}

/**
 * Swap bytes order within a 64 bit word
 * @param x
 * @return
 */
constexpr static inline uint64_t cuda_swab64(const uint64_t x) {
    return (((uint64_t) (x) &0xff00000000000000ULL) >> 56) | (((uint64_t) (x) &0x00ff000000000000ULL) >> 40) | (((uint64_t) (x) &0x0000ff0000000000ULL) >> 24) |
           (((uint64_t) (x) &0x000000ff00000000ULL) >> 8) | (((uint64_t) (x) &0x00000000ff000000ULL) << 8) | (((uint64_t) (x) &0x0000000000ff0000ULL) << 24) |
           (((uint64_t) (x) &0x000000000000ff00ULL) << 40) | (((uint64_t) (x) &0x00000000000000ffULL) << 56);
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


/**
 * 64 Bit rorate right
 * @param x
 * @param offset
 * @return
 */
constexpr static inline uint64_t ROTR64(const uint64_t x, const int offset) { return sycl::rotate<uint64_t>(x, 64 - offset); }

/**
 * 64 Bit rotate left
 * @param x
 * @param offset
 * @return
 */
constexpr static inline uint64_t ROTL64(const uint64_t x, const int offset) { return sycl::rotate<uint64_t>(x, offset); }

/**
 * Rotates 8 bytes left
 * @param x
 * @return
 */
constexpr static inline uint32_t ROL8(uint32_t x) { return sycl::rotate<uint32_t>(x, 8); }

OPT_CONSTEXPR static inline sycl::uint2 ROR8(const sycl::uint2& a) {
    uint64_t word = devectorize(a);
    word = sycl::rotate<uint64_t>(word, 64 - 8);
    return vectorize(word);
}

OPT_CONSTEXPR static inline sycl::uint2 ROL8(const sycl::uint2& a) {
    uint64_t word = devectorize(a);
    word = sycl::rotate<uint64_t>(word, 8);
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
template<uint32_t bit, uint32_t numBits> static inline uint32_t bfe(uint32_t x) {
    constexpr uint32_t mask = (numBits == 8 * sizeof(uint32_t)) ? ~(uint32_t(0)) : (uint32_t(1) << numBits) - uint32_t(1);
    return (x >> bit) & mask;
}


template<size_t width, typename T> static inline T shuffle_sync(const sycl::sub_group& sg, const T& val, int srcLane) {
    if constexpr (width == 1) {
        return val;
    } else {
        static_assert(width == 1 || width == 2 || width == 4 || width == 8 || width == 16 || width == 32);
        int32_t offset = (sg.get_local_linear_id() / width) * width;
        return sycl::select_from_group(sg, val, (srcLane % width) + offset);
    }
}
