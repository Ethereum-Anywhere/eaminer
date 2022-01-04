/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */
#pragma once


#include "sycl_helpers.hpp"
#include <array>

#define USE_PRECOMPUTED_KECCAK_ROUND_CONSTANTS
#ifdef USE_PRECOMPUTED_KECCAK_ROUND_CONSTANTS
static const OPT_CONSTEXPR std::array<sycl::uint2, 24> keccak_round_constants_global{
        sycl::uint2{0x00000001U, 0x00000000U}, sycl::uint2{0x00008082U, 0x00000000U}, sycl::uint2{0x0000808aU, 0x80000000U}, sycl::uint2{0x80008000U, 0x80000000U},
        sycl::uint2{0x0000808bU, 0x00000000U}, sycl::uint2{0x80000001U, 0x00000000U}, sycl::uint2{0x80008081U, 0x80000000U}, sycl::uint2{0x00008009U, 0x80000000U},
        sycl::uint2{0x0000008aU, 0x00000000U}, sycl::uint2{0x00000088U, 0x00000000U}, sycl::uint2{0x80008009U, 0x00000000U}, sycl::uint2{0x8000000aU, 0x00000000U},
        sycl::uint2{0x8000808bU, 0x00000000U}, sycl::uint2{0x0000008bU, 0x80000000U}, sycl::uint2{0x00008089U, 0x80000000U}, sycl::uint2{0x00008003U, 0x80000000U},
        sycl::uint2{0x00008002U, 0x80000000U}, sycl::uint2{0x00000080U, 0x80000000U}, sycl::uint2{0x0000800aU, 0x00000000U}, sycl::uint2{0x8000000aU, 0x80000000U},
        sycl::uint2{0x80008081U, 0x80000000U}, sycl::uint2{0x00008080U, 0x80000000U}, sycl::uint2{0x80000001U, 0x00000000U}, sycl::uint2{0x80008008U, 0x80000000U}};


OPT_CONSTEXPR static inline sycl::uint2 keccak_round_constants(int t) { return keccak_round_constants_global[t]; }
#else
OPT_CONSTEXPR static inline sycl::uint2 keccak_round_constants(int i) {
    const auto rc = [](int t) {
        uint64_t result = 0x1;
        for (int i = 1; i <= t; i++) {
            result <<= 1;
            if (result & 0x100) result ^= 0x71;
        }
        return result & 0x1;
    };

    uint64_t result = 0x0;
    uint32_t shift = 1;
    for (int j = 0; j < 7; j++) {
        uint64_t value = rc(7 * i + j);
        result |= value << (shift - 1);
        shift *= 2;
    }
    return vectorize(result);
}
#endif


OPT_CONSTEXPR static inline void keccak_f1600_init(std::array<sycl::uint2, 12>& state, const hash32_t& d_header) noexcept {
    sycl::uint2 s[25]{};
    sycl::uint2 t[5]{}, u{}, v{};

    devectorize2(d_header.uint4s[0], s[0], s[1]);
    devectorize2(d_header.uint4s[1], s[2], s[3]);
    s[4] = state[4];
    s[5] = {1, 0};
    s[8] = {0U, 0x80000000U};

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0].x() = s[0].x() ^ s[5].x();
    t[0].y() = s[0].y();
    t[1] = s[1];
    t[2] = s[2];
    t[3].x() = s[3].x();
    t[3].y() = s[3].y() ^ s[8].y();
    t[4] = s[4];

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[22] = ROL2(s[14], 39);
    s[14] = ROL2(s[20], 18);
    s[20] = ROL2(s[2], 62);
    s[2] = ROL2(s[12], 43);
    s[12] = ROL2(s[13], 25);
    s[13] = ROL8(s[19]);
    s[19] = ROR8(s[23]);
    s[23] = ROL2(s[15], 41);
    s[15] = ROL2(s[4], 27);
    s[4] = ROL2(s[24], 14);
    s[24] = ROL2(s[21], 2);
    s[21] = ROL2(s[8], 55);
    s[8] = ROL2(s[16], 45);
    s[16] = ROL2(s[5], 36);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[18] = ROL2(s[17], 15);
    s[17] = ROL2(s[11], 10);
    s[11] = ROL2(s[7], 6);
    s[7] = ROL2(s[10], 3);
    s[10] = ROL2(u, 1);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);

    u = s[5];
    v = s[6];
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);
    s[8] = chi(s[8], s[9], u);
    s[9] = chi(s[9], u, v);

    u = s[10];
    v = s[11];
    s[10] = chi(s[10], s[11], s[12]);
    s[11] = chi(s[11], s[12], s[13]);
    s[12] = chi(s[12], s[13], s[14]);
    s[13] = chi(s[13], s[14], u);
    s[14] = chi(s[14], u, v);

    u = s[15];
    v = s[16];
    s[15] = chi(s[15], s[16], s[17]);
    s[16] = chi(s[16], s[17], s[18]);
    s[17] = chi(s[17], s[18], s[19]);
    s[18] = chi(s[18], s[19], u);
    s[19] = chi(s[19], u, v);

    u = s[20];
    v = s[21];
    s[20] = chi(s[20], s[21], s[22]);
    s[21] = chi(s[21], s[22], s[23]);
    s[22] = chi(s[22], s[23], s[24]);
    s[23] = chi(s[23], s[24], u);
    s[24] = chi(s[24], u, v);

    /* iota: a[0,0] ^= round constant */
    //s[0] ^= keccak_round_constants[0];
    s[0] ^= keccak_round_constants(0);

    for (int i = 1; i < 23; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = t[4] ^ ROL2(t[1], 1);
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = t[0] ^ ROL2(t[2], 1);
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = t[1] ^ ROL2(t[3], 1);
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = t[2] ^ ROL2(t[4], 1);
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = t[3] ^ ROL2(t[0], 1);
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1] = ROL2(s[6], 44);
        s[6] = ROL2(s[9], 20);
        s[9] = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2] = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL8(s[19]);
        s[19] = ROR8(s[23]);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4] = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8] = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5] = ROL2(s[3], 28);
        s[3] = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7] = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

        u = s[0];
        v = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u = s[5];
        v = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u = s[10];
        v = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u = s[15];
        v = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u = s[20];
        v = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        //s[0] ^= keccak_round_constants[i];
        s[0] ^= keccak_round_constants(i);
    }

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[10] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[6] ^= u;
    s[16] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[12] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[18] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[9] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[2] = ROL2(s[12], 43);
    s[4] = ROL2(s[24], 14);
    s[8] = ROL2(s[16], 45);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[7] = ROL2(s[10], 3);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);

    /* iota: a[0,0] ^= round constant */
    //s[0] ^= keccak_round_constants[23];
    s[0] ^= keccak_round_constants(23);

#pragma unroll
    for (int i = 0; i < 12; ++i) state[i] = s[i];
}

OPT_CONSTEXPR static inline uint64_t keccak_f1600_final(std::array<sycl::uint2, 12>& state) noexcept {
    sycl::uint2 s[25]{};
    sycl::uint2 t[5]{}, u{}, v{};

#pragma unroll
    for (int i = 0; i < 12; ++i) s[i] = state[i];

    s[12] = {1U, 0U};
    s[16] = {0U, 0x80000000U};

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor3(s[0], s[5], s[10]);
    t[1] = xor3(s[1], s[6], s[11]) ^ s[16];
    t[2] = xor3(s[2], s[7], s[12]);
    t[3] = s[3] ^ s[8];
    t[4] = s[4] ^ s[9];

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[22] = ROL2(s[14], 39);
    s[14] = ROL2(s[20], 18);
    s[20] = ROL2(s[2], 62);
    s[2] = ROL2(s[12], 43);
    s[12] = ROL2(s[13], 25);
    s[13] = ROL8(s[19]);
    s[19] = ROR8(s[23]);
    s[23] = ROL2(s[15], 41);
    s[15] = ROL2(s[4], 27);
    s[4] = ROL2(s[24], 14);
    s[24] = ROL2(s[21], 2);
    s[21] = ROL2(s[8], 55);
    s[8] = ROL2(s[16], 45);
    s[16] = ROL2(s[5], 36);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[18] = ROL2(s[17], 15);
    s[17] = ROL2(s[11], 10);
    s[11] = ROL2(s[7], 6);
    s[7] = ROL2(s[10], 3);
    s[10] = ROL2(u, 1);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);

    u = s[5];
    v = s[6];
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);
    s[8] = chi(s[8], s[9], u);
    s[9] = chi(s[9], u, v);

    u = s[10];
    v = s[11];
    s[10] = chi(s[10], s[11], s[12]);
    s[11] = chi(s[11], s[12], s[13]);
    s[12] = chi(s[12], s[13], s[14]);
    s[13] = chi(s[13], s[14], u);
    s[14] = chi(s[14], u, v);

    u = s[15];
    v = s[16];
    s[15] = chi(s[15], s[16], s[17]);
    s[16] = chi(s[16], s[17], s[18]);
    s[17] = chi(s[17], s[18], s[19]);
    s[18] = chi(s[18], s[19], u);
    s[19] = chi(s[19], u, v);

    u = s[20];
    v = s[21];
    s[20] = chi(s[20], s[21], s[22]);
    s[21] = chi(s[21], s[22], s[23]);
    s[22] = chi(s[22], s[23], s[24]);
    s[23] = chi(s[23], s[24], u);
    s[24] = chi(s[24], u, v);

    /* iota: a[0,0] ^= round constant */
    //s[0] ^= keccak_round_constants[0];
    s[0] ^= keccak_round_constants(0);

    for (int i = 1; i < 23; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = t[4] ^ ROL2(t[1], 1);
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = t[0] ^ ROL2(t[2], 1);
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = t[1] ^ ROL2(t[3], 1);
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = t[2] ^ ROL2(t[4], 1);
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = t[3] ^ ROL2(t[0], 1);
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1] = ROL2(s[6], 44);
        s[6] = ROL2(s[9], 20);
        s[9] = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2] = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL8(s[19]);
        s[19] = ROR8(s[23]);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4] = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8] = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5] = ROL2(s[3], 28);
        s[3] = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7] = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        u = s[0];
        v = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u = s[5];
        v = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u = s[10];
        v = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u = s[15];
        v = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u = s[20];
        v = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        //s[0] ^= keccak_round_constants[i];
        s[0] ^= keccak_round_constants(i);
    }

    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    s[0] = xor3(s[0], t[4], ROL2(t[1], 1));
    s[6] = xor3(s[6], t[0], ROL2(t[2], 1));
    s[12] = xor3(s[12], t[1], ROL2(t[3], 1));

    s[1] = ROL2(s[6], 44);
    s[2] = ROL2(s[12], 43);

    s[0] = chi(s[0], s[1], s[2]);

    /* iota: a[0,0] ^= round constant */
    // s[0] ^= vectorize(keccak_round_constants[23]);
    //return devectorize(s[0] ^ keccak_round_constants[23]);
    return devectorize(s[0] ^ keccak_round_constants(23));
}

OPT_CONSTEXPR static inline void SHA3_512(sycl::uint2* s) noexcept {
    sycl::uint2 t[5]{}, u{}, v{};

#pragma unroll
    for (int i = 8; i < 25; i++) { s[i] = {}; }
    s[8].x() = 1U;
    s[8].y() = 0x80000000U;

    for (int i = 0; i < 23; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = t[4] ^ ROL2(t[1], 1);
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = t[0] ^ ROL2(t[2], 1);
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = t[1] ^ ROL2(t[3], 1);
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = t[2] ^ ROL2(t[4], 1);
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = t[3] ^ ROL2(t[0], 1);
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1] = ROL2(s[6], 44);
        s[6] = ROL2(s[9], 20);
        s[9] = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2] = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL2(s[19], 8);
        s[19] = ROL2(s[23], 56);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4] = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8] = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5] = ROL2(s[3], 28);
        s[3] = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7] = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        u = s[0];
        v = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u = s[5];
        v = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u = s[10];
        v = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u = s[15];
        v = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u = s[20];
        v = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        //s[0] ^= keccak_round_constants[i];
        s[0] ^= keccak_round_constants(i);
    }

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[10] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[6] ^= u;
    s[16] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[12] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[18] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[9] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[2] = ROL2(s[12], 43);
    s[4] = ROL2(s[24], 14);
    s[8] = ROL2(s[16], 45);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[7] = ROL2(s[10], 3);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);

    /* iota: a[0,0] ^= round constant */
    //s[0] ^= keccak_round_constants[23];
    s[0] ^= keccak_round_constants(23);
}
