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

static inline constexpr std::array<uint64_t, 24> round_constants = {   //
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a, 0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};

OPT_CONSTEXPR static inline sycl::uint2 keccak_round_constants(int t) { return vectorize(round_constants[t]); }


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

OPT_CONSTEXPR static inline void SHA3_512(sycl::uint2* __restrict s) noexcept {
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
