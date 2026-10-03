/* Byte-plane shuffle kernels, shared by _bytetools and the fused zstd
 * decode calls.
 *
 * oc_unshuffle interleaves `count` elements of `k` bytes from `k` planes
 * spaced `plane` bytes apart (plane b of element j at src[b * plane + j])
 * into dst (element j byte b at dst[j * k + b]); oc_shuffle is its
 * inverse, from dst-like items back into planes.
 *
 * It lives in C with __restrict pointers rather than as a Cython loop:
 * the identical loop inlined into _zstd.pyx compiled scalar under clang
 * (0.81 against 0.51 ms for a 2 MiB tile on arm64 macOS) because nothing
 * told the compiler that source and destination cannot overlap.
 *
 * For 2, 4 and 8 byte elements each element is assembled in a register
 * and moved whole, which every compiler vectorizes. The byte loops are
 * vectorized by GCC and Clang only for 2-byte elements: 4096 x 4096
 * uint16 took 11.6 against 3.1 ms on MSVC, and 32 MB of 4-byte elements
 * 20.6 against 4.6 on MSVC, 13.1 against 3.5 on GCC and 10.5 against 1.7
 * on Apple Clang. Loads go through memcpy and stores through a typed
 * pointer, the form each direction vectorized in on every compiler:
 * Apple Clang left a memcpy store scalar (5.8 against 1.7 ms), and
 * MSVC 14.41 took twice as long with typed loads (12.7 against 6.4 ms
 * for 4-byte elements). A typed store needs dst
 * aligned to the element, so an unaligned dst takes the byte loop, as
 * does a big-endian host, where a word's bytes run the other way.
 */
#ifndef OPENCODECS_BYTEPLANES_H
#define OPENCODECS_BYTEPLANES_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Plain C99 static inline: every compiler that builds CPython 3.10
   extensions accepts it, MSVC included (CPython's own headers use it). */

static inline int oc_little_endian(void)
{
    const union { uint16_t u; uint8_t b[2]; } one = {1};
    return one.b[0];
}

/* Whether elements of k bytes can be moved as words at dst. */
static inline int oc_words(const void *dst, size_t k)
{
    return (k == 2 || k == 4 || k == 8) && oc_little_endian()
        && ((uintptr_t)dst % k) == 0;
}

static inline void oc_unshuffle(uint8_t *__restrict dst,
                                const uint8_t *__restrict src,
                                size_t plane, size_t count, size_t k)
{
    size_t j, b;
    if (k == 1) {
        memcpy(dst, src, count);
#if defined(__GNUC__)
    } else if (k == 2) {
        /* GCC vectorizes this byte loop better than the word form: 1.46x
           faster for a 1000 x 1000 uint16 plane and 1.11x for 2000 x 2000
           (GCC 15, x86-64, -O3, 41 alternating rounds). Clang ties. MSVC
           takes the word form below, which it needs (see the top). */
        const uint8_t *__restrict lo = src;
        const uint8_t *__restrict hi = src + plane;
        for (j = 0; j < count; j++) {
            dst[2 * j] = lo[j];
            dst[2 * j + 1] = hi[j];
        }
#endif
    } else if (oc_words(dst, k)) {
        const uint8_t *__restrict p0 = src;
        const uint8_t *__restrict p1 = src + plane;
        if (k == 2) {
            uint16_t *__restrict o = (uint16_t *)dst;
            for (j = 0; j < count; j++)
                o[j] = (uint16_t)(p0[j] | ((uint16_t)p1[j] << 8));
        } else {
            const uint8_t *__restrict p2 = src + 2 * plane;
            const uint8_t *__restrict p3 = src + 3 * plane;
            uint32_t *__restrict o = (uint32_t *)dst;
            if (k == 4) {
                for (j = 0; j < count; j++)
                    o[j] = (uint32_t)p0[j] | ((uint32_t)p1[j] << 8)
                         | ((uint32_t)p2[j] << 16) | ((uint32_t)p3[j] << 24);
            } else {
                const uint8_t *__restrict p4 = src + 4 * plane;
                const uint8_t *__restrict p5 = src + 5 * plane;
                const uint8_t *__restrict p6 = src + 6 * plane;
                const uint8_t *__restrict p7 = src + 7 * plane;
                for (j = 0; j < count; j++) {
                    o[2 * j] = (uint32_t)p0[j] | ((uint32_t)p1[j] << 8)
                             | ((uint32_t)p2[j] << 16) | ((uint32_t)p3[j] << 24);
                    o[2 * j + 1] = (uint32_t)p4[j] | ((uint32_t)p5[j] << 8)
                                 | ((uint32_t)p6[j] << 16) | ((uint32_t)p7[j] << 24);
                }
            }
        }
    } else if (k == 2) {
        const uint8_t *__restrict lo = src;
        const uint8_t *__restrict hi = src + plane;
        for (j = 0; j < count; j++) {
            dst[2 * j] = lo[j];
            dst[2 * j + 1] = hi[j];
        }
    } else {
        for (b = 0; b < k; b++) {
            const uint8_t *__restrict p = src + b * plane;
            for (j = 0; j < count; j++)
                dst[j * k + b] = p[j];
        }
    }
}

static inline void oc_shuffle(uint8_t *__restrict dst,
                              const uint8_t *__restrict src,
                              size_t plane, size_t count, size_t k)
{
    size_t j, b;
    if (k == 1) {
        memcpy(dst, src, count);
    } else if (k == 2 && oc_little_endian()) {
        uint8_t *__restrict lo = dst;
        uint8_t *__restrict hi = dst + plane;
        for (j = 0; j < count; j++) {
            uint16_t v;
            memcpy(&v, src + 2 * j, 2);
            lo[j] = (uint8_t)v;
            hi[j] = (uint8_t)(v >> 8);
        }
    } else if (k == 4 && oc_little_endian()) {
        uint8_t *__restrict p0 = dst;
        uint8_t *__restrict p1 = dst + plane;
        uint8_t *__restrict p2 = dst + 2 * plane;
        uint8_t *__restrict p3 = dst + 3 * plane;
        for (j = 0; j < count; j++) {
            uint32_t v;
            memcpy(&v, src + 4 * j, 4);
            p0[j] = (uint8_t)v;
            p1[j] = (uint8_t)(v >> 8);
            p2[j] = (uint8_t)(v >> 16);
            p3[j] = (uint8_t)(v >> 24);
        }
    } else if (k == 8 && oc_little_endian()) {
        /* Each half of an 8-byte element in its own pass: eight planes
           written from one loop vectorized worse, 15.3 against 13.7 ms
           on MSVC and 11.6 against 6.0 on GCC. The strides are written
           out; passed as a parameter, MSVC vectorized the 4-byte loop
           worse (18.8 against 11.7 ms). */
        for (b = 0; b < 8; b += 4) {
            const uint8_t *__restrict s = src + b;
            uint8_t *__restrict p0 = dst + b * plane;
            uint8_t *__restrict p1 = p0 + plane;
            uint8_t *__restrict p2 = p1 + plane;
            uint8_t *__restrict p3 = p2 + plane;
            for (j = 0; j < count; j++) {
                uint32_t v;
                memcpy(&v, s + 8 * j, 4);
                p0[j] = (uint8_t)v;
                p1[j] = (uint8_t)(v >> 8);
                p2[j] = (uint8_t)(v >> 16);
                p3[j] = (uint8_t)(v >> 24);
            }
        }
    } else {
        for (b = 0; b < k; b++) {
            uint8_t *__restrict p = dst + b * plane;
            for (j = 0; j < count; j++)
                p[j] = src[j * k + b];
        }
    }
}

#endif
