/* Byte-plane unshuffle kernel shared by the fused zstd decode calls.
 *
 * Interleaves `count` elements of `k` bytes from `k` planes spaced
 * `plane` bytes apart (plane b of element j at src[b * plane + j]) into
 * dst (element j byte b at dst[j * k + b]).
 *
 * It lives in C with __restrict pointers rather than as a Cython loop:
 * the identical loop inlined into _zstd.pyx compiled scalar under clang
 * (0.81 against 0.51 ms for a 2 MiB tile on arm64 macOS) because nothing
 * told the compiler that source and destination cannot overlap, while
 * the same loop in _bytetools happened to vectorize.
 */
#ifndef OPENCODECS_BYTEPLANES_H
#define OPENCODECS_BYTEPLANES_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Plain C99 static inline: every compiler that builds CPython 3.10
   extensions accepts it, MSVC included (CPython's own headers use it). */
static inline void oc_unshuffle(uint8_t *__restrict dst,
                                const uint8_t *__restrict src,
                                size_t plane, size_t count, size_t k)
{
    size_t j, b;
    if (k == 1) {
        memcpy(dst, src, count);
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

#endif
