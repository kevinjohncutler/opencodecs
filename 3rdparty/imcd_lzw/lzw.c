/* TIFF LZW encoder (compression scheme 5, TIFF 6 §13).
 *
 * Self-contained excerpt from imagecodecs' imcd.c — BSD-3, Christoph
 * Gohlke; see LICENSE in this directory. Only the encoder is
 * vendored (decode is implemented in pure Cython directly in
 * src/opencodecs/codecs/_tiff.pyx).
 *
 * Algorithm preserved verbatim from imcd; only the public-symbol
 * names are prefixed with ``opencodecs_`` to keep the global ABI
 * clear of imagecodecs collisions when both libraries are loaded.
 *
 * Error codes match opencodecs' convention:
 *   -5 IMCD_VALUE_ERROR
 *   -2 IMCD_MEMORY_ERROR
 *   -7 IMCD_OUTPUT_TOO_SMALL
 */

#include "lzw.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define LZW_CLEAR 256
#define LZW_EOI 257
#define LZW_FIRST 258
#define LZW_HASH_SIZE 7349
#define LZW_HASH_STEP 257

#define LZW_VALUE_ERROR -5
#define LZW_MEMORY_ERROR -2
#define LZW_OUTPUT_TOO_SMALL -7

ssize_t opencodecs_lzw_encode_size(ssize_t srcsize) {
    return (srcsize * 141) / 100 + 3;
}

#define LZW_WRITE_DST \
{ \
    if (dstindex >= dstsize) { \
        dstindex = LZW_OUTPUT_TOO_SMALL; \
        goto DONE; \
    } \
    dst[dstindex++] = (uint8_t)(dstbyte >> bitc); \
}

ssize_t opencodecs_lzw_encode(
    const uint8_t* src, ssize_t srcsize,
    uint8_t* dst, ssize_t dstsize)
{
    ssize_t dstindex = 0;
    ssize_t srcindex = 0;
    int* hash_keys = NULL;
    int* hash_values = NULL;
    int hashkey = 0;
    int hashcode = 0;
    int nextcode = LZW_FIRST;
    int dstbyte = LZW_CLEAR;
    int bitw = 9;
    int bitc = 1;
    int omega = 0;
    int k = 0;

    if (src == NULL || srcsize < 0 || dst == NULL || dstsize < 0) {
        return LZW_VALUE_ERROR;
    }
    if (dstsize < 3) {
        return LZW_OUTPUT_TOO_SMALL;
    }

    /* Write CLEAR code at the head of every LZW stream. */
    bitc = 1;
    dstbyte = LZW_CLEAR;
    LZW_WRITE_DST;

    if (srcsize < 1) {
        /* Empty input: emit EOI and return. */
        dstbyte = ((dstbyte << bitw) | LZW_EOI) << 8;
        bitc += bitw;
        LZW_WRITE_DST;
        bitc -= 8;
        LZW_WRITE_DST;
        return dstindex;
    }

    /* Hash table = two parallel arrays of ints (keys + values),
       sized for the worst-case dictionary fill. */
    hash_keys = (int*)malloc(sizeof(int) * LZW_HASH_SIZE * 2);
    if (hash_keys == NULL) {
        return LZW_MEMORY_ERROR;
    }
    hash_values = hash_keys + LZW_HASH_SIZE;
    memset(hash_keys, 0xFF, sizeof(int) * LZW_HASH_SIZE);

    omega = src[0] & 0xff;

    for (srcindex = 1; srcindex < srcsize; srcindex++) {
        k = src[srcindex] & 0xff;
        hashkey = (omega << 8) | k;
        hashcode = (hashkey * LZW_HASH_STEP) % LZW_HASH_SIZE;

        while (hash_keys[hashcode] >= 0) {
            if (hash_keys[hashcode] == hashkey) {
                /* Omega+K already in table — extend. */
                omega = hash_values[hashcode];
                goto OUTER;
            }
            hashcode++;
            if (hashcode == LZW_HASH_SIZE) {
                hashcode = 0;
            }
        }

        /* Omega+K not in table — add new entry. */
        hash_keys[hashcode] = hashkey;
        hash_values[hashcode] = nextcode++;

        /* Emit Omega's code. */
        dstbyte = (dstbyte << bitw) | omega;
        bitc += bitw - 8;
        LZW_WRITE_DST;
        if (bitc >= 8) {
            bitc -= 8;
            LZW_WRITE_DST;
        }

        omega = k;

        /* Bit-width transitions (TIFF/MSB convention, TIFF6 p.60):
           the ADD step precedes WRITE so the code written when entry
           511 is added still uses 9-bit; the following code uses
           10-bit. Decoder's switch lags by 1 (on tablesize==511)
           which is the classic LZW decode lag — both sides agree. */
        switch (nextcode) {
            case 512:  bitw = 10; break;
            case 1024: bitw = 11; break;
            case 2048: bitw = 12; break;
            case 4096:
                /* Table full — emit CLEAR and reinitialize. */
                dstbyte = (dstbyte << bitw) | LZW_CLEAR;
                bitc += bitw - 8;
                LZW_WRITE_DST;
                if (bitc >= 8) {
                    bitc -= 8;
                    LZW_WRITE_DST;
                }
                memset(hash_keys, 0xFF, sizeof(int) * LZW_HASH_SIZE);
                nextcode = LZW_FIRST;
                bitw = 9;
                break;
        }
OUTER:  ;
    }

    /* Emit the final Omega code. */
    dstbyte = (dstbyte << bitw) | omega;
    bitc += bitw - 8;
    LZW_WRITE_DST;
    if (bitc >= 8) {
        bitc -= 8;
        LZW_WRITE_DST;
    }

    /* Final EOI; width may have just rolled forward for the last code. */
    switch (nextcode) {
        case 511:  bitw = 10; break;
        case 1023: bitw = 11; break;
        case 2047: bitw = 12; break;
    }
    dstbyte = ((dstbyte << bitw) | LZW_EOI) << 8;
    bitc += bitw;
    LZW_WRITE_DST;
    if (bitc >= 8) {
        bitc -= 8;
        LZW_WRITE_DST;
        if (bitc >= 8) {
            bitc -= 8;
            LZW_WRITE_DST;
        }
    }

DONE:
    free(hash_keys);
    return dstindex;
}
