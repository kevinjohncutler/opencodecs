/* EER (Electron Event Representation) decoder.
 *
 * Self-contained excerpt from imagecodecs' imcd.c (BSD-3 licensed,
 * Christoph Gohlke; see LICENSE in this directory). Only the two
 * EER entry points + the bitmask helper are kept; everything else
 * (delta / packints / LZW / ...) was excluded.
 *
 * Reference: "EER file format documentation 3.0" by M. Leichsenring,
 * March 2023. The decoder walks a variable-length bitstream of
 * (skip, h_offset, v_offset) events and increments output pixels at
 * each event location.
 *
 * Helper macros / branch structure preserved verbatim from imcd.c
 * so future imagecodecs upstream fixes can be applied directly.
 */

#include "eer.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define MAX_(a, b) ((a) > (b) ? (a) : (b))

static uint16_t eer_bitmask(int bps) {
    if (bps < 0 || bps > 16) return 0;
    /* 0xFFFFu is unsigned int (>=32 bits); right shift 0..16 is safe. */
    return (uint16_t)(0xFFFFu >> (16 - bps));
}

ssize_t opencodecs_eer_decode_u1(
    const uint8_t* src, ssize_t srcsize,
    uint8_t* dst, ssize_t height, ssize_t width,
    uint32_t skipbits, uint32_t horzbits, uint32_t vertbits,
    uint32_t superres)
{
    const ssize_t dstsize = height * width;
    const ssize_t nbits = (ssize_t)(skipbits + horzbits + vertbits);
    const ssize_t srcbits = srcsize * 8 - nbits;
    const uint32_t skipmask = eer_bitmask((int)skipbits);
    const uint32_t horzmask = eer_bitmask((int)horzbits);
    const uint32_t vertmask = eer_bitmask((int)vertbits);
    const uint32_t horzshift =
        (uint32_t)MAX_(0, (ssize_t)horzbits - (ssize_t)superres);
    const uint32_t vertshift =
        (uint32_t)MAX_(0, (ssize_t)vertbits - (ssize_t)superres);
    const ssize_t horzsize = (ssize_t)((horzmask + 1) >> horzshift);
    const ssize_t vertsize = (ssize_t)((vertmask + 1) >> vertshift);
    const ssize_t width2 = horzsize ? (width / horzsize) : width;
    const ssize_t pixelsize = (horzsize && vertsize)
        ? (dstsize / (horzsize * vertsize)) : dstsize;
    ssize_t bitindex = 0;
    ssize_t dstindex = 0;
    ssize_t pixelindex = 0;
    ssize_t events = 0;
    uint32_t word = 0;
    uint32_t skip = 0;
    ssize_t v, h;

    if (src == NULL || (srcsize % 2) || dst == NULL ||
        height < 1 || width < 1 || nbits > 16 || nbits <= 8 ||
        skipbits < 4 || horzbits < 1 || vertbits < 1) {
        return EER_VALUE_ERROR;
    }

    if (superres > 0 && (width % horzsize || height % vertsize)) {
        return EER_VALUE_ERROR;
    }

    while (bitindex < srcbits) {
        word = 0;
        v = bitindex / 8;
        h = srcsize - v;
        memcpy(&word, src + v, h > 3 ? 4 : (size_t)h);
        word >>= bitindex % 8;
        skip = word & skipmask;
        pixelindex += (ssize_t)skip;
        if (pixelindex == pixelsize) {
            break;
        }
        if (pixelindex < 0) {
            return EER_INPUT_CORRUPT;
        }
        if (pixelindex > pixelsize) {
            return EER_OUTPUT_TOO_SMALL;
        }
        if (skip == skipmask) {
            bitindex += skipbits;
            continue;
        }
        if (superres == 0) {
            dstindex = pixelindex;
        } else {
            word >>= skipbits;
            h = (ssize_t)(((word & horzmask) ^ horzbits) >> horzshift);
            h += (pixelindex % width2) * horzsize;

            word >>= horzbits;
            v = (ssize_t)(((word & vertmask) ^ vertbits) >> vertshift);
            v += (pixelindex / width2) * vertsize;

            dstindex = v * width + h;
            if (dstindex >= dstsize) {
                return EER_OUTPUT_TOO_SMALL;
            }
        }
        dst[dstindex] += 1;
        pixelindex++;
        bitindex += nbits;
        events++;
    }
    return events;
}

ssize_t opencodecs_eer_decode_u2(
    const uint8_t* src, ssize_t srcsize,
    uint16_t* dst, ssize_t height, ssize_t width,
    uint32_t skipbits, uint32_t horzbits, uint32_t vertbits,
    uint32_t superres)
{
    const ssize_t dstsize = height * width;
    const ssize_t nbits = (ssize_t)(skipbits + horzbits + vertbits);
    const ssize_t srcbits = srcsize * 8 - nbits;
    const uint32_t skipmask = eer_bitmask((int)skipbits);
    const uint32_t horzmask = eer_bitmask((int)horzbits);
    const uint32_t vertmask = eer_bitmask((int)vertbits);
    const uint32_t horzshift =
        (uint32_t)MAX_(0, (ssize_t)horzbits - (ssize_t)superres);
    const uint32_t vertshift =
        (uint32_t)MAX_(0, (ssize_t)vertbits - (ssize_t)superres);
    const ssize_t horzsize = (ssize_t)((horzmask + 1) >> horzshift);
    const ssize_t vertsize = (ssize_t)((vertmask + 1) >> vertshift);
    const ssize_t width2 = horzsize ? (width / horzsize) : width;
    const ssize_t pixelsize = (horzsize && vertsize)
        ? (dstsize / (horzsize * vertsize)) : dstsize;
    ssize_t bitindex = 0;
    ssize_t dstindex = 0;
    ssize_t pixelindex = 0;
    ssize_t events = 0;
    uint32_t word = 0;
    uint32_t skip = 0;
    ssize_t v, h;

    if (src == NULL || (srcsize % 2) || dst == NULL ||
        height < 1 || width < 1 || nbits > 16 || nbits <= 8 ||
        skipbits < 4 || horzbits < 1 || vertbits < 1) {
        return EER_VALUE_ERROR;
    }

    if (superres > 0 && (width % horzsize || height % vertsize)) {
        return EER_VALUE_ERROR;
    }

    while (bitindex < srcbits) {
        word = 0;
        v = bitindex / 8;
        h = srcsize - v;
        memcpy(&word, src + v, h > 3 ? 4 : (size_t)h);
        word >>= bitindex % 8;
        skip = word & skipmask;
        pixelindex += (ssize_t)skip;
        if (pixelindex == pixelsize) {
            break;
        }
        if (pixelindex < 0) {
            return EER_INPUT_CORRUPT;
        }
        if (pixelindex > pixelsize) {
            return EER_OUTPUT_TOO_SMALL;
        }
        if (skip == skipmask) {
            bitindex += skipbits;
            continue;
        }
        if (superres == 0) {
            dstindex = pixelindex;
        } else {
            word >>= skipbits;
            h = (ssize_t)(((word & horzmask) ^ horzbits) >> horzshift);
            h += (pixelindex % width2) * horzsize;

            word >>= horzbits;
            v = (ssize_t)(((word & vertmask) ^ vertbits) >> vertshift);
            v += (pixelindex / width2) * vertsize;

            dstindex = v * width + h;
            if (dstindex >= dstsize) {
                return EER_OUTPUT_TOO_SMALL;
            }
        }
        dst[dstindex] += 1;
        pixelindex++;
        bitindex += nbits;
        events++;
    }
    return events;
}
