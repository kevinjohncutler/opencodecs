/* EER (Electron Event Representation) decoder — header.
 *
 * Self-contained excerpt from imagecodecs' imcd.c (BSD-3, Christoph
 * Gohlke, see LICENSE in this directory). Just the two EER entry
 * points + the bitmask helper they need, so opencodecs can ship a
 * native EER decoder without a runtime dep on imagecodecs.
 *
 * Format: ISO/IEC bitstream of variable-length events; each event
 * encodes a "skip" count (gap from previous event) plus, in super-
 * resolution mode, sub-pixel horizontal & vertical offsets.
 *   skipbits = 7..14 (number of bits for the skip field)
 *   horzbits = 1..4  (number of bits for the sub-pixel H offset)
 *   vertbits = 1..4  (number of bits for the sub-pixel V offset)
 * superres = 0 -> output is binary event count per pixel
 * superres > 0 -> output is event count per super-resolution sub-pixel
 *
 * Output is a tightly-packed (height, width) raster of uint8 or
 * uint16; the decoder increments cells where events landed.
 */

#ifndef OPENCODECS_EER_H
#define OPENCODECS_EER_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EER_OK 0
#define EER_VALUE_ERROR -5
#define EER_INPUT_CORRUPT -6
#define EER_OUTPUT_TOO_SMALL -7

ssize_t opencodecs_eer_decode_u1(
    const uint8_t* src, ssize_t srcsize,
    uint8_t* dst, ssize_t height, ssize_t width,
    uint32_t skipbits, uint32_t horzbits, uint32_t vertbits,
    uint32_t superres
);

ssize_t opencodecs_eer_decode_u2(
    const uint8_t* src, ssize_t srcsize,
    uint16_t* dst, ssize_t height, ssize_t width,
    uint32_t skipbits, uint32_t horzbits, uint32_t vertbits,
    uint32_t superres
);

#ifdef __cplusplus
}
#endif

#endif
