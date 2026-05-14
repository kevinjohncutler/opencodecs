/* oc_giflzw.h — GIF-LZW decoder optimized for opencodecs.
 *
 * Decodes a stream of raw LZW-compressed sub-blocks (as produced by
 * giflib's DGifGetCode / DGifGetCodeNext) directly to palette indices.
 * Bypasses libgif's reference LZW which benchmarks ~30% slower than
 * Pillow's vendored implementation.
 *
 * Design choices:
 * - Bit accumulator in uint64_t (refill once, drain several codes).
 * - Suffix + prefix tables only — no per-code string buffer; we walk
 *   the prefix chain backwards onto a stack, then emit reversed.
 * - "First-byte" cache per code so the very first byte of any string
 *   is O(1) instead of O(string length).
 *
 * MIT license: see LICENSE in the same directory.
 */

#ifndef OC_GIFLZW_H
#define OC_GIFLZW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sub-block: a chunk of compressed LZW bytes inline in the GIF stream.
 * GIF sub-blocks are at most 255 bytes; a decoder gets fed an array of
 * (ptr, len) pairs back-to-back from DGifGetCode / DGifGetCodeNext. */
typedef struct {
    const uint8_t *data;
    size_t len;
} oc_giflzw_block_t;

/* Decode an array of sub-blocks (concatenated logically) into a
 * pre-allocated palette-index output buffer.
 *
 *  min_code_size: from the GIF image descriptor (typically 8 for
 *                 256-color palettes, smaller for low-color palettes).
 *  blocks / n_blocks: array of compressed sub-blocks.
 *  output: pre-allocated buffer of `output_len` bytes (= width*height).
 *
 * Return value:
 *   0 on success.
 *  -1 invalid min_code_size.
 *  -2 truncated input (ran out before EOI).
 *  -3 output overrun (image had more pixels than buffer).
 *  -4 invalid code referenced before it was defined.
 */
int oc_giflzw_decode_blocks(
    int min_code_size,
    const oc_giflzw_block_t *blocks, size_t n_blocks,
    uint8_t *output, size_t output_len);

/* Streaming alternative: decode from a single contiguous buffer of raw
 * LZW bytes (caller has already concatenated all sub-blocks). Saves a
 * blocks array on the common path where giflib's DGifGetCode loop
 * accumulates the bytes anyway.
 */
int oc_giflzw_decode(
    int min_code_size,
    const uint8_t *input, size_t input_len,
    uint8_t *output, size_t output_len);

#ifdef __cplusplus
}
#endif

#endif /* OC_GIFLZW_H */
