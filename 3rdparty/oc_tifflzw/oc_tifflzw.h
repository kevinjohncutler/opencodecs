/* oc_tifflzw.h — TIFF-flavor LZW decoder.
 *
 * TIFF's LZW (spec section 13 in TIFF 6.0) differs from GIF's in three
 * places that the inner decode loop has to handle:
 *
 *   1. Bits are packed MSB-first within each byte (GIF is LSB-first).
 *   2. Clear code = 256, EOI = 257 (fixed); initial width = 9 bits.
 *   3. Code width grows BEFORE reading the next code when
 *      ``next_code == (1 << width) - 1`` — a historical off-by-one
 *      quirk vs canonical LZW.
 *
 * Otherwise the dictionary management is identical to GIF's, so the
 * same flat prefix/suffix/first_byte tables + stack-based emit apply.
 *
 * MIT license: see LICENSE in this directory.
 */

#ifndef OC_TIFFLZW_H
#define OC_TIFFLZW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode a TIFF LZW strip / tile from a contiguous compressed buffer
 * into a pre-allocated output buffer.
 *
 *  input / input_len: the compressed LZW bytes (no sub-block prefixes).
 *  output: pre-allocated for the expected uncompressed size.
 *  output_len: input expected_size (decoder stops at the first EOI 257
 *              or end of input or when this many bytes have been
 *              emitted).
 *
 * Returns the number of bytes written, or a negative error:
 *  -1 truncated input.
 *  -2 invalid LZW code (refers to entry not yet defined).
 *  -3 output overrun (more pixels than buffer).
 */
ptrdiff_t oc_tifflzw_decode(
    const uint8_t *input, size_t input_len,
    uint8_t *output, size_t output_len);

#ifdef __cplusplus
}
#endif

#endif /* OC_TIFFLZW_H */
