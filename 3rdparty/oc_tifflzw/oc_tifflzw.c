/* oc_tifflzw.c — opencodecs's fast TIFF LZW decoder.
 *
 * MIT license. (c) 2026 opencodecs authors.
 *
 * Performance vs. the previous _tiff.pyx pure-Cython implementation,
 * which built one PyMem_Malloc'd string per dictionary entry and
 * memcpy'd the whole previous string on each new entry — O(N^2) cost
 * for any frame with long string entries. This decoder uses flat
 * prefix/suffix/first_byte tables (matching oc_giflzw) so each new
 * entry is O(1) to add and string emit is O(string_length).
 */

#include "oc_tifflzw.h"

#include <stdlib.h>
#include <string.h>

#define OC_LZW_MAX_CODES   4096
#define OC_LZW_STACK_SIZE  4096

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_expect)
#define OC_LIKELY(x)   __builtin_expect(!!(x), 1)
#define OC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define OC_LIKELY(x)   (x)
#define OC_UNLIKELY(x) (x)
#endif

#define TIFF_CLEAR_CODE 256
#define TIFF_EOI_CODE   257
#define TIFF_INIT_WIDTH 9


ptrdiff_t oc_tifflzw_decode(
    const uint8_t *input, size_t input_len,
    uint8_t *output, size_t output_len)
{
    /* Dictionary — flat arrays so each new entry is O(1) to add. */
    uint16_t prefix[OC_LZW_MAX_CODES];
    uint8_t  suffix[OC_LZW_MAX_CODES];
    uint8_t  first_byte[OC_LZW_MAX_CODES];

    /* Initialise literals 0..255. */
    for (int i = 0; i < 256; i++) {
        prefix[i] = 0xFFFF;
        suffix[i] = (uint8_t) i;
        first_byte[i] = (uint8_t) i;
    }

    int code_size = TIFF_INIT_WIDTH;
    int next_code = TIFF_EOI_CODE + 1;   /* = 258 */
    int prev_code = -1;

    /* Auto-detect bit ordering from the first byte. In an MSB-first
     * 9-bit stream the first code (typically CLEAR=256 = 0x100)
     * encodes as a byte whose high bit is set (0x80-0xFF). In the
     * old-style LSB-first variant the first byte's high bit is 0. */
    int lsb_first = 0;
    if (input_len > 0 && (input[0] & 0x80) == 0) {
        lsb_first = 1;
    }

    /* Bit accumulator. Layout depends on lsb_first. */
    uint64_t accum = 0;
    int accum_bits = 0;
    size_t in_pos = 0;
    uint8_t *out_p = output;
    uint8_t *out_end = output + output_len;

    uint8_t stack[OC_LZW_STACK_SIZE];

    for (;;) {
        /* TIFF LZW has two encoder dialects, both legal:
         *   * "Early-change" (libtiff modern): grow width when
         *     next_code == (1 << code_size) - 1.
         *   * "Late-change" (post-Welch-canonical, used by GhostScript,
         *     libtiff old, and the libtiff sample set): grow when
         *     next_code == (1 << code_size).
         * Both share CompressionTag=5 in the IFD. We pair the
         * transition rule with the bit-order auto-detection: the
         * old-style LSB-first variant ships with late-change; the
         * post-TIFF-6.0 MSB-first variant ships with early-change.
         * Check BEFORE reading the next code so the read happens at
         * the post-grow width. */
        int width_trigger = lsb_first
            ? (1 << code_size)            /* late-change */
            : ((1 << code_size) - 1);     /* early-change */
        if (code_size < 12 && next_code == width_trigger) {
            code_size++;
        }

        /* Refill the bit accumulator. MSB-first appends new bytes
         * into the LOW bits and shifts the old contents UP; we then
         * extract from the TOP. LSB-first appends new bytes into the
         * HIGH bits (positioned by accum_bits) and shifts down; we
         * extract from the BOTTOM. */
        while (accum_bits < code_size) {
            if (OC_UNLIKELY(in_pos >= input_len)) {
                if (out_p == out_end) return (ptrdiff_t)(out_p - output);
                return -1;
            }
            if (lsb_first) {
                accum |= (uint64_t) input[in_pos++] << accum_bits;
            } else {
                accum = (accum << 8) | (uint64_t) input[in_pos++];
            }
            accum_bits += 8;
        }

        int code;
        if (lsb_first) {
            code = (int)(accum & ((1u << code_size) - 1));
            accum >>= code_size;
            accum_bits -= code_size;
        } else {
            code = (int)((accum >> (accum_bits - code_size))
                         & ((1u << code_size) - 1));
            accum_bits -= code_size;
            accum &= (1ULL << accum_bits) - 1;
        }

        if (OC_UNLIKELY(code == TIFF_EOI_CODE)) {
            return (ptrdiff_t)(out_p - output);
        }
        if (OC_UNLIKELY(code == TIFF_CLEAR_CODE)) {
            code_size = TIFF_INIT_WIDTH;
            next_code = TIFF_EOI_CODE + 1;
            prev_code = -1;
            continue;
        }

        /* Emit the string for `code` onto our local stack (reversed),
         * then drain into output (forward). */
        int sp = 0;
        int c = code;

        if (OC_UNLIKELY(c >= next_code)) {
            /* K-w-K special case: code refers to a dict entry we're
             * about to add. Synthesize: prev string + first byte of
             * prev string. */
            if (OC_UNLIKELY(prev_code < 0)) return -2;
            stack[sp++] = first_byte[prev_code];
            c = prev_code;
        }

        while (c >= 256) {
            if (OC_UNLIKELY(sp >= OC_LZW_STACK_SIZE)) return -2;
            /* Guard against a corrupt chain (prefix table entry past
             * OC_LZW_MAX_CODES) before the suffix/prefix reads OOB. */
            if (OC_UNLIKELY(c >= OC_LZW_MAX_CODES)) return -2;
            stack[sp++] = suffix[c];
            c = prefix[c];
        }
        stack[sp++] = (uint8_t) c;
        uint8_t first = (uint8_t) c;

        if (OC_UNLIKELY(out_p + sp > out_end)) return -3;
        for (int i = sp - 1; i >= 0; i--) {
            *out_p++ = stack[i];
        }

        /* Add new dict entry: prev_code → first byte of new string. */
        if (prev_code >= 0 && next_code < OC_LZW_MAX_CODES) {
            prefix[next_code] = (uint16_t) prev_code;
            suffix[next_code] = first;
            first_byte[next_code] = first_byte[prev_code];
            next_code++;
        }

        prev_code = code;

        if (OC_UNLIKELY(out_p == out_end)) {
            return (ptrdiff_t)(out_p - output);
        }
    }
}
