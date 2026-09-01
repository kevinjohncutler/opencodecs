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

/* ------------------------------------------------------------------ *
 * Encoder
 *
 * Straight LZW per TIFF 6.0 section 13, with the dictionary lookup on
 * an open-addressed hash table rather than the chained/tree structure
 * a textbook implementation uses.
 *
 * The lookup is the whole hot loop: once per input byte we ask "does
 * (prefix, next_byte) already have a code?". The key is only 20 bits
 * (12-bit prefix, 8-bit suffix), so it hashes cheaply and exactly.
 *
 * Sizing the table at 8192 keeps the load factor at 0.5 even when the
 * dictionary is full at 4094, which keeps probe chains near 1, and a
 * power of two lets the index be a mask instead of a modulo. The
 * multiply-shift hash takes the top 13 bits of a Knuth multiplicative
 * scramble, so all 20 key bits reach the index.
 *
 * CLEAR does not clear anything. The 20-bit key leaves 12 spare bits
 * in the 32-bit slot, so each slot carries the epoch it was written
 * in, and a reset is just epoch++. A slot is live only when its epoch
 * matches, which makes stale slots both "not found" and "free to
 * overwrite" in the same comparison the lookup already does, at no
 * extra load. Only when the epoch wraps after 4095 resets do we
 * actually memset. Incompressible input resets the table every ~3836
 * codes, so this is the difference between a 32 KiB memset per few
 * thousand bytes and a single increment.
 * ------------------------------------------------------------------ */

#define OC_LZW_CLEAR      256
#define OC_LZW_EOI        257
#define OC_LZW_FIRST      258
#define OC_LZW_MAX_CODE   4094   /* reset the table when we reach this */
#define OC_LZW_HSIZE      8192   /* power of two, 2x the dictionary    */
#define OC_LZW_HMASK      (OC_LZW_HSIZE - 1)
#define OC_LZW_KEY_BITS   20     /* 12-bit prefix | 8-bit suffix       */
#define OC_LZW_KEY_MASK   ((1u << OC_LZW_KEY_BITS) - 1u)
#define OC_LZW_EPOCH_MAX  (1u << (32 - OC_LZW_KEY_BITS))

size_t oc_tifflzw_encode_bound(size_t input_len)
{
    /* 12 bits per input byte worst case, plus a CLEAR every 3836 codes,
       plus CLEAR/EOI and a partial trailing byte. */
    return input_len + (input_len >> 1) + (input_len >> 8) + 64;
}

ptrdiff_t oc_tifflzw_encode(
    const uint8_t *input, size_t input_len,
    uint8_t *output, size_t output_len)
{
    if (output == NULL || (input == NULL && input_len > 0)) {
        return -1;
    }

    uint32_t *tags = (uint32_t *)calloc((size_t)OC_LZW_HSIZE, sizeof(uint32_t));
    uint16_t *codes = (uint16_t *)malloc((size_t)OC_LZW_HSIZE * sizeof(uint16_t));
    if (tags == NULL || codes == NULL) {
        free(tags);
        free(codes);
        return -1;
    }
    uint32_t epoch = 1;   /* slots start at 0, so every slot reads stale */

    uint8_t *op = output;
    uint8_t *const oend = output + output_len;

    uint32_t acc = 0;      /* MSB-first bit accumulator */
    int nacc = 0;          /* valid bits held in acc     */
    int width = 9;
    uint32_t next_code = OC_LZW_FIRST;

    /* Emitting is a macro so the hot loop keeps acc/nacc in registers
       rather than round-tripping them through a struct on every code. */
    #define OC_PUT(code)                                            \
        do {                                                        \
            acc = (acc << width) | (uint32_t)(code);                \
            nacc += width;                                          \
            while (nacc >= 8) {                                     \
                if (op >= oend) { free(tags); free(codes); return -2; } \
                nacc -= 8;                                          \
                *op++ = (uint8_t)(acc >> nacc);                     \
            }                                                       \
        } while (0)

    OC_PUT(OC_LZW_CLEAR);

    if (input_len > 0) {
        uint32_t prefix = input[0];

        for (size_t i = 1; i < input_len; i++) {
            const uint32_t k = input[i];
            const uint32_t key = (prefix << 8) | k;
            const uint32_t tag = (epoch << OC_LZW_KEY_BITS) | key;

            uint32_t slot = (key * 2654435761u) >> 19;
            uint32_t probe = tags[slot];
            while (probe != tag && (probe >> OC_LZW_KEY_BITS) == epoch) {
                slot = (slot + 1) & OC_LZW_HMASK;
                probe = tags[slot];
            }

            if (probe == tag) {
                prefix = codes[slot];
                continue;
            }

            /* Not in the table: emit what we have and extend. */
            OC_PUT(prefix);

            if (next_code < OC_LZW_MAX_CODE) {
                tags[slot] = tag;
                codes[slot] = (uint16_t)next_code;
                next_code++;
                /* The decoder widens when ITS next free code reaches
                   (1 << width) - 1, checked before it reads each code,
                   and its table lags this one by exactly one entry: it
                   cannot add an entry until it has read a second code
                   after CLEAR. So the encoder-side threshold that keeps
                   the two in step is one higher, the canonical
                   1 << width. Widening at (1 << width) - 1 here makes
                   the decoder read the next code one bit too wide. */
                if (next_code == (1u << width) && width < 12) {
                    width++;
                }
            } else {
                OC_PUT(OC_LZW_CLEAR);
                if (++epoch == OC_LZW_EPOCH_MAX) {   /* wrapped: real reset */
                    memset(tags, 0, (size_t)OC_LZW_HSIZE * sizeof(uint32_t));
                    epoch = 1;
                }
                width = 9;
                next_code = OC_LZW_FIRST;
            }
            prefix = k;
        }
        OC_PUT(prefix);
    }

    OC_PUT(OC_LZW_EOI);

    if (nacc > 0) {                      /* flush the partial byte */
        if (op >= oend) { free(tags); free(codes); return -2; }
        *op++ = (uint8_t)(acc << (8 - nacc));
    }

    #undef OC_PUT

    free(tags);
    free(codes);
    return (ptrdiff_t)(op - output);
}
