/* oc_eer.c — EER decoder. See oc_eer.h for the bitstream layout. */

#include "oc_eer.h"

#include <string.h>

/* LSB-first bit reader over a 64-bit window.
 *
 * Every event needs at most skipbits + horzbits + vertbits = 14 + 4 + 4
 * = 22 bits, so one refill per event is always enough and the hot loop
 * never re-checks the source pointer mid-event. Refilling 8 bytes at a
 * time rather than reading a 32-bit word at a bit offset also means we
 * never read past the end of the caller's buffer, which the obvious
 * unaligned-load version does. */
typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    uint64_t acc;      /* bits, LSB-first */
    int nbits;         /* valid bits in acc */
} oc_eer_bits;

static inline void oc_eer_refill(oc_eer_bits *b)
{
    while (b->nbits <= 56 && b->p < b->end) {
        b->acc |= (uint64_t)(*b->p++) << b->nbits;
        b->nbits += 8;
    }
}

static inline uint32_t oc_eer_take(oc_eer_bits *b, unsigned n)
{
    const uint32_t v = (uint32_t)(b->acc & (((uint64_t)1 << n) - 1));
    b->acc >>= n;
    b->nbits -= (int)n;
    return v;
}

/* The two entry points differ only in the destination type, and the
 * inner loop is identical, so it lives in a macro rather than being
 * duplicated or paid for with a function pointer per event. */
#define OC_EER_DECODE_BODY(DSTTYPE, DSTMAX)                                  \
    do {                                                                     \
        if (dst == NULL || (src == NULL && srcsize > 0)) {                   \
            return OC_EER_VALUE_ERROR;                                       \
        }                                                                    \
        if (skipbits < 1 || skipbits > 14 ||                                 \
            horzbits < 1 || horzbits > 4 ||                                  \
            vertbits < 1 || vertbits > 4) {                                  \
            return OC_EER_VALUE_ERROR;                                       \
        }                                                                    \
        /* superres consumes that many top bits of each field */             \
        if (superres > horzbits || superres > vertbits) {                    \
            return OC_EER_VALUE_ERROR;                                       \
        }                                                                    \
        const size_t factor = (size_t)1 << superres;                         \
        if (height % factor || width % factor || height == 0 || width == 0) {\
            return OC_EER_VALUE_ERROR;                                       \
        }                                                                    \
                                                                             \
        const size_t base_w = width / factor;                                \
        const size_t base_h = height / factor;                               \
        const size_t ncells = base_w * base_h;                               \
        const uint32_t maxskip = (uint32_t)((1u << skipbits) - 1u);          \
        const uint32_t hflip = 1u << (horzbits - 1);                         \
        const uint32_t vflip = 1u << (vertbits - 1);                         \
        const unsigned hshift = horzbits - superres;                         \
        const unsigned vshift = vertbits - superres;                         \
        const unsigned subbits = horzbits + vertbits;                        \
                                                                             \
        oc_eer_bits b;                                                       \
        b.p = src; b.end = src + srcsize; b.acc = 0; b.nbits = 0;            \
                                                                             \
        /* Track the base row/column alongside the linear position so the    \
         * per-event divide becomes an add plus a rarely-taken fixup: a      \
         * skip cannot exceed 2^14 and base_w is thousands wide, so the      \
         * carry loop almost never runs more than once. */                   \
        size_t pos = 0, row = 0, col = 0;                                    \
        ptrdiff_t nevents = 0;                                               \
                                                                             \
        for (;;) {                                                           \
            oc_eer_refill(&b);                                               \
            /* Require a whole event to remain. An EER frame is byte      \
             * aligned, so up to 7 spare bits are ordinary padding that       \
             * would otherwise read as a plausible skip field and             \
             * manufacture a final phantom event. Real frames terminate on    \
             * the exact-fill rule below long before this fires. */           \
            if (b.nbits < (int)(skipbits + subbits)) {                       \
                break;                                                       \
            }                                                                \
            const uint32_t s = oc_eer_take(&b, skipbits);                    \
            pos += s; col += s;                                              \
            while (col >= base_w) { col -= base_w; row++; }                  \
            if (s == maxskip) {                                              \
                continue;           /* advance only, no event */             \
            }                                                                \
            const uint32_t h = oc_eer_take(&b, horzbits) ^ hflip;            \
            const uint32_t v = oc_eer_take(&b, vertbits) ^ vflip;            \
            if (pos >= ncells) {                                             \
                /* A well-formed frame terminates by landing its final skip  \
                 * EXACTLY on the last cell, so pos == ncells is the normal   \
                 * end and any trailing bits are the frame footer. Every one  \
                 * of the 12 EMPIAR-10568 frames ends this way, with 1 to 120 \
                 * bits left over, which is why "bits remain" cannot be used  \
                 * to detect a bad shape. Overshooting means the events do    \
                 * not fit the caller's dimensions. */                        \
                if (pos > ncells) {                                          \
                    return OC_EER_OUTPUT_TOO_SMALL;                          \
                }                                                            \
                break;                                                       \
            }                                                                \
            const size_t x = col * factor + (h >> hshift);                   \
            const size_t y = row * factor + (v >> vshift);                   \
            DSTTYPE *cell = &dst[y * width + x];                             \
            if (*cell < (DSTMAX)) { (*cell)++; }                             \
            nevents++;                                                       \
            pos++; col++;                                                    \
            if (col >= base_w) { col -= base_w; row++; }                     \
        }                                                                    \
        return nevents;                                                      \
    } while (0)

ptrdiff_t oc_eer_decode_u8(
    const uint8_t *src, size_t srcsize,
    uint8_t *dst, size_t height, size_t width,
    unsigned skipbits, unsigned horzbits, unsigned vertbits,
    unsigned superres)
{
    OC_EER_DECODE_BODY(uint8_t, 0xFFu);
}

ptrdiff_t oc_eer_decode_u16(
    const uint8_t *src, size_t srcsize,
    uint16_t *dst, size_t height, size_t width,
    unsigned skipbits, unsigned horzbits, unsigned vertbits,
    unsigned superres)
{
    OC_EER_DECODE_BODY(uint16_t, 0xFFFFu);
}

#undef OC_EER_DECODE_BODY
