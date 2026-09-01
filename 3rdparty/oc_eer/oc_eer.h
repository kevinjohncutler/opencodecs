/* oc_eer.h — EER (Electron Event Representation) decoder.
 *
 * EER is the on-camera format of Thermo Fisher Falcon 4 / 4i direct
 * electron detectors. A frame is a bitstream of electron events: each
 * event is a run-length gap from the previous event, then a sub-pixel
 * position within the landing pixel. Frames are wrapped in a TIFF with
 * private compression tags 65000 / 65001 / 65002.
 *
 * Bitstream, LSB-first within each byte:
 *
 *   skip   : skipbits           gap in scanline order from the last event
 *   horz   : horzbits           sub-pixel column, LOW bits of the symbol
 *   vert   : vertbits           sub-pixel row, HIGH bits of the symbol
 *
 * A skip field of all ones means "advance by that much and read another
 * skip", with no event and no sub-pixel fields. That is how gaps larger
 * than the field can represent are encoded.
 *
 * Sub-pixel convention: the TOP bit of each field is inverted. The
 * reference implementation (RELION renderEER.cpp) expresses this as an
 * XOR over the packed symbol, and its two supported widths agree on the
 * rule:
 *
 *   4-bit symbol (2 horz + 2 vert):  s ^= 0x0A   ->  0b1010
 *   2-bit symbol (1 horz + 1 vert):  s ^= 0x03   ->  0b11
 *
 * Both constants flip exactly the MSB of each field and leave the low
 * bits alone, so the rule is width-independent and we apply it for all
 * widths. NOTE: imagecodecs applies the inversion for field widths 1 and
 * 2 but not 3 and 4; real Falcon hardware only ever emits 1 or 2, so
 * that divergence does not affect real data. See
 * tests/test_eer.py::test_eer_subpixel_inversion_is_width_independent.
 *
 * Super-resolution: with factor F = 1 << superres, the top `superres`
 * bits of each (inverted) field select the sub-position, so
 *
 *   x = (pos % base_width)  * F + (horz >> (horzbits - superres))
 *   y = (pos / base_width)  * F + (vert >> (vertbits - superres))
 *
 * which degenerates correctly to plain raster indexing at superres = 0.
 *
 * MIT license: see LICENSE in this directory.
 */

#ifndef OC_EER_H
#define OC_EER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_EER_OK                0
#define OC_EER_VALUE_ERROR      -5   /* bad parameters */
#define OC_EER_INPUT_CORRUPT    -6
#define OC_EER_OUTPUT_TOO_SMALL -7   /* an event landed past the frame   */

/* Decode one EER frame, incrementing the cell each event lands on.
 * Counts saturate rather than wrap, so a hot pixel cannot roll over.
 *
 * height / width are the OUTPUT dimensions and must both be divisible
 * by (1 << superres). Returns the number of events decoded, or one of
 * the negative codes above.
 */
ptrdiff_t oc_eer_decode_u8(
    const uint8_t *src, size_t srcsize,
    uint8_t *dst, size_t height, size_t width,
    unsigned skipbits, unsigned horzbits, unsigned vertbits,
    unsigned superres);

ptrdiff_t oc_eer_decode_u16(
    const uint8_t *src, size_t srcsize,
    uint16_t *dst, size_t height, size_t width,
    unsigned skipbits, unsigned horzbits, unsigned vertbits,
    unsigned superres);

#ifdef __cplusplus
}
#endif

#endif /* OC_EER_H */
