/* Self-contained TIFF LZW encoder excerpted from imagecodecs' imcd.c
 * (BSD-3, Christoph Gohlke). The LZW decoder is implemented directly
 * in opencodecs.codecs._tiff.pyx (pure Cython); only the encoder is
 * vendored here so TiffWriter can produce LZW-compressed strips and
 * tiles without a runtime dep on imagecodecs.
 *
 * See LICENSE alongside this header.
 */

#ifndef OPENCODECS_IMCD_LZW_H
#define OPENCODECS_IMCD_LZW_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
  /* MSVC has no <sys/types.h>::ssize_t; the canonical equivalent is
     SSIZE_T from BaseTsd.h. */
  #include <BaseTsd.h>
  typedef SSIZE_T ssize_t;
#else
  #include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Worst-case encoded size for a given input. Slight over-estimate so
 * a malloc with this size never overflows during encode. */
ssize_t opencodecs_lzw_encode_size(ssize_t srcsize);

/* Encode srcsize bytes from src into dst (capacity dstsize). Returns
 * the number of bytes written, or a negative error code:
 *   -5 = invalid argument
 *   -2 = out of memory
 *   -7 = output buffer too small
 */
ssize_t opencodecs_lzw_encode(
    const uint8_t* src, ssize_t srcsize,
    uint8_t* dst, ssize_t dstsize
);

#ifdef __cplusplus
}
#endif

#endif
