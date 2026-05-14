/* webp_shim.h — minimal facade over libwebp's advanced encode API.
 *
 * libwebp's simple WebPEncodeRGB/RGBA helpers don't expose
 * WebPConfig.thread_level — that knob lives in the advanced API
 * (WebPPicture + WebPMemoryWriter + WebPEncode). Wrapping all of that
 * directly in Cython adds lots of struct declarations that we only
 * need internally. This shim hides it behind a single C function.
 */
#ifndef OPENCODECS_WEBP_SHIM_H
#define OPENCODECS_WEBP_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Encode (rgb_buf, w, h, stride) → newly-allocated bitstream.
 *
 *  has_alpha: 0 = RGB, 1 = RGBA
 *  lossless:  0 = lossy, 1 = near-lossless
 *  quality:   0..100 (lossy only)
 *  thread_level: 0 = single-thread (libwebp default), 1 = enable workers
 *  method:    0..6 (lossy) speed/quality tradeoff. -1 = libwebp default (4)
 *
 * On success returns 0 and writes ``*out_ptr`` / ``*out_size``.
 * Caller MUST free ``*out_ptr`` via ``oc_webp_free``.
 * On failure returns a libwebp WebPEncodingError code (> 0) and leaves
 * ``*out_ptr`` NULL.
 */
int oc_webp_encode(
    const uint8_t *rgb_buf, int width, int height, int stride,
    int has_alpha, int lossless, float quality,
    int thread_level, int method,
    uint8_t **out_ptr, size_t *out_size);

void oc_webp_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
