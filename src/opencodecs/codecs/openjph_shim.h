/* C-callable shim around OpenJPH's C++ ojph::codestream API.
 *
 * OpenJPH ships a class-based encoder/decoder where the caller pushes
 * one line of one component at a time via codestream::exchange() /
 * pull(). Wrapping that directly from Cython would require declaring
 * the full param_siz / param_cod / line_buf classes plus value-returning
 * accessors that lack default constructors. A thin C shim is far simpler.
 *
 * Each call here is self-contained: encode takes interleaved planar
 * pixel data + frame parameters and returns a malloc'd HTJ2K codestream
 * (caller frees via opencodecs_htj2k_free); decode reads info first,
 * then writes deinterleaved-into-planar bytes into a caller-provided
 * destination of the right size.
 *
 * Returns: 0 = success, non-zero = error. The last error message is
 * copied into a static thread_local buffer; query it with
 * opencodecs_htj2k_last_error().
 */

#ifndef OPENCODECS_HTJ2K_SHIM_H
#define OPENCODECS_HTJ2K_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encode planar input (one component at a time, in component order)
 * to an HTJ2K codestream.
 *
 *   src        : packed planar data — for component c, samples start at
 *                src + c * (width * height * bytes_per_sample).
 *   bytes_per_sample : 1 for bit_depth <= 8, 2 for bit_depth <= 16.
 *   reversible : 1 for mathematically lossless (5/3 reversible DWT),
 *                0 for irreversible (9/7 DWT) lossy mode.
 *   irrev_delta: quantization step for the irreversible mode.
 *                Ignored when reversible == 1. Typical range 1/256 ..
 *                1/4096; smaller -> closer to lossless.
 *   num_decomp : number of DWT decomposition levels (typical 5).
 *
 *   out_buf, out_size : on success, *out_buf is a malloc'd buffer of
 *                       *out_size bytes containing the codestream.
 *                       Caller must free via opencodecs_htj2k_free.
 */
int opencodecs_htj2k_encode(
    const void* src,
    int width,
    int height,
    int components,
    int bit_depth,
    int is_signed,
    int bytes_per_sample,
    int reversible,
    float irrev_delta,
    int num_decomp,
    void** out_buf,
    size_t* out_size
);

/* Reads the SIZ marker only; doesn't allocate or decode. */
int opencodecs_htj2k_decode_info(
    const void* src,
    size_t srcsize,
    int* width,
    int* height,
    int* components,
    int* bit_depth,
    int* is_signed
);

/* Decodes into a caller-allocated planar buffer of total
 * width * height * components * bytes_per_sample bytes (component
 * planes back-to-back). */
int opencodecs_htj2k_decode(
    const void* src,
    size_t srcsize,
    void* dst,
    size_t dst_size,
    int bytes_per_sample
);

void opencodecs_htj2k_free(void* buf);

const char* opencodecs_htj2k_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
