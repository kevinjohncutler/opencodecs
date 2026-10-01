/* C-callable shim around OpenJPH's C++ ojph::codestream API.
 *
 * OpenJPH ships a class-based encoder/decoder where the caller pushes
 * one line of one component at a time via codestream::exchange() /
 * pull(). Wrapping that directly from Cython would require declaring
 * the full param_siz / param_cod / line_buf classes plus value-returning
 * accessors that lack default constructors. A thin C shim is far simpler.
 *
 * Each call here is self-contained: encode takes pixel data in either
 * component-interleaved (H, W, C) or planar (C, H, W) order plus frame
 * parameters and returns a malloc'd HTJ2K codestream (caller frees via
 * opencodecs_htj2k_free); decode reads info first, then writes the
 * samples into a caller-provided destination of the right size, in
 * either order.
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

/* Everything the encoder needs besides the pixels.
 *
 *   bytes_per_sample : 1, 2 or 4. Samples are read as (u)int8, (u)int16
 *                      or (u)int32 according to is_signed. A float32
 *                      image is passed as its int32 bit pattern with
 *                      is_signed = 1 and nlt_binary_complement = 1.
 *   bit_depth        : precision written to SIZ, 1..32.
 *   src_planar       : 1 when src holds whole component planes back to
 *                      back (C, H, W); 0 for interleaved (H, W, C).
 *   reversible       : 1 for the reversible 5/3 path (lossless),
 *                      0 for the irreversible 9/7 path.
 *   irrev_delta      : quantization step for the irreversible path;
 *                      <= 0 leaves OpenJPH's default.
 *   qfactor          : 1..100 sets a JPEG-style quality factor on the
 *                      irreversible path instead of irrev_delta; 0 = off.
 *   num_decomp       : DWT decomposition levels; < 0 keeps the default.
 *   color_transform  : 1 to apply the component transform (RCT on the
 *                      reversible path, ICT on the irreversible one) to
 *                      components 0..2, signaled in COD SGcod.
 *   nlt_binary_complement : 1 to write an NLT marker of type 3 for all
 *                      components (ISO/IEC 15444-2), which is how
 *                      floating point samples are carried.
 *   tile_w, tile_h   : tile size; 0 for a single tile.
 *   tlm              : 1 to write a TLM marker.
 *   tilepart_resolutions, tilepart_components : tile-part divisions.
 *   block_w, block_h : code-block size; 0 keeps 64 x 64.
 *   prog_order       : "LRCP", "RLCP", "RPCL", "PCRL", "CPRL" or NULL.
 *   profile          : "IMF", "BROADCAST" or NULL.
 */
typedef struct {
    int width;
    int height;
    int components;
    int bit_depth;
    int is_signed;
    int bytes_per_sample;
    int src_planar;
    int reversible;
    float irrev_delta;
    int qfactor;
    int num_decomp;
    int color_transform;
    int nlt_binary_complement;
    int tile_w;
    int tile_h;
    int tlm;
    int tilepart_resolutions;
    int tilepart_components;
    int block_w;
    int block_h;
    const char* prog_order;
    const char* profile;
} opencodecs_htj2k_encode_params;

/* On success, *out_buf is a malloc'd buffer of *out_size bytes holding
 * the codestream. Caller must free it via opencodecs_htj2k_free. */
int opencodecs_htj2k_encode(
    const void* src,
    const opencodecs_htj2k_encode_params* params,
    void** out_buf,
    size_t* out_size
);

/* What the main header says, read without decoding any samples. */
typedef struct {
    int width;              /* reconstructed extent at the reduction */
    int height;
    int components;
    int bit_depth;          /* of component 0 */
    int is_signed;          /* of component 0 */
    int num_decompositions;
    int color_transform;    /* COD SGcod multiple component transform */
    int nlt_type;           /* NLT of component 0: 0 none, 3 binary complement */
    int uniform;            /* 1 when every component shares bit depth,
                               signedness, NLT and has no subsampling */
} opencodecs_htj2k_info;

/* `reduce_data` and `reduce_recon` are OpenJPH's
 * restrict_input_resolution() arguments: how many of the finest
 * resolutions to leave unread, and how many to leave out of the
 * reconstruction. Equal values shrink the output image by 2**n; the
 * reported width/height are the reconstructed extent, so a caller can
 * enumerate a pyramid's level shapes without decoding any of them. */
int opencodecs_htj2k_decode_info(
    const void* src,
    size_t srcsize,
    int reduce_data,
    int reduce_recon,
    int resilient,
    opencodecs_htj2k_info* info
);

/* Decodes into a caller-allocated buffer of
 * width * height * components * bytes_per_sample bytes, where
 * width/height are the reconstructed extent; query
 * opencodecs_htj2k_decode_info with the same reductions to size it.
 *
 * dst_planar = 1 writes component planes back to back (C, H, W);
 * 0 writes interleaved (H, W, C). Either works whatever order the
 * codestream is pulled in: a codestream that uses the component
 * transform has to be pulled one row of every component at a time,
 * and this takes care of that.
 *
 * Samples are clamped to the component's nominal range, which is what
 * ISO/IEC 15444-1 Annex G.1.2 calls the typical treatment of
 * quantization overshoot and what OpenJPH's own ojph_expand does.
 * Thirty-two bit samples are copied bit for bit. */
int opencodecs_htj2k_decode(
    const void* src,
    size_t srcsize,
    void* dst,
    size_t dst_size,
    int bytes_per_sample,
    int reduce_data,
    int reduce_recon,
    int resilient,
    int dst_planar
);

void opencodecs_htj2k_free(void* buf);

const char* opencodecs_htj2k_last_error(void);

/* Warnings OpenJPH raised during the most recent call, newline-separated
 * and empty when there were none. Call clear_warnings() before a decode
 * to start a fresh collection; it also installs the collector, which
 * keeps OpenJPH from printing to the process's stdout.
 *
 * These matter because OpenJPH treats several unimplemented marker
 * segments as warnings and keeps decoding, so a caller can otherwise be
 * handed an image the decoder knows it did not fully read. */
const char* opencodecs_htj2k_last_warnings(void);
void opencodecs_htj2k_clear_warnings(void);

#ifdef __cplusplus
}
#endif

#endif
