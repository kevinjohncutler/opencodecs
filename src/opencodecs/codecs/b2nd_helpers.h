/* Thin C helper bridging Cython to c-blosc2's b2nd API.
 *
 * b2nd is the multidimensional layer of c-blosc2: an ndarray serialized
 * to a single self-contained "cframe" byte buffer. The cframe stores
 * shape, dtype, chunk shape, and the compressed chunks. Decoding
 * reconstructs the array from the cframe alone — no out-of-band shape
 * info needed.
 *
 * The b2nd public API requires populating blosc2_cparams + blosc2_storage
 * structs, which are too big to declare in Cython without copying every
 * field. These helpers hide that struct work behind small functions
 * that Cython can call directly.
 */

#ifndef OC_B2ND_HELPERS_H
#define OC_B2ND_HELPERS_H

#include <stdint.h>
#include "b2nd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum supported number of dimensions. b2nd's internal cap is
 * B2ND_MAX_DIM (currently 8); we expose the same. */
#define OC_B2ND_MAX_DIM 8

/* Encode a contiguous C buffer as a b2nd cframe.
 *
 * Parameters:
 *   data         - input bytes (ndim-dimensional in row-major order)
 *   data_size    - size of input in bytes
 *   ndim         - number of dimensions (1..OC_B2ND_MAX_DIM)
 *   shape        - shape of the input (length=ndim)
 *   itemsize     - bytes per element (1, 2, 4, 8, ...)
 *   dtype_str    - NumPy dtype string (e.g. "<u2"); may be NULL
 *   level        - compression level 0..9 (5 = library default)
 *   compressor   - inner blosc2 codec ("zstd","lz4","lz4hc","blosclz","zlib"); NULL = default
 *   do_bitshuffle - 0 = byte shuffle, 1 = bit shuffle, -1 = no shuffle
 *
 * On success returns 0, sets *out_cframe to a freshly allocated buffer
 * (caller must free via free()) and *out_cframe_len to its length.
 * Returns a negative blosc2 error code on failure.
 */
int oc_b2nd_encode(
    const void* data,
    int64_t data_size,
    int8_t ndim,
    const int64_t* shape,
    int32_t itemsize,
    const char* dtype_str,
    int level,
    const char* compressor,
    int do_bitshuffle,
    uint8_t** out_cframe,
    int64_t* out_cframe_len
);

/* Inspect a cframe's metadata without decompressing the data.
 *
 * Fills *out_ndim, out_shape (length OC_B2ND_MAX_DIM), out_itemsize.
 * out_dtype receives a pointer to dtype string memory owned by an
 * internal b2nd_array_t, valid until oc_b2nd_meta_release is called.
 *
 * Returns 0 on success, negative blosc2 error code otherwise.
 *
 * The caller must call oc_b2nd_meta_release(handle) when done.
 */
int oc_b2nd_inspect(
    const void* cframe,
    int64_t cframe_len,
    int8_t* out_ndim,
    int64_t* out_shape,
    int32_t* out_itemsize,
    char** out_dtype,
    void** handle
);

/* Release the b2nd_array_t handle returned by oc_b2nd_inspect. */
void oc_b2nd_release(void* handle);

/* Decode a cframe into a pre-allocated buffer.
 *
 * The caller must size the buffer to product(shape)*itemsize bytes
 * (same values inspect() reports).
 *
 * Returns 0 on success, negative blosc2 error code on failure.
 */
int oc_b2nd_decode(
    const void* cframe,
    int64_t cframe_len,
    void* dest_buffer,
    int64_t dest_buffer_size
);

#ifdef __cplusplus
}
#endif

#endif /* OC_B2ND_HELPERS_H */
