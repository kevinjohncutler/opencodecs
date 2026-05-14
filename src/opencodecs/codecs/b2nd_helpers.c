/* Implementation of oc_b2nd_helpers — see b2nd_helpers.h. */

#include <stdlib.h>
#include <string.h>

#include "blosc2.h"
#include "b2nd.h"
#include "b2nd_helpers.h"


/* Map compressor name to blosc2 compcode. -1 = unknown -> default. */
static int oc_compcode_from_name(const char* name) {
    if (name == NULL) return -1;
    return blosc2_compname_to_compcode(name);
}


/* blosc2 needs blosc2_init() called once before any frame I/O backend
 * registration is available. The library is reference-counted internally,
 * so calling more than once is safe — but we still gate to avoid the
 * tiny atomic overhead. blosc2_destroy() is intentionally NOT paired
 * here: we rely on Python's process-lifetime cleanup. */
static int oc_blosc2_initialized = 0;

static void oc_ensure_blosc2_init(void) {
    if (!oc_blosc2_initialized) {
        blosc2_init();
        oc_blosc2_initialized = 1;
    }
}


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
) {
    if (ndim < 1 || ndim > OC_B2ND_MAX_DIM) return -1;
    if (data == NULL || shape == NULL) return -1;
    if (out_cframe == NULL || out_cframe_len == NULL) return -1;

    oc_ensure_blosc2_init();

    /* Choose chunk shape == full array (single-chunk) for simplicity.
     * b2nd internally splits chunks into blocks; we keep the array as
     * one chunk with a default block shape derived per-dimension. */
    int32_t chunkshape[OC_B2ND_MAX_DIM] = {0};
    int32_t blockshape[OC_B2ND_MAX_DIM] = {0};
    for (int i = 0; i < ndim; i++) {
        if (shape[i] > INT32_MAX) return -1;
        chunkshape[i] = (int32_t)shape[i];
    }

    /* Pick a block shape: split the FIRST dim only, leave the rest
     * matching the chunk. Aim for ~256 KB blocks to balance throughput
     * vs random-access granularity. */
    int64_t bytes_per_row = (int64_t)itemsize;
    for (int i = 1; i < ndim; i++) bytes_per_row *= shape[i];
    int64_t target_block_bytes = 256 * 1024;
    int64_t rows_per_block = bytes_per_row > 0
        ? target_block_bytes / bytes_per_row : shape[0];
    if (rows_per_block < 1) rows_per_block = 1;
    if (rows_per_block > shape[0]) rows_per_block = shape[0];
    blockshape[0] = (int32_t)rows_per_block;
    for (int i = 1; i < ndim; i++) blockshape[i] = (int32_t)shape[i];

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = itemsize;
    if (level >= 0 && level <= 9) cparams.clevel = level;
    if (compressor != NULL) {
        int code = oc_compcode_from_name(compressor);
        if (code < 0) return -2;  /* unknown compressor */
        cparams.compcode = (uint8_t)code;
    }
    /* Filter pipeline: shuffle / bitshuffle / no shuffle.
     * BLOSC2_CPARAMS_DEFAULTS sets filters[5] = BLOSC_SHUFFLE; override here. */
    for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cparams.filters[i] = 0;
    if (do_bitshuffle == 1) {
        cparams.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_BITSHUFFLE;
    } else if (do_bitshuffle == 0) {
        cparams.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE;
    }
    /* do_bitshuffle == -1: no shuffle */

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    /* dparams.typesize was added in c-blosc2 3.x. The chunk schema carries
     * its own typesize, so we don't need to set it here for decompression
     * to work. Skip it for compat with system blosc2 2.x. */

    /* Use the default (non-contiguous in-memory) storage. b2nd_to_cframe
     * still serializes a non-contiguous schunk into a contiguous bytes
     * buffer — that's what we want for a portable cframe payload. */
    blosc2_storage storage = BLOSC2_STORAGE_DEFAULTS;
    storage.cparams = &cparams;
    storage.dparams = &dparams;

    b2nd_context_t* ctx = b2nd_create_ctx(
        &storage,
        ndim,
        shape,
        chunkshape,
        blockshape,
        dtype_str,         /* may be NULL */
        DTYPE_NUMPY_FORMAT,
        NULL, 0
    );
    if (ctx == NULL) return -3;

    b2nd_array_t* array = NULL;
    int rc = b2nd_from_cbuffer(ctx, &array, data, data_size);
    if (rc != BLOSC2_ERROR_SUCCESS) {
        b2nd_free_ctx(ctx);
        return rc;
    }

    uint8_t* cframe = NULL;
    int64_t cframe_len = 0;
    bool needs_free = false;
    rc = b2nd_to_cframe(array, &cframe, &cframe_len, &needs_free);
    if (rc != BLOSC2_ERROR_SUCCESS) {
        b2nd_free(array);
        b2nd_free_ctx(ctx);
        return rc;
    }

    if (needs_free) {
        /* Caller will free the returned buffer. */
        *out_cframe = cframe;
    } else {
        /* b2nd may return a buffer it owns internally — copy so the
         * caller has a clean owned buffer regardless. */
        uint8_t* copy = (uint8_t*)malloc((size_t)cframe_len);
        if (copy == NULL) {
            b2nd_free(array);
            b2nd_free_ctx(ctx);
            return -4;
        }
        memcpy(copy, cframe, (size_t)cframe_len);
        *out_cframe = copy;
    }
    *out_cframe_len = cframe_len;

    b2nd_free(array);
    b2nd_free_ctx(ctx);
    return 0;
}


int oc_b2nd_inspect(
    const void* cframe,
    int64_t cframe_len,
    int8_t* out_ndim,
    int64_t* out_shape,
    int32_t* out_itemsize,
    char** out_dtype,
    void** handle
) {
    if (cframe == NULL || handle == NULL) return -1;
    oc_ensure_blosc2_init();
    b2nd_array_t* array = NULL;
    int rc = b2nd_from_cframe((uint8_t*)cframe, cframe_len, false, &array);
    if (rc != BLOSC2_ERROR_SUCCESS) {
        *handle = NULL;
        return rc;
    }
    if (out_ndim) *out_ndim = array->ndim;
    if (out_shape) {
        for (int i = 0; i < array->ndim && i < OC_B2ND_MAX_DIM; i++) {
            out_shape[i] = array->shape[i];
        }
    }
    if (out_itemsize) {
        if (array->sc != NULL) {
            blosc2_cparams* cparams = NULL;
            blosc2_schunk_get_cparams(array->sc, &cparams);
            *out_itemsize = cparams ? cparams->typesize : 1;
            if (cparams) free(cparams);
        } else {
            *out_itemsize = 1;
        }
    }
    if (out_dtype) *out_dtype = array->dtype;
    *handle = (void*)array;
    return 0;
}


void oc_b2nd_release(void* handle) {
    if (handle == NULL) return;
    b2nd_free((b2nd_array_t*)handle);
}


int oc_b2nd_decode(
    const void* cframe,
    int64_t cframe_len,
    void* dest_buffer,
    int64_t dest_buffer_size
) {
    if (cframe == NULL || dest_buffer == NULL) return -1;
    oc_ensure_blosc2_init();
    b2nd_array_t* array = NULL;
    int rc = b2nd_from_cframe((uint8_t*)cframe, cframe_len, false, &array);
    if (rc != BLOSC2_ERROR_SUCCESS) return rc;

    rc = b2nd_to_cbuffer(array, dest_buffer, dest_buffer_size);
    b2nd_free(array);
    return rc;
}
