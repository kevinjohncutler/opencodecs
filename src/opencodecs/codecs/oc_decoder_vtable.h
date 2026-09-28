/* A decompressor one extension module can drive from another, without
 * the GIL. The providing module (_deflate, _zstd) hands out a pointer to
 * one of these through a PyCapsule named OC_DECODER_VTABLE_CAPSULE; the
 * consumer (_tiff) calls it in a nogil loop over many segments, so a whole
 * batch of tiles decodes, un-predicts and lands in its output under one
 * GIL release. Linking the consumer against every compression library
 * instead would have tied its build to theirs.
 */
#ifndef OC_DECODER_VTABLE_H
#define OC_DECODER_VTABLE_H

#include <stddef.h>
#include <stdint.h>

#define OC_DECODER_VTABLE_CAPSULE "opencodecs.decoder_vtable"

typedef struct {
    /* A context for one thread at a time; NULL on allocation failure. */
    void* (*create)(void);
    void (*destroy)(void* ctx);
    /* Decode one complete stream from src into dst (capacity cap). Returns
     * the number of bytes written, or -1 on any error, including output
     * that does not fit in cap. */
    ptrdiff_t (*decode)(void* ctx, const uint8_t* src, size_t n,
                        uint8_t* dst, size_t cap);
} oc_decoder_vtable;

#endif
