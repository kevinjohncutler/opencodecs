/* JPEG XR decode shim over jxrlib; see jpegxr_shim.c. */
#ifndef OPENCODECS_JPEGXR_SHIM_H
#define OPENCODECS_JPEGXR_SHIM_H

#include <stddef.h>

#define OC_JXR_EINVAL (-1000)
#define OC_JXR_ENOMEM (-1001)

typedef struct oc_jxr oc_jxr;

typedef struct {
    int width;
    int height;
    int channels;       /* color channels, alpha included */
    int bitdepth;       /* jxrlib BITDEPTH_BITS: 1 = BD_8, 2 = BD_16, ... */
    int bits_per_pixel; /* storage bits per pixel, padding included */
    int has_alpha;
    int bgr;            /* channels stored B, G, R */
} oc_jxr_info;

/* Parse the header of an in-memory image; 0 on success, else a negative
 * jxrlib error code. The data must outlive the handle. */
int oc_jxr_open(const void *data, size_t size, oc_jxr **handle,
                oc_jxr_info *info);

/* Decode the whole image into out, rows `stride` bytes apart. */
int oc_jxr_copy(oc_jxr *handle, void *out, size_t stride);

void oc_jxr_close(oc_jxr *handle);

#endif
