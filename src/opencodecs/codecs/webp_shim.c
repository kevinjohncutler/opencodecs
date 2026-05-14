/* webp_shim.c — advanced-API WebP encode with thread_level exposed. */
#include "webp_shim.h"
#include <stdlib.h>
#include <string.h>
#include <webp/encode.h>


int oc_webp_encode(
    const uint8_t *rgb_buf, int width, int height, int stride,
    int has_alpha, int lossless, float quality,
    int thread_level, int method,
    uint8_t **out_ptr, size_t *out_size)
{
    *out_ptr = NULL;
    *out_size = 0;

    WebPConfig config;
    if (!WebPConfigInit(&config)) {
        return 1;  /* version mismatch */
    }
    if (lossless) {
        if (!WebPConfigLosslessPreset(&config, 6)) {
            return 1;
        }
    } else {
        config.quality = quality;
    }
    if (method >= 0 && method <= 6) {
        config.method = method;
    }
    config.thread_level = thread_level ? 1 : 0;

    if (!WebPValidateConfig(&config)) {
        return 1;
    }

    WebPPicture picture;
    if (!WebPPictureInit(&picture)) {
        return 1;
    }
    picture.width = width;
    picture.height = height;
    /* use_argb=1 is required for both lossless AND threaded lossy encode. */
    picture.use_argb = lossless ? 1 : 0;

    int rc;
    if (has_alpha) {
        rc = WebPPictureImportRGBA(&picture, rgb_buf, stride);
    } else {
        rc = WebPPictureImportRGB(&picture, rgb_buf, stride);
    }
    if (!rc) {
        WebPPictureFree(&picture);
        return picture.error_code ? picture.error_code : 1;
    }

    WebPMemoryWriter writer;
    WebPMemoryWriterInit(&writer);
    picture.writer = WebPMemoryWrite;
    picture.custom_ptr = &writer;

    int ok = WebPEncode(&config, &picture);
    int err = picture.error_code;
    WebPPictureFree(&picture);

    if (!ok) {
        WebPMemoryWriterClear(&writer);
        return err ? err : 1;
    }

    /* WebPMemoryWriter owns the buffer via malloc/realloc; transfer
     * ownership to the caller (we don't call WebPMemoryWriterClear). */
    *out_ptr = writer.mem;
    *out_size = writer.size;
    return 0;
}


void oc_webp_free(void *ptr)
{
    /* WebPMemoryWriter allocates with malloc/realloc; plain free() owns it. */
    free(ptr);
}
