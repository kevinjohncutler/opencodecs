/* Thin JPEG XR (ITU-T T.832) decode shim over jxrlib's JXRGlue API.
 *
 * jxrlib's decoder is a struct of function pointers created from a
 * WMPStream; its in-memory stream constructor lives in the codec library
 * but not in the installed headers, so it is declared here. The shim keeps
 * all of that out of Cython: open parses the header and reports the pixel
 * layout, copy decodes into a caller buffer, close releases everything.
 */
#include <stdlib.h>
#include <string.h>

#include "jpegxr_shim.h"

#include <JXRGlue.h>

ERR CreateWS_Memory(struct WMPStream **ppWS, void *pv, size_t cb);

struct oc_jxr {
    PKImageDecode *decoder;
};

int oc_jxr_open(const void *data, size_t size, oc_jxr **handle,
                oc_jxr_info *info)
{
    struct WMPStream *stream = NULL;
    PKImageDecode *decoder = NULL;
    PKPixelFormatGUID format;
    PKPixelInfo pixel;
    I32 width = 0, height = 0;
    ERR err;
    oc_jxr *h;

    *handle = NULL;
    memset(info, 0, sizeof(*info));
    if (data == NULL || size == 0)
        return OC_JXR_EINVAL;
    err = CreateWS_Memory(&stream, (void *)data, size);
    if (err < 0)
        return (int)err;
    err = PKImageDecode_Create_WMP(&decoder);
    if (err < 0) {
        stream->Close(&stream);
        return (int)err;
    }
    err = decoder->Initialize(decoder, stream);
    if (err < 0) {
        decoder->Release(&decoder);
        stream->Close(&stream);
        return (int)err;
    }
    /* From here the decoder owns the stream and closes it on Release. */
    decoder->fStreamOwner = 1;

    err = decoder->GetPixelFormat(decoder, &format);
    if (err >= 0)
        err = decoder->GetSize(decoder, &width, &height);
    if (err >= 0) {
        memset(&pixel, 0, sizeof(pixel));
        pixel.pGUIDPixFmt = &format;
        err = PixelFormatLookup(&pixel, LOOKUP_FORWARD);
    }
    if (err < 0) {
        decoder->Release(&decoder);
        return (int)err;
    }
    /* As jxrlib's own decoder does: decode the image plus its alpha
     * plane when there is one; left unset, the alpha samples come back
     * as whatever the buffer held. */
    decoder->WMP.wmiSCP.uAlphaMode = decoder->WMP.bHasAlpha ? 2 : 0;
    if (width <= 0 || height <= 0) {
        decoder->Release(&decoder);
        return OC_JXR_EINVAL;
    }
    h = (oc_jxr *)malloc(sizeof(*h));
    if (h == NULL) {
        decoder->Release(&decoder);
        return OC_JXR_ENOMEM;
    }
    h->decoder = decoder;
    info->width = (int)width;
    info->height = (int)height;
    info->channels = (int)pixel.cChannel;
    info->bitdepth = (int)pixel.bdBitDepth;
    info->bits_per_pixel = (int)pixel.cbitUnit;
    info->has_alpha = (pixel.grBit & PK_pixfmtHasAlpha) != 0;
    info->bgr = (pixel.grBit & PK_pixfmtBGR) != 0;
    *handle = h;
    return 0;
}

int oc_jxr_copy(oc_jxr *handle, void *out, size_t stride)
{
    PKRect rect;
    PKImageDecode *decoder;
    I32 width = 0, height = 0;
    ERR err;

    if (handle == NULL || out == NULL)
        return OC_JXR_EINVAL;
    decoder = handle->decoder;
    err = decoder->GetSize(decoder, &width, &height);
    if (err < 0)
        return (int)err;
    rect.X = 0;
    rect.Y = 0;
    rect.Width = width;
    rect.Height = height;
    err = decoder->Copy(decoder, &rect, (U8 *)out, (U32)stride);
    return err < 0 ? (int)err : 0;
}

void oc_jxr_close(oc_jxr *handle)
{
    if (handle == NULL)
        return;
    if (handle->decoder != NULL)
        handle->decoder->Release(&handle->decoder);
    free(handle);
}
