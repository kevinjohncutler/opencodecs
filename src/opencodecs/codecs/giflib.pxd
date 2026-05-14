# cython: language_level = 3
"""Cython declarations for giflib (libgif) — the canonical C GIF library.

We bind only the surface we need: in-memory I/O via callbacks, slurp/
spew the whole file in one shot, plus enough of the structure types to
walk frames and color maps.
"""

from libc.stdint cimport uint8_t


cdef extern from "gif_lib.h" nogil:
    ctypedef int GifWord
    ctypedef unsigned char GifByteType
    ctypedef unsigned char GifPixelType

    int GIF_OK
    int GIF_ERROR
    int DISPOSE_DO_NOT
    int DISPOSE_BACKGROUND
    int DISPOSE_PREVIOUS

    ctypedef struct GifColorType:
        GifByteType Red
        GifByteType Green
        GifByteType Blue

    ctypedef struct ColorMapObject:
        int ColorCount
        int BitsPerPixel
        bint SortFlag
        GifColorType* Colors

    ctypedef struct GifImageDesc:
        GifWord Left
        GifWord Top
        GifWord Width
        GifWord Height
        bint Interlace
        ColorMapObject* ColorMap

    ctypedef struct ExtensionBlock:
        int ByteCount
        GifByteType* Bytes
        int Function

    ctypedef struct SavedImage:
        GifImageDesc ImageDesc
        GifByteType* RasterBits
        int ExtensionBlockCount
        ExtensionBlock* ExtensionBlocks

    ctypedef struct GifFileType:
        GifWord SWidth
        GifWord SHeight
        GifWord SColorResolution
        GifWord SBackGroundColor
        GifByteType AspectByte
        ColorMapObject* SColorMap
        int ImageCount
        GifImageDesc Image
        SavedImage* SavedImages
        int ExtensionBlockCount
        ExtensionBlock* ExtensionBlocks
        int Error
        void* UserData

    ctypedef int (*InputFunc)(GifFileType*, GifByteType*, int) nogil
    ctypedef int (*OutputFunc)(GifFileType*, const GifByteType*, int) nogil

    const char* GifErrorString(int ErrorCode)

    GifFileType* DGifOpen(void* userPtr, InputFunc readFunc, int* Error)
    int DGifSlurp(GifFileType* GifFile)
    int DGifCloseFile(GifFileType* GifFile, int* ErrorCode)

    GifFileType* EGifOpen(void* userPtr, OutputFunc writeFunc, int* Error)
    int EGifCloseFile(GifFileType* GifFile, int* ErrorCode)
    void EGifSetGifVersion(GifFileType* GifFile, bint gif89)
    int EGifPutScreenDesc(
        GifFileType* GifFile,
        int GifWidth, int GifHeight,
        int GifColorRes, int GifBackGround,
        const ColorMapObject* GifColorMap,
    )
    int EGifPutImageDesc(
        GifFileType* GifFile,
        int GifLeft, int GifTop,
        int GifWidth, int GifHeight,
        bint GifInterlace,
        const ColorMapObject* GifColorMap,
    )
    int EGifPutLine(
        GifFileType* GifFile,
        GifPixelType* GifLine,
        int GifLineLen,
    )
    ColorMapObject* GifMakeMapObject(int ColorCount, const GifColorType* ColorMap)
    void GifFreeMapObject(ColorMapObject* Object)
