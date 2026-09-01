# cython: language_level = 3
"""Cython header for the LERC C API (Esri Limited Error Raster Compression)."""

# Transcribed from upstream 'Lerc_c_api.h'. imagecodecs declares the same C
# API in its own lerc.pxd; the two overlap because the header fixes the
# names and signatures, not because either was copied from the other.


cdef extern from "Lerc_c_api.h" nogil:

    ctypedef unsigned int lerc_status

    lerc_status lerc_computeCompressedSize(
        const void* pData,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned int* numBytes,
    )

    lerc_status lerc_encode(
        const void* pData,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned char* pOutBuffer,
        unsigned int outBufferSize,
        unsigned int* nBytesWritten,
    )

    lerc_status lerc_getBlobInfo(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        unsigned int* infoArray,
        double* dataRangeArray,
        int infoArraySize,
        int dataRangeArraySize,
    )

    lerc_status lerc_decode(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        int nMasks,
        unsigned char* pValidBytes,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        unsigned int dataType,
        void* pData,
    )
