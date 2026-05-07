# Minimal Cython declarations for liblz4 — LZ4 frame format.

cdef extern from 'lz4frame.h' nogil:
    int LZ4F_VERSION
    ctypedef size_t LZ4F_errorCode_t

    unsigned LZ4F_isError(LZ4F_errorCode_t code)
    const char* LZ4F_getErrorName(LZ4F_errorCode_t code)

    ctypedef enum LZ4F_blockSizeID_t:
        LZ4F_default
        LZ4F_max64KB
        LZ4F_max256KB
        LZ4F_max1MB
        LZ4F_max4MB

    ctypedef enum LZ4F_blockMode_t:
        LZ4F_blockLinked
        LZ4F_blockIndependent

    ctypedef enum LZ4F_contentChecksum_t:
        LZ4F_noContentChecksum
        LZ4F_contentChecksumEnabled

    ctypedef enum LZ4F_blockChecksum_t:
        LZ4F_noBlockChecksum
        LZ4F_blockChecksumEnabled

    ctypedef enum LZ4F_frameType_t:
        LZ4F_frame
        LZ4F_skippableFrame

    ctypedef struct LZ4F_frameInfo_t:
        LZ4F_blockSizeID_t blockSizeID
        LZ4F_blockMode_t blockMode
        LZ4F_contentChecksum_t contentChecksumFlag
        LZ4F_frameType_t frameType
        unsigned long long contentSize
        unsigned dictID
        LZ4F_blockChecksum_t blockChecksumFlag

    ctypedef struct LZ4F_preferences_t:
        LZ4F_frameInfo_t frameInfo
        int compressionLevel
        unsigned autoFlush
        unsigned favorDecSpeed
        unsigned[3] reserved

    size_t LZ4F_compressFrameBound(
        size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr,
    )

    size_t LZ4F_compressFrame(
        void* dstBuffer, size_t dstCapacity,
        const void* srcBuffer, size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr,
    )

    ctypedef struct LZ4F_dctx:
        pass

    LZ4F_errorCode_t LZ4F_createDecompressionContext(
        LZ4F_dctx** dctxPtr, unsigned version,
    )

    LZ4F_errorCode_t LZ4F_freeDecompressionContext(LZ4F_dctx* dctx)

    ctypedef struct LZ4F_decompressOptions_t:
        unsigned stableDst
        unsigned skipChecksums
        unsigned[2] reserved

    size_t LZ4F_decompress(
        LZ4F_dctx* dctx,
        void* dstBuffer, size_t* dstSizePtr,
        const void* srcBuffer, size_t* srcSizePtr,
        const LZ4F_decompressOptions_t* dOptPtr,
    )

    size_t LZ4F_getFrameInfo(
        LZ4F_dctx* dctx,
        LZ4F_frameInfo_t* frameInfoPtr,
        const void* srcBuffer, size_t* srcSizePtr,
    )
