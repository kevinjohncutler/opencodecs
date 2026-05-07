# Minimal Cython declarations for libzstd (Facebook's Zstandard).
# Only the basic compress/decompress + introspection we need.

cdef extern from 'zstd.h' nogil:
    int ZSTD_VERSION_MAJOR
    int ZSTD_VERSION_MINOR
    int ZSTD_VERSION_RELEASE
    int ZSTD_CLEVEL_DEFAULT

    size_t ZSTD_compress(
        void* dst, size_t dstCapacity,
        const void* src, size_t srcSize,
        int compressionLevel,
    )

    size_t ZSTD_decompress(
        void* dst, size_t dstCapacity,
        const void* src, size_t compressedSize,
    )

    size_t ZSTD_compressBound(size_t srcSize)
    unsigned long long ZSTD_getFrameContentSize(const void* src, size_t srcSize)

    int ZSTD_minCLevel()
    int ZSTD_maxCLevel()

    unsigned ZSTD_isError(size_t code)
    const char* ZSTD_getErrorName(size_t code)

    int ZSTD_CONTENTSIZE_UNKNOWN
    int ZSTD_CONTENTSIZE_ERROR
