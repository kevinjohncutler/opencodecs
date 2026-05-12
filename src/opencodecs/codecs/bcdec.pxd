# Cython declarations for bcdec (vendored at 3rdparty/bcdec/bcdec.h).
#
# We compile bcdec.h with BCDEC_IMPLEMENTATION + BCDEC_STATIC defined
# in _bcdec.pyx so the implementation lives inline in our extension —
# no separate C file or library.

from libc.stdint cimport uint8_t


cdef extern from "bcdec.h" nogil:

    # Block sizes (bytes per 4x4 block).
    cdef int BCDEC_BC1_BLOCK_SIZE
    cdef int BCDEC_BC2_BLOCK_SIZE
    cdef int BCDEC_BC3_BLOCK_SIZE
    cdef int BCDEC_BC4_BLOCK_SIZE
    cdef int BCDEC_BC5_BLOCK_SIZE
    cdef int BCDEC_BC6H_BLOCK_SIZE
    cdef int BCDEC_BC7_BLOCK_SIZE

    # Decompress one 4x4 block. Output strides (destinationPitch) are
    # in BYTES, not pixels — e.g. for BC1 (RGBA8 output) pitch is
    # 4 * width_in_pixels.
    void bcdec_bc1(const void* block, void* out, int pitch)
    void bcdec_bc2(const void* block, void* out, int pitch)
    void bcdec_bc3(const void* block, void* out, int pitch)
    void bcdec_bc4(const void* block, void* out, int pitch, int is_signed)
    void bcdec_bc5(const void* block, void* out, int pitch, int is_signed)
    void bcdec_bc6h_half(const void* block, void* out, int pitch, int is_signed)
    void bcdec_bc6h_float(const void* block, void* out, int pitch, int is_signed)
    void bcdec_bc7(const void* block, void* out, int pitch)
