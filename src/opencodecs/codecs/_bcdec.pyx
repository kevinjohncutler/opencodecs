# opencodecs/codecs/_bcdec.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Block-compressed texture decoder (BC1-7 / DXT / ATI / BPTC).

BC formats are GPU-side block-compressed texture formats used by
game assets (.dds files), Direct3D / OpenGL / Vulkan, and some image
editors. Each format compresses 4x4 pixel blocks at a fixed rate:

  * BC1 (DXT1)  — RGB or RGB+1-bit alpha, 8 bytes/block (4 bpp)
  * BC2 (DXT3)  — RGB + 4-bit explicit alpha, 16 bytes/block (8 bpp)
  * BC3 (DXT5)  — RGB + interpolated alpha, 16 bytes/block (8 bpp)
  * BC4 (ATI1N) — single-channel (R), 8 bytes/block (4 bpp)
  * BC5 (ATI2N) — two-channel (RG), 16 bytes/block (8 bpp)
  * BC6H        — HDR RGB, half/float, 16 bytes/block (8 bpp)
  * BC7         — RGBA, 16 bytes/block (8 bpp)

This module is decode-only — these formats target GPU pipelines and
encoding is typically done offline with vendor tools (NVIDIA Texture
Tools, AMD Compressonator, Microsoft's bc6h.exe). The decoder is
:command:`bcdec`'s single-header implementation (MIT) vendored at
``opencodecs/3rdparty/bcdec/``.

Usage
=====

::

    from opencodecs.codecs._bcdec import decode_bc1, decode_bc7

    rgba = decode_bc1(raw_bc1_bytes, width=512, height=512)
    rgba = decode_bc7(raw_bc7_bytes, width=512, height=512)

All decoders return numpy arrays:

  * BC1/2/3/7 → ``(H, W, 4)`` uint8 RGBA
  * BC4       → ``(H, W)`` uint8 (or int8 with signed=True)
  * BC5       → ``(H, W, 2)`` uint8 (or int8 with signed=True)
  * BC6H      → ``(H, W, 3)`` float32 (or float16 with format='half')
"""

from libc.stdint cimport uint8_t, uint16_t

import numpy as np
cimport numpy as cnp

from bcdec cimport (
    BCDEC_BC1_BLOCK_SIZE, BCDEC_BC2_BLOCK_SIZE, BCDEC_BC3_BLOCK_SIZE,
    BCDEC_BC4_BLOCK_SIZE, BCDEC_BC5_BLOCK_SIZE,
    BCDEC_BC6H_BLOCK_SIZE, BCDEC_BC7_BLOCK_SIZE,
    bcdec_bc1, bcdec_bc2, bcdec_bc3,
    bcdec_bc4, bcdec_bc5,
    bcdec_bc6h_half, bcdec_bc6h_float,
    bcdec_bc7,
)

cnp.import_array()


class BcdecError(RuntimeError):
    """Raised on malformed BC input."""


# Each BC format's compressed block is 8 or 16 bytes. ``width`` /
# ``height`` MUST be multiples of 4 — BC textures are block-compressed
# at fixed 4x4 tile granularity and there's no per-block padding flag.
def _check_block_geometry(width: int, height: int, fmt: str) -> None:
    if width % 4 or height % 4:
        raise BcdecError(
            f"BC decode: width and height must be multiples of 4 "
            f"(got {width}x{height} for {fmt})"
        )


cdef _decode_rgba_blocks(
    const uint8_t* src, Py_ssize_t src_len, int width, int height,
    int block_bytes, int fmt_id,
):
    """Decompress an array of BCn blocks into (H, W, 4) uint8 RGBA.
    fmt_id picks the block-decode function: 1=BC1, 2=BC2, 3=BC3, 7=BC7."""
    cdef int n_blocks_x = width // 4
    cdef int n_blocks_y = height // 4
    cdef Py_ssize_t expected = <Py_ssize_t> n_blocks_x * n_blocks_y * block_bytes
    if src_len < expected:
        raise BcdecError(
            f"BC decode: input too short — got {src_len} bytes, "
            f"need {expected} for {width}x{height} (fmt id {fmt_id})"
        )

    cdef cnp.npy_intp shape[3]
    shape[0] = height
    shape[1] = width
    shape[2] = 4
    cdef cnp.ndarray out = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
    cdef uint8_t* dst = <uint8_t*> cnp.PyArray_DATA(out)
    cdef int pitch = width * 4   # bytes per row in destination
    cdef int by, bx
    cdef const uint8_t* block_p
    cdef uint8_t* tile_p

    with nogil:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block_p = src + (by * n_blocks_x + bx) * block_bytes
                tile_p = dst + (by * 4) * pitch + (bx * 4) * 4
                if fmt_id == 1:
                    bcdec_bc1(block_p, tile_p, pitch)
                elif fmt_id == 2:
                    bcdec_bc2(block_p, tile_p, pitch)
                elif fmt_id == 3:
                    bcdec_bc3(block_p, tile_p, pitch)
                else:  # fmt_id == 7
                    bcdec_bc7(block_p, tile_p, pitch)
    return out


def decode_bc1(data, *, width: int, height: int) -> np.ndarray:
    """Decode BC1 (DXT1) blocks → (H, W, 4) uint8 RGBA."""
    cdef const uint8_t[::1] buf
    _check_block_geometry(width, height, "BC1")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return _decode_rgba_blocks(&buf[0], buf.shape[0], width, height,
                                BCDEC_BC1_BLOCK_SIZE, 1)


def decode_bc2(data, *, width: int, height: int) -> np.ndarray:
    """Decode BC2 (DXT3) blocks → (H, W, 4) uint8 RGBA."""
    cdef const uint8_t[::1] buf
    _check_block_geometry(width, height, "BC2")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return _decode_rgba_blocks(&buf[0], buf.shape[0], width, height,
                                BCDEC_BC2_BLOCK_SIZE, 2)


def decode_bc3(data, *, width: int, height: int) -> np.ndarray:
    """Decode BC3 (DXT5) blocks → (H, W, 4) uint8 RGBA."""
    cdef const uint8_t[::1] buf
    _check_block_geometry(width, height, "BC3")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return _decode_rgba_blocks(&buf[0], buf.shape[0], width, height,
                                BCDEC_BC3_BLOCK_SIZE, 3)


def decode_bc7(data, *, width: int, height: int) -> np.ndarray:
    """Decode BC7 blocks → (H, W, 4) uint8 RGBA."""
    cdef const uint8_t[::1] buf
    _check_block_geometry(width, height, "BC7")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return _decode_rgba_blocks(&buf[0], buf.shape[0], width, height,
                                BCDEC_BC7_BLOCK_SIZE, 7)


def decode_bc4(data, *, width: int, height: int,
                is_signed: bool = False) -> np.ndarray:
    """Decode BC4 (ATI1N) blocks → (H, W) uint8 (or int8 if signed)."""
    cdef const uint8_t[::1] buf
    cdef int is_signed_c
    cdef int n_blocks_x
    cdef int n_blocks_y
    cdef Py_ssize_t expected
    _check_block_geometry(width, height, "BC4")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    is_signed_c = 1 if is_signed else 0
    n_blocks_x = width // 4
    n_blocks_y = height // 4
    expected = <Py_ssize_t> n_blocks_x * n_blocks_y * BCDEC_BC4_BLOCK_SIZE
    if buf.shape[0] < expected:
        raise BcdecError(
            f"BC4 decode: input too short ({buf.shape[0]} < {expected})"
        )
    cdef cnp.npy_intp shape2[2]
    shape2[0] = height
    shape2[1] = width
    cdef cnp.ndarray out = cnp.PyArray_EMPTY(
        2, shape2,
        cnp.NPY_INT8 if is_signed else cnp.NPY_UINT8, 0,
    )
    cdef uint8_t* dst = <uint8_t*> cnp.PyArray_DATA(out)
    cdef int pitch = width
    cdef int by, bx
    cdef const uint8_t* block_p
    cdef uint8_t* tile_p
    with nogil:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block_p = &buf[(by * n_blocks_x + bx) * BCDEC_BC4_BLOCK_SIZE]
                tile_p = dst + (by * 4) * pitch + (bx * 4)
                bcdec_bc4(block_p, tile_p, pitch, is_signed_c)
    return out


def decode_bc5(data, *, width: int, height: int,
                is_signed: bool = False) -> np.ndarray:
    """Decode BC5 (ATI2N) blocks → (H, W, 2) uint8 (or int8 if signed)."""
    cdef const uint8_t[::1] buf
    cdef int is_signed_c
    cdef int n_blocks_x
    cdef int n_blocks_y
    cdef Py_ssize_t expected
    _check_block_geometry(width, height, "BC5")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    is_signed_c = 1 if is_signed else 0
    n_blocks_x = width // 4
    n_blocks_y = height // 4
    expected = <Py_ssize_t> n_blocks_x * n_blocks_y * BCDEC_BC5_BLOCK_SIZE
    if buf.shape[0] < expected:
        raise BcdecError(
            f"BC5 decode: input too short ({buf.shape[0]} < {expected})"
        )
    cdef cnp.npy_intp shape3[3]
    shape3[0] = height
    shape3[1] = width
    shape3[2] = 2
    cdef cnp.ndarray out = cnp.PyArray_EMPTY(
        3, shape3,
        cnp.NPY_INT8 if is_signed else cnp.NPY_UINT8, 0,
    )
    cdef uint8_t* dst = <uint8_t*> cnp.PyArray_DATA(out)
    cdef int pitch = width * 2
    cdef int by, bx
    cdef const uint8_t* block_p
    cdef uint8_t* tile_p
    with nogil:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block_p = &buf[(by * n_blocks_x + bx) * BCDEC_BC5_BLOCK_SIZE]
                tile_p = dst + (by * 4) * pitch + (bx * 4) * 2
                bcdec_bc5(block_p, tile_p, pitch, is_signed_c)
    return out


def decode_bc6h(data, *, width: int, height: int,
                 is_signed: bool = False, format: str = "float") -> np.ndarray:
    """Decode BC6H (HDR) blocks → (H, W, 3).

    ``format='float'`` (default) returns float32 RGB; ``'half'`` returns
    float16 RGB. ``signed=True`` enables the signed BC6H variant.
    """
    cdef const uint8_t[::1] buf
    cdef int is_signed_c
    cdef int n_blocks_x
    cdef int n_blocks_y
    cdef Py_ssize_t expected
    _check_block_geometry(width, height, "BC6H")
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    is_signed_c = 1 if is_signed else 0
    n_blocks_x = width // 4
    n_blocks_y = height // 4
    expected = <Py_ssize_t> n_blocks_x * n_blocks_y * BCDEC_BC6H_BLOCK_SIZE
    if buf.shape[0] < expected:
        raise BcdecError(
            f"BC6H decode: input too short ({buf.shape[0]} < {expected})"
        )

    cdef cnp.npy_intp shape3[3]
    shape3[0] = height
    shape3[1] = width
    shape3[2] = 3

    cdef cnp.ndarray out
    cdef int by, bx
    cdef const uint8_t* block_p
    cdef int pitch
    # ``destinationPitch`` for bcdec_bc6h_half / bcdec_bc6h_float is the
    # row stride in *destination elements* (half or float), not bytes:
    # the C function does ``decompressed += destinationPitch`` against a
    # typed pointer, so the compiler scales by sizeof(element). Passing
    # bytes-per-row overshoots the destination by 2x (half) / 4x (float)
    # and writes past the numpy buffer on the final block row.
    if format == "half":
        out = cnp.PyArray_EMPTY(3, shape3, cnp.NPY_FLOAT16, 0)
        pitch = width * 3
        with nogil:
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    block_p = &buf[(by * n_blocks_x + bx) * BCDEC_BC6H_BLOCK_SIZE]
                    bcdec_bc6h_half(
                        block_p,
                        (<uint16_t*> cnp.PyArray_DATA(out)) + (by * 4) * pitch + (bx * 4 * 3),
                        pitch, is_signed_c,
                    )
    elif format == "float":
        out = cnp.PyArray_EMPTY(3, shape3, cnp.NPY_FLOAT32, 0)
        pitch = width * 3
        with nogil:
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    block_p = &buf[(by * n_blocks_x + bx) * BCDEC_BC6H_BLOCK_SIZE]
                    bcdec_bc6h_float(
                        block_p,
                        (<float*> cnp.PyArray_DATA(out)) + (by * 4) * pitch + (bx * 4 * 3),
                        pitch, is_signed_c,
                    )
    else:
        raise BcdecError(
            f"BC6H decode: format must be 'float' or 'half', got {format!r}"
        )
    return out
