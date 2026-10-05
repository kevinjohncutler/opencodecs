"""HEIF decode, and AVIF and HEIF encode, on NVIDIA's video engines.

Selected with ``backend="nvvideocodec"`` on the heif codec (decode and
encode) and the avif codec (encode); never chosen by default. Needs
NVIDIA's PyNvVideoCodec (the Video Codec SDK's own Python package, which
reaches NVDEC and NVENC through the driver) and CuPy, both in
``pip install 'opencodecs[gpu]'``, and an NVIDIA GPU. Nothing here is
imported until the first call that asks for it.

Decode. A HEIF image is HEVC, which NVDEC decodes in hardware: a 4096 x
3072 picture in about 4 ms on an RTX 4090 (8-bit 4:2:0), 11 ms end to
end into a new numpy array, where libheif with libde265 took 66 ms
threaded on one machine and 330 ms on another. The coded pictures
are taken out of the container here (one picture, or every tile of a
grid), decoded on the GPU, converted to RGB on the GPU exactly the way
libheif converts them (same matrix, range, chroma placement and
rounding), tiles placed into one canvas, and copied to the host through
a reused page-locked buffer. Files the hardware would return in another
layout go to libheif instead, exactly as with ``backend=None``: an image
other than the primary, alpha, monochrome, a rotation, mirror or crop
property, a chroma format or depth the GPU cannot decode, or a color
matrix libheif handles specially. :func:`route` says which, and why.

Encode. NVENC writes AV1 (AVIF) and HEVC (HEIF) in 4:2:0, 8 or 10 bits,
lossy only, at a constant quantizer derived from ``level``: hundreds of
times faster than aom and tens of times faster than x265, with larger
files at equal quality. The container is written here, around the
encoder's output, and the stream headers are corrected where NVENC's
would mislead a reader (color signaling; HEVC HRD parameters that
libde265 refuses).

The RGB/YCbCr conversions are CUDA kernels CuPy compiles with NVRTC, so
CuPy needs NVRTC and the CUDA runtime headers (the ``gpu`` extra's
``cuda-toolkit[cudart,nvrtc]``).

Sessions. The first call in a process costs CuPy's and PyNvVideoCodec's
imports, the CUDA context, and the decoder or encoder session (encoder
sessions take about half a second to create). Sessions are kept, one
decoder per chroma format and depth and one encoder, and closed
explicitly at interpreter exit (an NVENC session left to the
interpreter's own teardown once hung a process for minutes), or by
:func:`close`.
"""

from __future__ import annotations

import atexit
import logging
import struct
import threading

import numpy as np

from . import BackendUnavailable, _bitstreams as _bs, _heifbox
from ..core.errors import OpenCodecsError

_log = logging.getLogger("opencodecs")

NAME = "nvvideocodec"
_lock = threading.Lock()
_modules = None            # (PyNvVideoCodec, cupy)
_engines = {}              # CUDA device id -> _Engine


def load():
    """Import PyNvVideoCodec and CuPy and check for a CUDA device, once."""
    global _modules
    if _modules is not None:
        return _modules
    with _lock:
        if _modules is not None:
            return _modules
        try:
            import cupy
        except ImportError as exc:
            raise BackendUnavailable(
                f"backend='{NAME}' needs CuPy; install the build for your "
                f"CUDA version (e.g. pip install cupy-cuda13x), or "
                f"pip install 'opencodecs[gpu]'") from exc
        try:
            import PyNvVideoCodec as nvc
        except ImportError as exc:
            raise BackendUnavailable(
                f"backend='{NAME}' needs NVIDIA's PyNvVideoCodec (pip install "
                f"PyNvVideoCodec), or pip install 'opencodecs[gpu]'") from exc
        except Exception as exc:  # noqa: BLE001 -- e.g. no driver library
            raise BackendUnavailable(
                f"backend='{NAME}': PyNvVideoCodec could not load: {exc}"
            ) from exc
        try:
            count = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:  # noqa: BLE001 -- CUDA reports many ways
            raise BackendUnavailable(
                f"backend='{NAME}' found no usable CUDA device: {exc}") from exc
        if count < 1:
            raise BackendUnavailable(f"backend='{NAME}' found no CUDA device")
        _modules = (nvc, cupy)
        atexit.register(close)
        return _modules


def close():
    """Close every decoder and encoder session and free the staging
    memory. Called at interpreter exit; safe to call any time (the next
    call opens new sessions)."""
    with _lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.close()


# ------------------------------------------------------- GPU kernels

_KERNELS = r"""
__device__ __forceinline__ int clip_f(float fx, int maxi) {
    int x = (int) (fx + 0.5f);
    return x < 0 ? 0 : (x > maxi ? maxi : x);
}
__device__ __forceinline__ int clip_i(int x, int maxi) {
    return x < 0 ? 0 : (x > maxi ? maxi : x);
}

// One output pixel per thread: YCbCr (as NVDEC wrote it) to RGB, the
// arithmetic of libheif 1.21's color conversion. mode 0: matrix 0
// (identity); mode 1: 8-bit 4:2:0 full range, the fixed-point
// Op_YCbCr420_to_RGB24; mode 2: the floating-point Op_YCbCr_to_RGB /
// Op_YCbCr420_to_RRGGBBaa. Chroma is nearest-neighbor, as there.
#define YUV2RGB(NAME, T, O)                                                  \
extern "C" __global__ void NAME(                                             \
    const T* y, long y_pitch, const T* u, const T* v, long c_pitch,          \
    int c_step, int shift, int ssx, int ssy,                                 \
    O* out, long out_pitch, int x0, int y0, int w, int h,                    \
    int mode, float r_cr, float g_cb, float g_cr, float b_cb,                \
    int ir_cr, int ig_cb, int ig_cr, int ib_cb, int full, int bpp)           \
{                                                                            \
    int x = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int yy = blockIdx.y * blockDim.y + threadIdx.y;                          \
    if (x >= w || yy >= h) return;                                           \
    int Y = y[yy * y_pitch + x] >> shift;                                    \
    long ci = (long) (yy >> ssy) * c_pitch + (long) (x >> ssx) * c_step;     \
    int Cb = u[ci] >> shift;                                                 \
    int Cr = v[ci] >> shift;                                                 \
    int maxv = (1 << bpp) - 1;                                               \
    int R, G, B;                                                             \
    if (mode == 0) {                                                         \
        if (full) { R = Cr; G = Y; B = Cb; }                                 \
        else {                                                               \
            float off = (float) (16 << (bpp - 8));                           \
            R = clip_f(((float) Cr - off) * 1.1429f, maxv);                  \
            G = clip_f(((float) Y - off) * 1.1689f, maxv);                   \
            B = clip_f(((float) Cb - off) * 1.1429f, maxv);                  \
        }                                                                    \
    } else if (mode == 1) {                                                  \
        int cb = Cb - 128, cr = Cr - 128;                                    \
        R = clip_i(Y + ((ir_cr * cr + 128) >> 8), 255);                      \
        G = clip_i(Y + ((ig_cb * cb + ig_cr * cr + 128) >> 8), 255);         \
        B = clip_i(Y + ((ib_cb * cb + 128) >> 8), 255);                      \
    } else {                                                                 \
        int half = 1 << (bpp - 1);                                           \
        float yf = (float) Y;                                                \
        float cb = (float) (Cb - half), cr = (float) (Cr - half);           \
        if (!full) {                                                         \
            yf = (yf - (float) (16 << (bpp - 8))) * 1.1689f;                 \
            cb = cb * 1.1429f;                                               \
            cr = cr * 1.1429f;                                               \
        }                                                                    \
        R = clip_f(yf + r_cr * cr, maxv);                                    \
        G = clip_f(yf + g_cb * cb + g_cr * cr, maxv);                        \
        B = clip_f(yf + b_cb * cb, maxv);                                    \
    }                                                                        \
    O* p = out + (long) (y0 + yy) * out_pitch + (long) (x0 + x) * 3;        \
    p[0] = (O) R; p[1] = (O) G; p[2] = (O) B;                                \
}
YUV2RGB(yuv2rgb_u8, unsigned char, unsigned char)
YUV2RGB(yuv2rgb_u16, unsigned short, unsigned short)

// One 2 x 2 block per thread: RGB to 4:2:0 YCbCr for NVENC (NV12, or
// P010 with the sample in the high bits). Luma per pixel, chroma from
// the block's mean RGB, rounded to nearest. An odd width or height is
// padded to even by repeating the last column or row.
#define RGB2YUV(NAME, I, T)                                                  \
extern "C" __global__ void NAME(                                             \
    const I* rgb, long rgb_pitch, int width, int height,                     \
    T* luma, T* chroma, long pitch, int shift,                               \
    float kr, float kg, float kb, int full, int bpp)                         \
{                                                                            \
    int bx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int by = blockIdx.y * blockDim.y + threadIdx.y;                          \
    if (2 * bx >= width || 2 * by >= height) return;                         \
    float maxv = (float) ((1 << bpp) - 1);                                   \
    float ys = full ? maxv : (float) (219 << (bpp - 8));                     \
    float cs = full ? maxv : (float) (224 << (bpp - 8));                     \
    float yo = full ? 0.0f : (float) (16 << (bpp - 8));                      \
    float co = (float) (1 << (bpp - 1));                                     \
    float sr = 0.0f, sg = 0.0f, sb = 0.0f;                                   \
    for (int dy = 0; dy < 2; dy++) {                                         \
        int yy = min(2 * by + dy, height - 1);                               \
        for (int dx = 0; dx < 2; dx++) {                                     \
            int xx = min(2 * bx + dx, width - 1);                            \
            const I* p = rgb + (long) yy * rgb_pitch + (long) xx * 3;        \
            float r = (float) p[0] / maxv, g = (float) p[1] / maxv;          \
            float b = (float) p[2] / maxv;                                   \
            sr += r; sg += g; sb += b;                                       \
            float yv = kr * r + kg * g + kb * b;                             \
            int out = (int) (yv * ys + yo + 0.5f);                           \
            out = out < 0 ? 0 : (out > (int) maxv ? (int) maxv : out);       \
            luma[(long) (2 * by + dy) * pitch + 2 * bx + dx] =               \
                (T) (out << shift);                                          \
        }                                                                    \
    }                                                                        \
    sr *= 0.25f; sg *= 0.25f; sb *= 0.25f;                                   \
    float yv = kr * sr + kg * sg + kb * sb;                                  \
    float cb = (sb - yv) / (2.0f * (1.0f - kb));                             \
    float cr = (sr - yv) / (2.0f * (1.0f - kr));                             \
    int u = (int) floorf(cb * cs + co + 0.5f);                               \
    int v = (int) floorf(cr * cs + co + 0.5f);                               \
    u = u < 0 ? 0 : (u > (int) maxv ? (int) maxv : u);                       \
    v = v < 0 ? 0 : (v > (int) maxv ? (int) maxv : v);                       \
    T* c = chroma + (long) by * pitch + 2 * bx;                              \
    c[0] = (T) (u << shift); c[1] = (T) (v << shift);                        \
}
RGB2YUV(rgb2nv12, unsigned char, unsigned char)
RGB2YUV(rgb2p010, unsigned short, unsigned short)
"""


class _CAI:
    """A bare __cuda_array_interface__ holder, for NVENC's input planes."""

    def __init__(self, shape, strides, typestr, ptr):
        self.__cuda_array_interface__ = {
            "shape": shape, "strides": strides, "data": (int(ptr), False),
            "typestr": typestr, "version": 3}


class _Planes:
    def __init__(self, planes):
        self._planes = planes

    def cuda(self):
        return self._planes


class _Engine:
    """Decoder and encoder sessions, kernels, a CUDA stream and staging
    buffers for one CUDA device. Used by one thread at a time."""

    def __init__(self, device_id: int):
        nvc, cp = load()
        self.device_id = device_id
        self.lock = threading.Lock()
        with cp.cuda.Device(device_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.context = int(cp.cuda.driver.ctxGetCurrent())
            # --fmad=false: libheif's float arithmetic on the CPU has no
            # fused multiply-add, and contracting changes some roundings.
            self.module = cp.RawModule(code=_KERNELS,
                                       options=("--fmad=false",))
        # (chroma, bits) -> [decoder, max width, max height, current size]
        self.decoders = {}
        self.encoder = None      # (settings key, encoder): the last one used
        self.caps = {}
        self._pinned = None      # (pinned memory, nbytes)

    # -- staging --------------------------------------------------------

    def pinned(self, nbytes):
        """A reused page-locked host buffer of at least ``nbytes``. A fresh
        pageable buffer per call copied at about a tenth of the speed."""
        _, cp = load()
        if self._pinned is None or self._pinned[1] < nbytes:
            self._pinned = None
            self._pinned = (cp.cuda.alloc_pinned_memory(nbytes), nbytes)
        return self._pinned[0]

    def close(self):
        with self.lock:
            for entry in self.decoders.values():
                entry[0] = None
            self.decoders.clear()
            self.encoder = None
            self._pinned = None
            try:
                self.stream.synchronize()
            except Exception:  # noqa: BLE001 -- at exit CUDA may be gone
                pass

    # -- decode ---------------------------------------------------------

    def decoder_caps(self, chroma, bits):
        key = (chroma, bits)
        if key not in self.caps:
            nvc, _ = load()
            fmt = {1: "420", 2: "422", 3: "444"}[chroma]
            try:
                caps = nvc.GetDecoderCaps(
                    gpuid=self.device_id, codec=nvc.cudaVideoCodec.HEVC,
                    chromaformat=getattr(nvc.cudaVideoChromaFormat, fmt),
                    bitdepth=bits)
            except Exception as exc:  # noqa: BLE001
                _log.debug("GetDecoderCaps failed: %s", exc)
                caps = {}
            self.caps[key] = dict(caps)
        return self.caps[key]

    def decoder(self, chroma, bits, width, height):
        """The session for this chroma format and depth, set up for
        ``width`` x ``height`` pictures. A session takes smaller or equal
        sizes after setReconfigParams (without it, PyNvVideoCodec kept
        reporting the previous size); a larger one needs a new session."""
        nvc, _ = load()
        wanted = (width, height)
        entry = self.decoders.get((chroma, bits))
        if entry is not None and entry[1] >= width and entry[2] >= height:
            if entry[3] != wanted:
                entry[0].setReconfigParams(width, height)
                entry[3] = wanted
            return entry[0]
        if entry is not None:
            entry[0] = None                    # close the smaller session
            width, height = max(width, entry[1]), max(height, entry[2])
        dec = nvc.CreateDecoder(
            gpuid=self.device_id, codec=nvc.cudaVideoCodec.HEVC,
            cudacontext=self.context, cudastream=self.stream.ptr,
            usedevicememory=True, maxwidth=width, maxheight=height,
            latency=nvc.DisplayDecodeLatencyType.LOW)
        # A new session sizes itself from the first picture's SPS.
        self.decoders[(chroma, bits)] = [dec, width, height, wanted]
        return dec


def _engine() -> _Engine:
    _, cp = load()
    device_id = int(cp.cuda.runtime.getDevice())
    engine = _engines.get(device_id)
    if engine is None:
        with _lock:
            engine = _engines.get(device_id)
            if engine is None:
                engine = _engines[device_id] = _Engine(device_id)
    return engine


# ------------------------------------------------- libheif's color math

_F = np.float32

# Kr, Kb of the matrices libheif knows (nclx.cc get_Kr_Kb).
_KR_KB = {1: (0.2126, 0.0722), 4: (0.30, 0.11), 5: (0.299, 0.114),
          6: (0.299, 0.114), 7: (0.212, 0.087), 9: (0.2627, 0.0593),
          10: (0.2627, 0.0593)}

# Color primaries (gx, gy, bx, by, rx, ry, wx, wy), for matrices 12/13.
_PRIMARIES = {
    1: (0.300, 0.600, 0.150, 0.060, 0.640, 0.330, 0.3127, 0.3290),
    4: (0.21, 0.71, 0.14, 0.08, 0.67, 0.33, 0.310, 0.316),
    5: (0.29, 0.60, 0.15, 0.06, 0.64, 0.33, 0.3127, 0.3290),
    6: (0.310, 0.595, 0.155, 0.070, 0.630, 0.340, 0.3127, 0.3290),
    7: (0.310, 0.595, 0.155, 0.070, 0.630, 0.340, 0.3127, 0.3290),
    8: (0.243, 0.692, 0.145, 0.049, 0.681, 0.319, 0.310, 0.316),
    9: (0.170, 0.797, 0.131, 0.046, 0.708, 0.292, 0.3127, 0.3290),
    10: (0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.333333, 0.33333),
    11: (0.265, 0.690, 0.150, 0.060, 0.680, 0.320, 0.314, 0.351),
    12: (0.265, 0.690, 0.150, 0.060, 0.680, 0.320, 0.3127, 0.3290),
    22: (0.295, 0.605, 0.155, 0.077, 0.630, 0.340, 0.3127, 0.3290),
}

# Matrices whose conversion is not the Kr/Kb formula or identity: YCgCo
# (8, 16) and the ones libheif does not convert at all (11, 14). Files
# using them are left to libheif.
_SPECIAL_MATRICES = (8, 11, 14, 16)


def _kr_kb(matrix, primaries):
    """libheif's get_Kr_Kb in float32 arithmetic, or (0, 0)."""
    if matrix in (12, 13):
        p = _PRIMARIES.get(primaries)
        if p is None:
            return _F(0), _F(0)
        gx, gy, bx, by, rx, ry, wx, wy = (_F(v) for v in p)
        one = _F(1)
        zr, zg, zb, zw = one - (rx + ry), one - (gx + gy), one - (bx + by), one - (wx + wy)
        denom = wy * (rx * (gy * zb - by * zg) + gx * (by * zr - ry * zb)
                      + bx * (ry * zg - gy * zr))
        if denom == 0:
            return _F(0), _F(0)
        kr = (ry * (wx * (gy * zb - by * zg) + wy * (bx * zg - gx * zb)
                    + zw * (gx * by - bx * gy))) / denom
        kb = (by * (wx * (ry * zg - gy * zr) + wy * (gx * zr - rx * zg)
                    + zw * (rx * gy - gx * ry))) / denom
        return _F(kr), _F(kb)
    kr, kb = _KR_KB.get(matrix, (0.0, 0.0))
    return _F(kr), _F(kb)


def _ycbcr_to_rgb_coefficients(matrix, primaries):
    """(r_cr, g_cb, g_cr, b_cb) as float32, as libheif computes them."""
    kr, kb = _kr_kb(matrix, primaries)
    if kr == 0 and kb == 0:
        return _F(1.402), _F(-0.344136), _F(-0.714136), _F(1.772)
    one, two = _F(1), _F(2)
    return (_F(two * (-kr + one)),
            _F(two * kb * (-kb + one) / (kb + kr - one)),
            _F(two * kr * (-kr + one) / (kb + kr - one)),
            _F(two * (-kb + one)))


# ---------------------------------------------------------- decode plan

class _Plan:
    """What a HEIF's primary image needs for the GPU path."""

    def __init__(self):
        self.width = self.height = 0
        self.tile_w = self.tile_h = 0
        self.rows = self.cols = 1
        self.tiles = []          # Annex-B stream per tile, row-major
        self.chroma = 1
        self.bits = 8
        self.nclx = None         # (primaries, transfer, matrix, full)


def _plan(data, *, index=None, photometric=None):
    """(plan, "") if the GPU should decode ``data``, else (None, reason).

    Reads only the container's metadata and the SPS."""
    if index is not None:
        return None, "index= selects an image other than the primary"
    if photometric is not None:
        return None, "photometric= is a libheif decode option"
    try:
        items = _heifbox.parse(memoryview(data))
    except (struct.error, IndexError, ValueError):
        items = None
    if items is None or items.primary is None:
        return None, "could not read the HEIF item structure"
    primary = items.primary
    kind = items.types.get(primary)
    plan = _Plan()
    try:
        if kind == b"grid":
            tiles = items.references(b"dimg", primary)
            if not tiles:
                return None, "grid image without tiles"
            plan.rows, plan.cols, plan.width, plan.height = \
                _heifbox.grid_layout(items, primary)
            if len(tiles) != plan.rows * plan.cols:
                return None, "grid with a tile count unlike its layout"
        elif kind == b"hvc1":
            tiles = [primary]
            size = items.size(primary)
            if size is None:
                return None, "image without a size (ispe)"
            plan.width, plan.height = size
        else:
            return None, f"primary image is {kind!r}, not HEVC"
        reason = _heifbox.transform_reason(items, primary)
        if reason:
            return None, reason
        if _heifbox.has_alpha(items, primary):
            return None, "the image has an alpha plane"
        config = None
        for tile in tiles:
            if items.types.get(tile) != b"hvc1":
                return None, "grid tiles are not HEVC"
            if tile != primary and _heifbox.transform_reason(items, tile):
                return None, "a grid tile has a transform property"
            size = items.size(tile)
            if size is None:
                return None, "grid tile without a size"
            hvcc = items.properties(tile).get(b"hvcC")
            if hvcc is None:
                return None, "no HEVC configuration"
            this = _heifbox.hvcc(items.data, *hvcc)
            if config is None:
                config, (plan.tile_w, plan.tile_h) = this, size
            elif (size != (plan.tile_w, plan.tile_h)
                  or (this["chroma"], this["luma_bits"])
                  != (config["chroma"], config["luma_bits"])):
                return None, "grid tiles of different sizes or formats"
            params = b"".join(b"\0\0\0\1" + n for n in this["nals"])
            plan.tiles.append(params + _heifbox.length_prefixed_to_annexb(
                items.item_data(tile), this["length_size"]))
        if (plan.cols * plan.tile_w < plan.width
                or plan.rows * plan.tile_h < plan.height):
            return None, "grid tiles do not cover the image"
        plan.chroma, plan.bits = config["chroma"], config["luma_bits"]
        if plan.chroma == 0:
            return None, "monochrome"
        if config["chroma_bits"] != plan.bits:
            return None, "luma and chroma of different depths"
        # The color libheif converts with: the primary item's colr nclx
        # unless it is absent or "undefined" (2, 2, 2, full range); for a
        # grid, then the first tile's; then the first tile's SPS VUI.
        undefined = (2, 2, 2, True)
        for item in (primary, tiles[0]):
            nclx = items.nclx(item)
            if nclx is not None and nclx != undefined:
                plan.nclx = nclx
                break
        if plan.nclx is None:
            sps = next((n for n in config["nals"]
                        if _bs.nal_type(n) == 33), None)
            if sps is None:
                return None, "no SPS in the HEVC configuration"
            plan.nclx = _bs.hevc_sps(sps)["vui"]
    except (struct.error, IndexError, ValueError) as exc:
        return None, f"could not read the coded image ({exc})"
    if plan.nclx[2] in _SPECIAL_MATRICES:
        return None, (f"color matrix {plan.nclx[2]}, which libheif converts "
                      f"with its own special case")
    return plan, ""


def route(data, *, index=None, photometric=None) -> tuple[str, str]:
    """``("nvvideocodec", "")`` if NVDEC should decode ``data``, else
    ``("native", reason)``. Reads only the container's metadata (and asks
    the GPU what it can decode, which loads the backend)."""
    plan, reason = _plan(data, index=index, photometric=photometric)
    if plan is None:
        return "native", reason
    reason = _hardware_reason(_engine(), plan)
    return ("native", reason) if reason else (NAME, "")


def _hardware_reason(engine, plan) -> str:
    caps = engine.decoder_caps(plan.chroma, plan.bits)
    fmt = {1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}[plan.chroma]
    if not caps.get("supported", 0):
        return f"this GPU's NVDEC does not decode {plan.bits}-bit {fmt} HEVC"
    w, h = plan.tile_w, plan.tile_h
    min_w, min_h = caps.get("width_min", 0), caps.get("height_min", 0)
    if w < min_w or h < min_h:
        return (f"a {w} x {h} coded picture is smaller than this GPU's "
                f"NVDEC takes ({min_w} x {min_h})")
    max_w, max_h = caps.get("width_max", 0), caps.get("height_max", 0)
    blocks = ((w + 15) // 16) * ((h + 15) // 16)
    if ((max_w and w > max_w) or (max_h and h > max_h)
            or blocks > caps.get("mb_num_max", blocks)):
        return (f"a {w} x {h} coded picture is larger than this GPU's "
                f"NVDEC takes ({max_w} x {max_h})")
    return ""


# --------------------------------------------------------------- decode

def _convert_args(plan):
    """The kernel's mode and coefficient arguments for this image."""
    cp_, _, mc, full = plan.nclx
    bits, chroma = plan.bits, plan.chroma
    r_cr, g_cb, g_cr, b_cb = _ycbcr_to_rgb_coefficients(mc, cp_)
    replaced = 6 if mc == 2 else mc
    if mc == 0:
        mode = 0
    elif bits == 8 and chroma == 1 and full and replaced not in (0, 8, 11, 14):
        mode = 1
    else:
        mode = 2
    # std::lround rounds half away from zero; numpy rounds half to even.
    ints = [int(np.floor(abs(float(_F(256) * c)) + 0.5)) * (1 if c >= 0 else -1)
            for c in (r_cr, g_cb, g_cr, b_cb)]
    return (np.int32(mode), r_cr, g_cb, g_cr, b_cb,
            *(np.int32(i) for i in ints), np.int32(1 if full else 0),
            np.int32(bits))


def _planes(frame, elem):
    """(Y, Cb, Cr pointers, chroma step, row pitch in samples, stream) of
    a decoded frame.

    The planes are read from the luma plane's interface alone, because
    PyNvVideoCodec 2.2 describes 16-bit frames wrongly: strides in
    samples rather than bytes (a pitch smaller than a packed row in bytes
    can only be in samples), and for 16-bit 4:4:4 chroma plane addresses
    computed at one byte a sample, inside the luma plane. Its frames are
    one allocation, luma then chroma (NV12, P016) or luma, Cb, Cr (4:4:4)
    at the luma pitch, which is what NVDEC writes.
    """
    planes = frame.cuda()
    cai = planes[0].__cuda_array_interface__
    shape, pitch = cai["shape"], int(cai["strides"][0])
    if pitch < int(shape[1]) * elem:
        pitch *= elem
    luma = int(cai["data"][0])
    size = pitch * int(shape[0])
    if len(planes) == 2:                    # NV12 / P016 / NV16 / P216
        cb, step = luma + size, 2
        cr = cb + elem
    else:                                   # planar 4:4:4
        cb, cr, step = luma + size, luma + 2 * size, 1
    return luma, cb, cr, step, pitch // elem, cai.get("stream")


def _stream(cp, handle):
    """A CuPy stream for an array-interface stream value: 1 is the legacy
    default stream, 2 the per-thread default stream (what PyNvVideoCodec
    2.2 reports), anything else a stream handle."""
    if handle == 1:
        return cp.cuda.Stream.null
    if handle == 2:
        return cp.cuda.Stream.ptds
    return cp.cuda.ExternalStream(handle)


def _decode_on_gpu(engine, plan, canvas):
    """Decode every tile of ``plan`` into the device array ``canvas``."""
    nvc, cp = load()
    dec = engine.decoder(plan.chroma, plan.bits, plan.tile_w, plan.tile_h)
    kernel = engine.module.get_function(
        "yuv2rgb_u8" if plan.bits == 8 else "yuv2rgb_u16")
    elem = 1 if plan.bits == 8 else 2
    args = _convert_args(plan)
    shift = np.int32(0 if plan.bits == 8 else 16 - plan.bits)
    ssx = np.int32(1 if plan.chroma in (1, 2) else 0)
    ssy = np.int32(1 if plan.chroma == 1 else 0)
    out_pitch = np.int64(plan.width * 3)
    streams = {}
    done = 0

    def place(frame):
        nonlocal done
        tile = int(frame.getPTS())
        if not 0 <= tile < len(plan.tiles):
            raise OpenCodecsError("heif decode: NVDEC returned an unexpected "
                                  "frame")
        y_ptr, u_ptr, v_ptr, c_step, y_pitch, handle = _planes(frame, elem)
        c_pitch = y_pitch
        # The frame's pixels are ready on the stream it names, and the
        # decoder writes its next frame into the same memory on that
        # stream, so the conversion runs there, between the two.
        stream = engine.stream
        if handle is not None:
            stream = streams.get(int(handle))
            if stream is None:
                stream = streams[int(handle)] = _stream(cp, int(handle))
                stream.wait_event(engine.stream.record())     # canvas ready
        row, col = divmod(tile, plan.cols)
        x0, y0 = col * plan.tile_w, row * plan.tile_h
        w = min(plan.tile_w, plan.width - x0)
        h = min(plan.tile_h, plan.height - y0)
        if w > 0 and h > 0:
            block = (32, 8)
            grid = ((w + 31) // 32, (h + 7) // 8)
            kernel(grid, block, (
                np.uint64(y_ptr), np.int64(y_pitch), np.uint64(u_ptr),
                np.uint64(v_ptr), np.int64(c_pitch), np.int32(c_step), shift,
                ssx, ssy, canvas, out_pitch, np.int32(x0), np.int32(y0),
                np.int32(w), np.int32(h), *args), stream=stream)
        done += 1

    packet = nvc.PacketData()
    for i, stream in enumerate(plan.tiles):
        buf = np.frombuffer(stream, np.uint8)
        packet.bsl_data = buf.ctypes.data
        packet.bsl = buf.size
        packet.pts = i
        for frame in dec.Decode(packet):
            place(frame)
    for frame in dec.Decode(nvc.PacketData()):        # end of stream: flush
        place(frame)
    for stream in streams.values():
        engine.stream.wait_event(stream.record())
    if done != len(plan.tiles):
        raise OpenCodecsError(
            f"heif decode: NVDEC returned {done} of {len(plan.tiles)} "
            f"pictures; decode it with backend=None")


def _is_pinned(cp, arr) -> bool:
    try:
        attrs = cp.cuda.runtime.pointerGetAttributes(arr.ctypes.data)
    except Exception:  # noqa: BLE001 -- pageable memory raises on some CUDA
        return False
    return getattr(attrs, "type", 0) == 1       # cudaMemoryTypeHost


def decode(data, plan, *, out=None):
    """Decode a planned HEIF on the GPU into ``out`` or a new array."""
    _, cp = load()
    engine = _engine()
    dtype = np.dtype(np.uint8 if plan.bits == 8 else np.uint16)
    shape = (plan.height, plan.width, 3)
    nbytes = plan.height * plan.width * 3 * dtype.itemsize
    with engine.lock, cp.cuda.Device(engine.device_id):
        if out is not None and isinstance(out, cp.ndarray):
            _check_out(out, shape, dtype)
            canvas = out
        else:
            canvas = cp.empty(shape, dtype)
        # The decoder writes on engine.stream; CuPy allocated canvas on
        # the current stream, so order the two.
        engine.stream.wait_event(cp.cuda.get_current_stream().record())
        _decode_on_gpu(engine, plan, canvas)
        if canvas is out:
            cp.cuda.get_current_stream().wait_event(engine.stream.record())
            return out
        if out is not None:
            from ..core.buffers import array_output
            out = array_output(out)
            _check_out(out, shape, dtype)
            if _is_pinned(cp, out):
                canvas.get(stream=engine.stream, out=out)
                engine.stream.synchronize()
                return out
            result = out
        else:
            result = np.empty(shape, dtype)
        staging = np.frombuffer(engine.pinned(nbytes), dtype,
                                nbytes // dtype.itemsize).reshape(shape)
        canvas.get(stream=engine.stream, out=staging)
        engine.stream.synchronize()
        _host_copy(result, staging)
        return result


def _host_copy(dst, src):
    """``dst[...] = src`` in row bands on a few threads. Copying from the
    staging buffer into a new array is bound by page faults and memory
    bandwidth: 5.2 ms for 36 MB on one thread, 2.0 ms on eight."""
    from ..core.parallel import auto_threads, run_batched
    workers = auto_threads(None, work=src.nbytes, per_thread=4 << 20,
                           max_threads=8)
    rows = src.shape[0]
    if workers <= 1 or rows < workers:
        np.copyto(dst, src)
        return
    edges = [rows * i // workers for i in range(workers + 1)]
    run_batched(lambda i: np.copyto(dst[edges[i]:edges[i + 1]],
                                    src[edges[i]:edges[i + 1]]),
                list(range(workers)), workers, name="copy")


def _check_out(out, shape, dtype):
    if tuple(out.shape) != shape or out.dtype != dtype:
        raise ValueError(f"heif decode: out= is {tuple(out.shape)} "
                         f"{out.dtype}; the image is {shape} {dtype.name}")
    if not out.flags.c_contiguous:
        raise ValueError("heif decode: out= must be C-contiguous")


def decode_heif(data, native_decode, *, out=None, index=None,
                photometric=None, **native_kw):
    """The heif codec's ``backend="nvvideocodec"`` decode: NVDEC where
    :func:`route` says it can, libheif (``native_decode``) otherwise."""
    engine = _engine()
    plan, reason = _plan(data, index=index, photometric=photometric)
    if plan is not None:
        reason = _hardware_reason(engine, plan)
        if not reason:
            return decode(data, plan, out=out)
    _log.debug("heif decode, backend='%s': using libheif (%s)", NAME, reason)
    _, cp = load()
    if out is not None and isinstance(out, cp.ndarray):
        pixels = native_decode(data, index=index, photometric=photometric,
                               **native_kw)
        if tuple(out.shape) != pixels.shape or out.dtype != pixels.dtype:
            raise ValueError(f"heif decode: out= is {tuple(out.shape)} "
                             f"{out.dtype}; the image is {pixels.shape} "
                             f"{pixels.dtype}")
        out.set(pixels)
        return out
    if out is not None:
        from ..core.buffers import array_output
        out = array_output(out)
    return native_decode(data, out=out, index=index, photometric=photometric,
                         **native_kw)


# --------------------------------------------------------------- encode

# libaom's quantizer (0..63) to AV1 qindex (0..255) table; libavif maps
# its quality to that quantizer, so the same level lands on the same
# qindex here.
_QINDEX = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
           68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
           128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176,
           180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228,
           232, 236, 240, 244, 249, 255)

_TUNING = "high_quality"

# What the pixels are converted with and the container says: BT.601
# full range with sRGB primaries and transfer, libheif's and libavif's
# own default for RGB input.
_NCLX = (1, 13, 6, True)


def _quality(level, name):
    if level is None:
        raise ValueError(
            f"{name} encode: backend='{NAME}' is lossy only and the codec's "
            f"default is lossless; pass level= (0 to 99) to accept loss")
    level = float(level)
    if not 0 <= level < 100:
        raise ValueError(f"{name} encode: backend='{NAME}' takes level= from "
                         f"0 to 99 (lossy); {level:g} is outside it")
    return level


def _encoder_input(cp, data, name):
    """The input as a C-contiguous (H, W, 3) array on the device."""
    arr = data if isinstance(data, cp.ndarray) else np.asarray(data)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype not in (np.uint8,
                                                               np.uint16):
        raise ValueError(
            f"{name} encode: backend='{NAME}' takes (H, W, 3) RGB, uint8 or "
            f"uint16, not {arr.shape} {arr.dtype}")
    return arr


def _depth(arr, bit_depth, name):
    if arr.dtype == np.uint8:
        if bit_depth not in (None, 8):
            raise ValueError(f"{name} encode: uint8 input is coded at 8 bits")
        return 8
    if bit_depth not in (None, 10):
        raise ValueError(f"{name} encode: backend='{NAME}' codes uint16 at "
                         f"10 bits only")
    if int(arr.max()) > 1023:
        raise ValueError(f"{name} encode: backend='{NAME}' codes uint16 at "
                         f"10 bits; this data needs more")
    return 10


def _encode_frame(engine, codec, arr, bits, qp, preset):
    """Convert ``arr`` to 4:2:0 on the GPU and encode it as one intra
    frame; the encoder's raw output."""
    nvc, cp = load()
    height, width = arr.shape[:2]
    caps_key = ("enc", codec)
    if caps_key not in engine.caps:
        try:
            engine.caps[caps_key] = dict(nvc.GetEncoderCaps(
                gpuid=engine.device_id, codec=codec))
        except Exception as exc:  # noqa: BLE001
            _log.debug("GetEncoderCaps failed: %s", exc)
            engine.caps[caps_key] = {}
    caps = engine.caps[caps_key]
    if not caps.get("width_max"):
        # An empty report: no such encoder (AV1 needs Ada or newer).
        raise BackendUnavailable(
            f"this GPU's NVENC does not encode {codec.upper()}")
    if bits > 8 and not caps.get("support_10bit_encode", 0):
        raise BackendUnavailable(f"this GPU's NVENC does not encode 10-bit "
                                 f"{codec.upper()}")
    if not (caps.get("width_min", 0) <= width <= caps.get("width_max", 1 << 30)
            and caps.get("height_min", 0) <= height
            <= caps.get("height_max", 1 << 30)):
        raise ValueError(
            f"{codec} encode: NVENC takes {caps.get('width_min')} x "
            f"{caps.get('height_min')} to {caps.get('width_max')} x "
            f"{caps.get('height_max')} pixels, not {width} x {height}; "
            f"use backend=None")
    # NVENC wants even dimensions for 4:2:0; odd ones are padded by
    # repeating the last row/column and cropped again on decode (clap).
    cw, ch = width + (width & 1), height + (height & 1)
    fmt = "NV12" if bits == 8 else "P010"
    key = (codec, cw, ch, fmt, qp, preset)
    elem = 1 if bits == 8 else 2
    with cp.cuda.Device(engine.device_id):
        if isinstance(arr, cp.ndarray):
            src = cp.ascontiguousarray(arr)
        else:
            host = np.ascontiguousarray(arr)
            staging = np.frombuffer(engine.pinned(host.nbytes), host.dtype,
                                    host.size).reshape(host.shape)
            _host_copy(staging, host)
            src = cp.empty(host.shape, host.dtype)
            engine.stream.wait_event(cp.cuda.get_current_stream().record())
            src.set(staging, stream=engine.stream)
        yuv = cp.empty((ch * 3 // 2, cw), np.uint8 if bits == 8 else np.uint16)
        luma, chroma = yuv[:ch], yuv[ch:]
        kr, kb = 0.299, 0.114
        kernel = engine.module.get_function(
            "rgb2nv12" if bits == 8 else "rgb2p010")
        grid = (((cw // 2) + 15) // 16, ((ch // 2) + 15) // 16)
        kernel(grid, (16, 16), (
            src, np.int64(width * 3), np.int32(width), np.int32(height),
            luma, chroma, np.int64(cw), np.int32(16 - bits if bits > 8 else 0),
            np.float32(kr), np.float32(1 - kr - kb), np.float32(kb),
            np.int32(1), np.int32(bits)), stream=engine.stream)
        if engine.encoder is None or engine.encoder[0] != key:
            engine.encoder = None          # close the old session first
            kw = dict(codec=codec, preset=preset, tuning_info=_TUNING,
                      rc="constqp", constqp=qp, gop=1, bf=0, idrperiod=1,
                      cudacontext=engine.context, cudastream=engine.stream.ptr)
            if codec == "hevc":
                kw["colorspace"] = "bt601"
            engine.encoder = (key, nvc.CreateEncoder(cw, ch, fmt, False, **kw))
        planes = _Planes([
            # PyNvVideoCodec checks these strides against its own
            # convention (its samples' P010 frames give (pitch, 2, 1)
            # for both planes), which is not the array-interface meaning.
            _CAI((ch, cw, 1), (cw * elem, elem, 1), f"|u{elem}", luma.data.ptr),
            _CAI((ch // 2, cw // 2, 2), (cw * elem, 2, 1),
                 f"|u{elem}", chroma.data.ptr)])
        engine.stream.synchronize()
        enc = engine.encoder[1]
        packets = list(enc.Encode(planes)) + list(enc.EndEncode())
    return b"".join(bytes(p["data"]) if isinstance(p, dict) else bytes(p)
                    for p in packets), (cw, ch)


def _clap(width, height, coded_w, coded_h) -> bytes:
    """A clap box keeping the top-left ``width`` x ``height``."""
    return struct.pack(">I4s", 40, b"clap") + struct.pack(
        ">IIIIiIiI", width, 1, height, 1,
        width - coded_w, 2, height - coded_h, 2)


def encode_avif(data, *, level=None, lossless=None, speed=None, bit_depth=None,
                yuv_format=None, iccprofile=None, **unsupported) -> bytes:
    """AVIF from (H, W, 3) RGB with NVENC's AV1 encoder (4:2:0, lossy)."""
    nvc, cp = load()
    bad = sorted(k for k, v in unsupported.items() if v not in (None, False))
    if bad:
        raise ValueError(f"avif encode: backend='{NAME}' does not support "
                         f"{', '.join(bad)}; use backend=None for them")
    if lossless:
        raise ValueError(f"avif encode: backend='{NAME}' is lossy only; use "
                         f"backend=None for lossless")
    if yuv_format not in (None, "420", 420, "yuv420"):
        raise ValueError(f"avif encode: backend='{NAME}' writes 4:2:0 only, "
                         f"not yuv_format={yuv_format!r}")
    level = _quality(level, "avif")
    arr = _encoder_input(cp, data, "avif")
    bits = _depth(arr, bit_depth, "avif")
    if arr.shape[0] & 1 or arr.shape[1] & 1:
        # PyNvVideoCodec takes odd-sized 4:2:0 input only with the chroma
        # plane truncated, and padding plus a clap box would show the
        # padding in readers that ignore clap, as libavif-based ones do.
        raise ValueError(
            f"avif encode: backend='{NAME}' needs an even width and height "
            f"for 4:2:0, not {arr.shape[1]} x {arr.shape[0]}; use "
            f"backend=None")
    qp = _QINDEX[int(((100 - level) * 63 + 50) // 100)]
    preset = _preset(speed)
    engine = _engine()
    with engine.lock:
        raw, (cw, ch) = _encode_frame(engine, "av1", arr, bits, qp, preset)
    obus = _bs.split_obus(raw)
    seq = next((o for o in obus if o[0] == _bs.OBU_SEQUENCE_HEADER), None)
    if seq is None:
        raise OpenCodecsError("avif encode: NVENC wrote no sequence header")
    header = _bs.av1_set_color(seq[1], seq[2], _NCLX)
    info = _bs.av1_sequence_header(header[_payload_at(header):])
    payload = header + b"".join(
        o[1] for o in obus
        if o[0] not in (_bs.OBU_SEQUENCE_HEADER, _bs.OBU_TEMPORAL_DELIMITER,
                        _bs.OBU_PADDING))
    height, width = arr.shape[:2]
    compatible = [b"avif", b"mif1", b"miaf"]
    if info["profile"] == 0 and info["level"] <= 13:
        compatible.append(b"MA1B")
    return _write(b"avif", compatible, b"av01", _bs.av1c_record(info, header),
                  width, height, cw, ch, bits, payload, iccprofile)


def _payload_at(obu: bytes) -> int:
    ext = (obu[0] >> 2) & 1
    at = 1 + ext
    while obu[at] & 0x80:
        at += 1
    return at + 1


def _write(major, compatible, item_type, config, width, height, cw, ch, bits,
           payload, icc):
    clap = None if (cw, ch) == (width, height) else _clap(width, height, cw, ch)
    return _heifbox.write_single_item(
        brands=(major, compatible), item_type=item_type, config=config,
        width=cw, height=ch, bits=bits, channels=3, payload=payload,
        nclx=_NCLX, icc=icc, clap=clap)


def _preset(speed):
    """NVENC preset for libavif's speed (0 slowest .. 10 fastest)."""
    if speed is None:
        return "P7"
    speed = int(speed)
    if not 0 <= speed <= 10:
        raise ValueError(f"encode: speed={speed} is not in 0..10")
    return f"P{7 - round(speed * 6 / 10)}"


def encode(data, *, level=None, lossless=None, bit_depth=None,
                iccprofile=None, color=None) -> bytes:
    """HEIC from (H, W, 3) RGB with NVENC's HEVC encoder (4:2:0, lossy).

    ``level`` (0 to 99) maps to a constant QP the way x265's quality
    scale does in libheif: ``qp = 51 - level * 51 / 100``, rounded."""
    nvc, cp = load()
    if lossless:
        raise ValueError(f"heif encode: backend='{NAME}' is lossy only; use "
                         f"backend=None for lossless")
    if color is not None:
        raise ValueError(f"heif encode: backend='{NAME}' does not take "
                         f"color=; it writes sRGB")
    level = _quality(level, "heif")
    arr = _encoder_input(cp, data, "heif")
    bits = _depth(arr, bit_depth, "heif")
    qp = int(round(51 - level * 51 / 100))
    engine = _engine()
    with engine.lock:
        raw, (cw, ch) = _encode_frame(engine, "hevc", arr, bits, qp, "P7")
    nals = _bs.split_annexb(raw)
    sets = {}
    slices = []
    for nal in nals:
        kind = _bs.nal_type(nal)
        if kind in (32, 33, 34):
            sets.setdefault(kind, nal)
        elif kind != 35:                     # drop access unit delimiters
            slices.append(struct.pack(">I", len(nal)) + nal)
    if len(sets) != 3:
        raise OpenCodecsError("heif encode: NVENC wrote no parameter sets")
    # NVENC's SPS signals no color (so a reader going by the stream alone
    # takes it as limited range) and carries HRD buffering parameters
    # whose values libde265, the HEVC decoder of most libheif builds,
    # rejects above 8.9 megapixels. Write the color the pixels were
    # converted with and drop the HRD, which a still image has no use for.
    try:
        sets[33] = _bs.hevc_sps_rewrite(sets[33], signal=_NCLX, drop_hrd=True)
    except ValueError:                       # no VUI to carry the color
        sets[33] = _bs.hevc_sps_rewrite(sets[33], drop_hrd=True)
    height, width = arr.shape[:2]
    major = b"heic" if bits == 8 else b"heix"
    return _write(major, [b"mif1", major], b"hvc1",
                  _bs.hvcc_record(sets[32], sets[33], sets[34]),
                  width, height, cw, ch, bits, b"".join(slices), iccprofile)
