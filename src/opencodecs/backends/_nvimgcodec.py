"""JPEG, JPEG 2000 and HTJ2K on an NVIDIA GPU through nvImageCodec.

Selected with ``backend="nvimgcodec"`` on the jpeg, jpeg2k and htj2k
codecs; never chosen by default. Needs CuPy and nvImageCodec
(``pip install opencodecs[gpu]`` pulls the CUDA 13 builds) and a CUDA
device. Nothing here is imported until the first call that asks for it.

Engines. nvJPEG2000 decodes and encodes JPEG 2000 and HTJ2K on the GPU
(nvImageCodec's GPU_ONLY backend). JPEG goes through nvJPEG: the
hardware JPEG engine where the GPU has one (A100, H100 and similar), and
otherwise the hybrid decoder, which does Huffman decoding on the CPU and
the rest on the GPU; consumer GPUs have no JPEG engine, and nvImageCodec's
GPU_ONLY JPEG decoder returns nothing there. nvImageCodec's own CPU
decoders are never enabled, so a stream the GPU cannot take raises rather
than quietly running on the CPU.

Output. Decoded pixels are copied to the host with CuPy (into a new
array, or into ``out=``), not with nvImageCodec's ``Image.cpu()``, which
measured four times slower for a 48 MB image. The result has the dtype
and shape the CPU decoder returns; for the lossless codecs the pixels are
the same, bit for bit. ``out=`` may also be a CuPy array, in which case
the pixels stay on the GPU.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from . import BackendUnavailable
from ..core.errors import OpenCodecsError

_lock = threading.Lock()
_modules = None          # (cupy, nvimgcodec) once loaded
_engines = {}            # CUDA device id -> _Engine


def load():
    """Import CuPy and nvImageCodec and check for a CUDA device, once."""
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
                "backend='nvimgcodec' needs CuPy; install the build for "
                "your CUDA version (e.g. pip install cupy-cuda13x), or "
                "pip install 'opencodecs[gpu]'") from exc
        try:
            from nvidia import nvimgcodec
        except ImportError as exc:
            raise BackendUnavailable(
                "backend='nvimgcodec' needs nvImageCodec with its nvJPEG and "
                "nvJPEG2000 extensions (e.g. pip install "
                "nvidia-nvimgcodec-cu13 nvidia-nvjpeg nvidia-nvjpeg2k-cu13), "
                "or pip install 'opencodecs[gpu]'") from exc
        try:
            count = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:  # noqa: BLE001 -- CUDA reports many ways
            raise BackendUnavailable(
                f"backend='nvimgcodec' found no usable CUDA device: {exc}"
            ) from exc
        if count < 1:
            raise BackendUnavailable(
                "backend='nvimgcodec' found no CUDA device")
        _modules = (cupy, nvimgcodec)
        return _modules


class _Engine:
    """The decoders and encoders of one CUDA device, made on first use.

    Each one is used by one thread at a time (nvImageCodec does not
    promise more); the copy back to the host happens outside that lock.
    """

    def __init__(self, device_id: int):
        self.device_id = device_id
        self._made = {}
        self._lock = threading.Lock()

    def _get(self, key, make):
        pair = self._made.get(key)
        if pair is None:
            with self._lock:
                pair = self._made.get(key)
                if pair is None:
                    pair = self._made[key] = (make(), threading.Lock())
        return pair[0]

    def _guard(self, key):
        return self._made[key][1]

    def decode(self, codec: str, stream, params):
        decoder = self.decoder(codec)
        with self._guard(("decoder", "jpeg" if codec == "jpeg" else "j2k")):
            return decoder.decode(stream, params=params)

    def encode(self, codec: str, image, fmt: str, params):
        encoder = self.encoder(codec)
        with self._guard(("encoder", codec)):
            return encoder.encode(image, fmt, params=params)

    def decoder(self, codec: str):
        cp, n = load()
        if codec == "jpeg":
            kinds = (n.BackendKind.HW_GPU_ONLY, n.BackendKind.HYBRID_CPU_GPU)
        else:
            kinds = (n.BackendKind.GPU_ONLY,)
        key = ("decoder", "jpeg" if codec == "jpeg" else "j2k")
        return self._get(key, lambda: n.Decoder(
            device_id=self.device_id,
            backends=[n.Backend(k) for k in kinds]))

    def encoder(self, codec: str):
        cp, n = load()
        if codec == "jpeg":
            kinds = (n.BackendKind.HYBRID_CPU_GPU, n.BackendKind.GPU_ONLY)
        else:
            kinds = (n.BackendKind.GPU_ONLY,)
        # One encoder per codec, HTJ2K apart from JPEG 2000: with
        # nvImageCodec 0.9 a color JPEG 2000 encode failed (nvJPEG2000
        # "internal error") on an encoder that had just written HTJ2K.
        key = ("encoder", codec)
        return self._get(key, lambda: n.Encoder(
            device_id=self.device_id,
            backends=[n.Backend(k) for k in kinds]))


def _engine() -> _Engine:
    cp, _ = load()
    device_id = int(cp.cuda.runtime.getDevice())
    engine = _engines.get(device_id)
    if engine is None:
        with _lock:
            engine = _engines.setdefault(device_id, _Engine(device_id))
    return engine


def pinned_empty(shape, dtype) -> np.ndarray:
    """An empty numpy array backed by page-locked host memory."""
    cp, _ = load()
    dtype = np.dtype(dtype)
    shape = tuple(int(s) for s in np.atleast_1d(shape))
    count = int(np.prod(shape, dtype=np.int64))
    memory = cp.cuda.alloc_pinned_memory(max(count * dtype.itemsize, 1))
    # The array's base holds the pinned allocation alive.
    return np.frombuffer(memory, dtype, count).reshape(shape)


# --------------------------------------------------------------- decode

def _expected(codec: str, data: bytes, cs):
    """(shape, dtype) the CPU decoder would return, or raise if the GPU
    would not return the same thing."""
    if codec == "jpeg2k":
        from ..codecs._jpeg2k import decode_info
        info = decode_info(data)
        return tuple(info["shape"]), np.dtype(info["dtype"])
    if codec == "htj2k":
        from ..codecs._openjph import decode_info
        info = decode_info(data)
        if info["dtype"] is None or info["nlt_type"]:
            raise OpenCodecsError(
                "htj2k decode: backend='nvimgcodec' cannot decode this "
                "codestream the way the CPU decoder does (mixed component "
                "precision, or float samples marked by an NLT segment); "
                "decode it with backend=None")
        h, w, c = info["height"], info["width"], info["components"]
        return ((h, w) if c == 1 else (h, w, c)), np.dtype(info["dtype"])
    # JPEG: 8-bit gray or three-component color only.
    if cs.precision not in (0, 8) or cs.num_channels not in (1, 3):
        raise OpenCodecsError(
            f"jpeg decode: backend='nvimgcodec' decodes 8-bit gray or "
            f"color JPEG; this stream has {cs.num_channels} components at "
            f"{cs.precision} bits. Decode it with backend=None")
    h, w = cs.height, cs.width
    return ((h, w) if cs.num_channels == 1 else (h, w, 3)), np.dtype(np.uint8)


def _decode_params(n, codec: str, cs):
    if codec == "jpeg":
        if cs.num_channels == 1:
            color, sample = n.ColorSpec.GRAY, n.SampleFormat.I_Y
        else:
            color, sample = n.ColorSpec.SRGB, n.SampleFormat.I_RGB
        return n.DecodeParams(apply_exif_orientation=False, color_spec=color,
                              sample_format=sample)
    # JPEG 2000: the stored samples, at their own depth, in their own
    # color space, as OpenJPEG and OpenJPH hand them back.
    return n.DecodeParams(apply_exif_orientation=False, allow_any_depth=True,
                          color_spec=n.ColorSpec.UNCHANGED,
                          sample_format=n.SampleFormat.I_UNCHANGED)


def decode(codec: str, data: bytes, *, out=None, planar=None):
    """Decode ``data`` on the GPU; see the module docstring."""
    cp, n = load()
    engine = _engine()
    try:
        stream = n.CodeStream(data)
    except Exception as exc:  # noqa: BLE001 -- nvImageCodec raises RuntimeError
        raise OpenCodecsError(
            f"{codec} decode: nvImageCodec cannot parse this stream: {exc}"
        ) from exc
    shape, dtype = _expected(codec, data, stream)
    image = engine.decode(codec, stream, _decode_params(n, codec, stream))
    if image is None:
        raise OpenCodecsError(
            f"{codec} decode: nvImageCodec could not decode this stream on "
            f"the GPU (see its log above); decode it with backend=None")
    pixels = cp.asarray(image)
    if pixels.dtype != dtype or pixels.size != int(np.prod(shape)):
        raise OpenCodecsError(
            f"{codec} decode: the GPU decoded {pixels.shape} {pixels.dtype} "
            f"where the CPU decoder returns {shape} {dtype.name}; decode "
            f"it with backend=None")
    pixels = pixels.reshape(shape)
    if planar and len(shape) == 3:
        pixels = cp.ascontiguousarray(cp.moveaxis(pixels, -1, 0))
        shape = pixels.shape
    return _to_output(codec, pixels, shape, dtype, out, cp)


def _to_output(codec, pixels, shape, dtype, out, cp):
    if out is None:
        return pixels.get()
    if isinstance(out, cp.ndarray):
        _check_out(codec, out, shape, dtype)
        cp.copyto(out, pixels)
        return out
    from ..core.buffers import array_output
    out = array_output(out)
    _check_out(codec, out, shape, dtype)
    pixels.get(out=out)
    return out


def _check_out(codec, out, shape, dtype):
    if tuple(out.shape) != tuple(shape):
        raise ValueError(
            f"{codec} decode: out= shape {tuple(out.shape)} does not match "
            f"the image {tuple(shape)}")
    if out.dtype != dtype:
        raise ValueError(
            f"{codec} decode: out= dtype {out.dtype} does not match the "
            f"image's {dtype.name}")
    if not out.flags.c_contiguous:
        raise ValueError(f"{codec} decode: out= must be C-contiguous")


# --------------------------------------------------------------- encode

_SUBSAMPLING = {"444": "CSS_444", "422": "CSS_422", "420": "CSS_420",
                "440": "CSS_440", "411": "CSS_411", "410": "CSS_410",
                (1, 1): "CSS_444", (2, 1): "CSS_422", (2, 2): "CSS_420",
                (1, 2): "CSS_440", (4, 1): "CSS_411"}

_PROG_ORDER = ("LRCP", "RLCP", "RPCL", "PCRL", "CPRL")


def _device_image(cp, n, data, codec, dtypes, planar=None):
    """The input as a contiguous (H, W, C) device array, and its C."""
    arr = cp.asarray(data)
    if arr.dtype not in dtypes:
        raise ValueError(
            f"{codec} encode: backend='nvimgcodec' takes "
            f"{', '.join(np.dtype(d).name for d in dtypes)}, not {arr.dtype}")
    if arr.ndim == 3 and planar is None:
        # imagecodecs' rule, which the CPU encoders follow too.
        planar = arr.shape[2] > 4 and arr.shape[0] <= 4
    if arr.ndim == 3 and planar:
        arr = cp.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] < 1:
        raise ValueError(
            f"{codec} encode: expected (H, W) or (H, W, C), got {arr.shape}")
    return cp.ascontiguousarray(arr)


def _encode(codec, cp, n, arr, params, fmt="jpeg2k"):
    engine = _engine()
    stream = engine.encode(codec, n.as_image(arr), fmt, params)
    if stream is None:
        raise OpenCodecsError(
            f"{codec} encode: nvImageCodec could not encode a "
            f"{tuple(arr.shape)} {arr.dtype} image on the GPU (see its log "
            f"above); encode it with backend=None")
    return bytes(stream)


def _resolutions(requested, height, width):
    """nvJPEG2000's resolution count, capped so the coarsest level keeps
    at least one sample on each axis (OpenJPEG caps the same way)."""
    count = max(1, int(requested))
    while count > 1 and min(height, width) >> (count - 1) < 1:
        count -= 1
    return count


def _reject(codec, **given):
    bad = sorted(k for k, v in given.items() if v is not None)
    if bad:
        raise ValueError(
            f"{codec} encode: backend='nvimgcodec' does not support "
            f"{', '.join(bad)}; use backend=None for them")


def encode_jpeg(data, *, level=None, subsampling=None, optimize=None,
                lossless=None, colorspace=None, outcolorspace=None,
                smoothing=None, predictor=None, bitspersample=None,
                validate=None, iccprofile=None) -> bytes:
    """Baseline 8-bit JPEG from uint8 gray or RGB, quality ``level``.

    Defaults match the CPU encoder: quality 95, 4:2:0 chroma, standard
    Huffman tables. ``optimize=True`` computes optimal tables.
    """
    cp, n = load()
    if lossless:
        raise ValueError("jpeg encode: backend='nvimgcodec' writes lossy "
                         "JPEG only; use backend=None for lossless=True")
    if bitspersample not in (None, 8):
        raise ValueError("jpeg encode: backend='nvimgcodec' writes 8-bit "
                         "JPEG only")
    if smoothing:
        raise ValueError("jpeg encode: smoothing is not supported")
    _reject("jpeg", colorspace=colorspace, outcolorspace=outcolorspace,
            predictor=predictor, iccprofile=iccprofile)
    arr = _device_image(cp, n, data, "jpeg", (np.uint8,), planar=False)
    channels = arr.shape[2]
    if channels not in (1, 3):
        raise ValueError(
            f"jpeg encode: backend='nvimgcodec' takes gray or RGB, not "
            f"{channels} samples")
    quality = 95 if level is None else int(level)
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg encode: level={level} is not in 1..100")
    if channels == 1:
        css = n.ChromaSubsampling.CSS_GRAY
    else:
        key = "420" if subsampling is None else subsampling
        key = tuple(key) if isinstance(key, (list, tuple)) else str(key)
        if key not in _SUBSAMPLING:
            raise ValueError(f"jpeg encode: subsampling={subsampling!r} is "
                             f"not one of '444', '422', '420', '440', '411'")
        css = getattr(n.ChromaSubsampling, _SUBSAMPLING[key])
    params = n.EncodeParams(
        quality_type=n.QualityType.QUALITY, quality_value=float(quality),
        chroma_subsampling=css,
        jpeg_encode_params=n.JpegEncodeParams(
            optimized_huffman=bool(optimize)))
    return _encode("jpeg", cp, n, arr, params, fmt="jpeg")


def encode_jpeg2k(data, *, level=None, lossless=None, ratio=None,
                  codec=None, codecformat=None, colorspace=None,
                  planar=None, tile=None, bitspersample=None,
                  resolutions=None, reversible=None, mct=True,
                  verbose=None) -> bytes:
    """JPEG 2000 with ``level`` and ``lossless`` as on the CPU.

    Lossless (the default) is the reversible 5/3 path. A ``level`` from 1
    to 1000 is a PSNR target in dB, lossy on the 9/7 path. ``ratio`` is
    not available.
    """
    cp, n = load()
    _reject("jpeg2k", colorspace=colorspace, tile=tile,
            bitspersample=bitspersample)
    fmt = (codecformat or codec or "jp2")
    fmt = str(fmt).lower()
    if fmt not in ("jp2", "j2k", "j2c"):
        raise ValueError(f"jpeg2k encode: codecformat={fmt!r} is not 'jp2' "
                         f"or 'j2k'")
    lossy_level = level is not None and 1 <= float(level) <= 1000
    if lossless and (lossy_level or ratio is not None or reversible is False):
        raise ValueError("jpeg2k encode: lossless=True with a lossy level, "
                         "ratio or reversible=False")
    # nvJPEG2000 (0.9) has no rate target: its SIZE_RATIO quality type
    # fails every encode, so ratio=, and lossless=False without a level
    # (a 10:1 rate on the CPU), are refused.
    if ratio is not None or (lossless is False and not lossy_level):
        raise ValueError(
            "jpeg2k encode: backend='nvimgcodec' has no compression-ratio "
            "target; pass a PSNR level= instead, or use backend=None")
    if lossy_level:
        quality = (n.QualityType.PSNR, float(level))
    else:
        if reversible is False:
            raise ValueError("jpeg2k encode: reversible=False needs a lossy "
                             "level or ratio")
        quality = (n.QualityType.LOSSLESS, 0.0)
    if quality[0] != n.QualityType.LOSSLESS and reversible:
        raise ValueError("jpeg2k encode: backend='nvimgcodec' has no lossy "
                         "5/3 (reversible) mode")
    arr = _device_image(cp, n, data, "jpeg2k",
                        (np.uint8, np.uint16, np.int16), planar)
    height, width, channels = arr.shape
    params = n.EncodeParams(
        quality_type=quality[0], quality_value=quality[1],
        jpeg2k_encode_params=n.Jpeg2kEncodeParams(
            bitstream_type=(n.Jpeg2kBitstreamType.JP2 if fmt == "jp2"
                            else n.Jpeg2kBitstreamType.J2K),
            num_resolutions=_resolutions(
                6 if resolutions is None else resolutions, height, width),
            # As on the CPU: the component transform for 3-sample color.
            mct_mode=1 if (mct and channels == 3) else 0))
    return _encode("jpeg2k", cp, n, arr, params)


def encode_htj2k(data, level=None, *, rgb=None, planar=None, tile=None,
                 resolutions=None, reversible=None, tlm=None,
                 tilepart=None, block_size=None, prog_order=None,
                 profile=None, num_decomp=None) -> bytes:
    """HTJ2K codestream (.j2c) with ``level`` as on the CPU.

    ``level=None`` is lossless (5/3), 1 to 100 a quality factor on the
    9/7 path. A quantization step (``level`` below 1) is not offered:
    nvJPEG2000's step is not on OpenJPH's scale.
    """
    cp, n = load()
    _reject("htj2k", tlm=tlm, tilepart=tilepart, profile=profile)
    if level is None or float(level) < 1e-5:
        if reversible is False:
            raise ValueError("htj2k encode: reversible=False needs a lossy "
                             "level")
        quality = (n.QualityType.LOSSLESS, 0.0)
    elif float(level) < 1:
        raise ValueError(
            "htj2k encode: backend='nvimgcodec' takes a quality from 1 to "
            "100; a quantization step (level below 1) needs backend=None")
    elif float(level) <= 100:
        if reversible:
            raise ValueError("htj2k encode: reversible=True with a lossy "
                             "level")
        quality = (n.QualityType.QUALITY, float(level))
    else:
        raise ValueError(f"htj2k encode: level={level} is not in 1..100")
    arr = _device_image(cp, n, data, "htj2k",
                        (np.uint8, np.uint16, np.int16), planar)
    height, width, channels = arr.shape
    if rgb and channels < 3:
        raise ValueError("htj2k encode: rgb=True needs 3 or more components")
    use_mct = (channels in (3, 4)) if rgb is None else bool(rgb)
    decomp = num_decomp if num_decomp is not None else resolutions
    decomp = 5 if not decomp else int(decomp)
    kw = dict(bitstream_type=n.Jpeg2kBitstreamType.J2K, ht=True,
              num_resolutions=_resolutions(decomp + 1, height, width),
              mct_mode=1 if use_mct else 0)
    if block_size is not None:
        bw, bh = block_size
        kw["code_block_size"] = (int(bh), int(bw))
    if prog_order is not None:
        name = str(prog_order).upper()
        if name not in _PROG_ORDER:
            raise ValueError(f"htj2k encode: prog_order={prog_order!r}")
        kw["prog_order"] = getattr(n.Jpeg2kProgOrder, name)
    extra = {}
    if tile is not None:
        tw, th = tile
        extra = dict(tile_width=int(tw), tile_height=int(th))
    params = n.EncodeParams(quality_type=quality[0],
                            quality_value=quality[1],
                            jpeg2k_encode_params=n.Jpeg2kEncodeParams(**kw),
                            **extra)
    return _encode("htj2k", cp, n, arr, params)
