"""HEIF through Apple's ImageIO and its hardware HEVC codec (macOS only).

Selected with ``backend="imageio"`` on the heif codec; never chosen by
default. The frameworks are reached through ctypes, so no Python package
is needed, and nothing is loaded until the first call that asks for it.

Decode. ImageIO hands the decode to the media engine, which is 10 to 19
times faster than libde265 on a large image held as ONE coded picture
(measured on an M1 Ultra, 4096 x 3072). A HEIF whose image is a grid of
small tiles, which is how iPhones and ImageIO itself write them, gains
nothing: libheif decodes the tiles on several threads and was as fast.
So ``backend="imageio"`` sends a file to ImageIO only when the image is
one picture (or a grid of tiles at least ``MIN_TILE`` on a side), coded at
8 bits in color, with no alpha plane and no rotation, mirror or crop
property, and only for the primary image; every other file is decoded by
libheif exactly as with ``backend=None``. :func:`route` says which.

The pixels are the decoder's own output (CGDataProviderCopyData, the
image's stored samples; drawing into a bitmap context instead changed
some): within one level of libde265's for 4:4:4 and exact for lossless.
For 4:2:0 the two upsample chroma differently, up to 4 to 6 levels on
photographic content. ImageIO returns four bytes a pixel; Accelerate's
vImage drops the fourth into a (H, W, 3) array, which costs about 3 ms
for 4096 x 3072, against 55 ms for the same copy in numpy.

Encode. ``encode(..., backend="imageio")`` uses the hardware HEVC
encoder: 5 to 15 times faster than libheif/x265, but 8-bit 4:2:0 only,
lossy only, in 512 x 512 tiles, with coarse quality steps and files 1.4
to 3.2 times larger at equal PSNR. Use it when encode speed matters more
than size. ``level`` is passed to ImageIO as its quality (``level/100``);
it is not on libheif's quality scale.
"""

from __future__ import annotations

import ctypes as C
import logging
import struct
import threading

import numpy as np

from . import BackendUnavailable
from ..core.errors import OpenCodecsError

_log = logging.getLogger("opencodecs")

# Tiles at least this many pixels on each side are worth the hardware
# decoder. ImageIO's own 512 x 512 grids were no faster than libheif.
MIN_TILE = 1024

# Auxiliary-image types that mean "this is the alpha plane" (HEVC's and
# MPEG-B's spellings); a depth map or other auxiliary image does not
# change how the primary image decodes.
_ALPHA_URNS = (b"urn:mpeg:hevc:2015:auxid:1",
               b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")

_lock = threading.Lock()
_api = None

_FRAMEWORK = "/System/Library/Frameworks/{0}.framework/{0}"

# CGBitmapInfo / CGImageAlphaInfo values.
_ALPHA_MASK = 0x1F
_ORDER_MASK = 0x7000
_ORDER_32_LITTLE = 2 << 12
_FLOAT = 1 << 8
_ALPHA_NONE_SKIP_LAST = 5
_ALPHA_NONE_SKIP_FIRST = 6
_ALPHA_NONE = 0
_COLOR_MODEL_RGB = 1
_CF_UTF8 = 0x08000100
_CF_DOUBLE = 13


class _VImageBuffer(C.Structure):
    _fields_ = [("data", C.c_void_p), ("height", C.c_size_t),
                ("width", C.c_size_t), ("rowBytes", C.c_size_t)]


class _Api:
    """The CoreFoundation, CoreGraphics, ImageIO and vImage entry points."""

    def __init__(self):
        cf = C.CDLL(_FRAMEWORK.format("CoreFoundation"))
        cg = C.CDLL(_FRAMEWORK.format("CoreGraphics"))
        io = C.CDLL(_FRAMEWORK.format("ImageIO"))
        acc = C.CDLL(_FRAMEWORK.format("Accelerate"))
        vp, sz, u32, i32 = C.c_void_p, C.c_size_t, C.c_uint32, C.c_int32
        cfindex = C.c_long

        def bind(lib, name, res, *args):
            fn = getattr(lib, name)
            fn.restype = res
            fn.argtypes = list(args)
            setattr(self, name, fn)

        bind(cf, "CFRelease", None, vp)
        bind(cf, "CFDataCreateWithBytesNoCopy", vp, vp, vp, cfindex, vp)
        bind(cf, "CFDataCreateMutable", vp, vp, cfindex)
        bind(cf, "CFDataGetBytePtr", vp, vp)
        bind(cf, "CFDataGetLength", cfindex, vp)
        bind(cf, "CFStringCreateWithCString", vp, vp, C.c_char_p, u32)
        bind(cf, "CFNumberCreate", vp, vp, cfindex, vp)
        bind(cf, "CFDictionaryCreate", vp, vp, C.POINTER(vp), C.POINTER(vp),
             cfindex, vp, vp)
        bind(io, "CGImageSourceCreateWithData", vp, vp, vp)
        bind(io, "CGImageSourceGetCount", sz, vp)
        bind(io, "CGImageSourceGetPrimaryImageIndex", sz, vp)
        bind(io, "CGImageSourceCreateImageAtIndex", vp, vp, sz, vp)
        bind(io, "CGImageDestinationCreateWithData", vp, vp, vp, sz, vp)
        bind(io, "CGImageDestinationAddImage", None, vp, vp, vp)
        bind(io, "CGImageDestinationFinalize", C.c_bool, vp)
        for name in ("CGImageGetWidth", "CGImageGetHeight",
                     "CGImageGetBitsPerComponent", "CGImageGetBitsPerPixel",
                     "CGImageGetBytesPerRow"):
            bind(cg, name, sz, vp)
        bind(cg, "CGImageGetBitmapInfo", u32, vp)
        bind(cg, "CGImageGetColorSpace", vp, vp)
        bind(cg, "CGColorSpaceGetModel", i32, vp)
        bind(cg, "CGImageGetDataProvider", vp, vp)
        bind(cg, "CGDataProviderCopyData", vp, vp)
        bind(cg, "CGDataProviderCreateWithData", vp, vp, vp, sz, vp)
        bind(cg, "CGColorSpaceCreateWithName", vp, vp)
        bind(cg, "CGImageCreate", vp, sz, sz, sz, sz, sz, vp, u32, vp, vp,
             C.c_bool, i32)
        for name in ("vImageConvert_RGBA8888toRGB888",
                     "vImageConvert_ARGB8888toRGB888",
                     "vImageConvert_BGRA8888toRGB888"):
            bind(acc, name, C.c_ssize_t, C.POINTER(_VImageBuffer),
                 C.POINTER(_VImageBuffer), u32)

        def const(lib, name):
            return vp.in_dll(lib, name).value

        self.kCFAllocatorNull = const(cf, "kCFAllocatorNull")
        self.key_callbacks = C.addressof(
            C.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
        self.value_callbacks = C.addressof(
            C.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))
        self.kQuality = const(io, "kCGImageDestinationLossyCompressionQuality")
        self.kSRGB = const(cg, "kCGColorSpaceSRGB")
        self.heic = self.CFStringCreateWithCString(None, b"public.heic",
                                                   _CF_UTF8)


def load() -> _Api:
    """Load the frameworks once; raise BackendUnavailable where they are
    missing (anywhere but macOS). The frameworks themselves are the
    probe, so there is no operating-system test to keep in step."""
    global _api
    if _api is not None:
        return _api
    with _lock:
        if _api is not None:
            return _api
        try:
            _api = _Api()
        except (OSError, AttributeError, ValueError) as exc:
            raise BackendUnavailable(
                f"backend='imageio' needs Apple's ImageIO, CoreGraphics and "
                f"Accelerate frameworks, which only macOS has: {exc}") from exc
        return _api


# ------------------------------------------------------- HEIF structure

def _boxes(data, start, end):
    """(type, payload start, box end) for each ISOBMFF box in a range."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        head = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            return
        yield kind, pos + head, pos + size
        pos += size


def _items(data):
    """Primary item id, item types, references and properties of a HEIF."""
    meta = next(((s, e) for k, s, e in _boxes(data, 0, len(data))
                 if k == b"meta"), None)
    if meta is None:
        return None
    primary, types, refs, props, assoc = None, {}, [], [], {}
    for kind, s, e in _boxes(data, meta[0] + 4, meta[1]):
        version = data[s]
        if kind == b"pitm":
            primary = (struct.unpack_from(">H", data, s + 4)[0] if version == 0
                       else struct.unpack_from(">I", data, s + 4)[0])
        elif kind == b"iinf":
            first = s + (6 if version == 0 else 8)
            for k2, s2, e2 in _boxes(data, first, e):
                if k2 != b"infe" or data[s2] < 2:
                    continue
                if data[s2] == 2:
                    item = struct.unpack_from(">H", data, s2 + 4)[0]
                    at = s2 + 8
                else:
                    item = struct.unpack_from(">I", data, s2 + 4)[0]
                    at = s2 + 10
                types[item] = bytes(data[at:at + 4])
        elif kind == b"iref":
            wide = version != 0
            for k2, s2, e2 in _boxes(data, s + 4, e):
                fmt, step = (">I", 4) if wide else (">H", 2)
                src = struct.unpack_from(fmt, data, s2)[0]
                count = struct.unpack_from(">H", data, s2 + step)[0]
                at = s2 + step + 2
                dst = [struct.unpack_from(fmt, data, at + i * step)[0]
                       for i in range(count)]
                refs.append((bytes(k2), src, dst))
        elif kind == b"iprp":
            for k2, s2, e2 in _boxes(data, s, e):
                if k2 == b"ipco":
                    props = [(bytes(k3), s3, e3)
                             for k3, s3, e3 in _boxes(data, s2, e2)]
                elif k2 == b"ipma":
                    v2, flags = data[s2], data[s2 + 3]
                    count = struct.unpack_from(">I", data, s2 + 4)[0]
                    at = s2 + 8
                    for _ in range(count):
                        if v2 < 1:
                            item = struct.unpack_from(">H", data, at)[0]
                            at += 2
                        else:
                            item = struct.unpack_from(">I", data, at)[0]
                            at += 4
                        n = data[at]
                        at += 1
                        idx = []
                        for _ in range(n):
                            if flags & 1:
                                idx.append(struct.unpack_from(">H", data, at)[0]
                                           & 0x7FFF)
                                at += 2
                            else:
                                idx.append(data[at] & 0x7F)
                                at += 1
                        assoc.setdefault(item, []).extend(idx)
    return primary, types, refs, props, assoc


def _clap_is_identity(data, props) -> bool:
    """True if the clean-aperture box keeps the whole coded image."""
    ispe = props.get(b"ispe")
    if ispe is None:
        return False
    width, height = struct.unpack_from(">II", data, ispe[0] + 4)
    wn, wd, hn, hd, xn, xd, yn, yd = struct.unpack_from(
        ">IIIIiIiI", data, props[b"clap"][0])
    if not (wd and hd and xd and yd):
        return False
    return wn == width * wd and hn == height * hd and xn == 0 and yn == 0


def route(data, *, index=None, photometric=None) -> tuple[str, str]:
    """``("imageio", "")`` if ImageIO should decode ``data``, else
    ``("native", reason)``. Reads only the container's metadata."""
    if index is not None:
        return "native", "index= selects an image other than the primary"
    if photometric is not None:
        return "native", "photometric= is a libheif decode option"
    try:
        parsed = _items(memoryview(data))
    except (struct.error, IndexError, ValueError):
        parsed = None
    if not parsed or parsed[0] is None:
        return "native", "could not read the HEIF item structure"
    primary, types, refs, props, assoc = parsed

    def properties(item):
        out = {}
        for i in assoc.get(item, ()):
            if 1 <= i <= len(props):
                kind, s, e = props[i - 1]
                out.setdefault(kind, (s, e))
        return out

    kind = types.get(primary)
    coded = primary
    if kind == b"grid":
        tiles = [d for k, s, d in refs if k == b"dimg" and s == primary]
        tiles = tiles[0] if tiles else []
        if not tiles:
            return "native", "grid image without tiles"
        coded = tiles[0]
        ispe = properties(coded).get(b"ispe")
        if ispe is None:
            return "native", "grid tile without a size"
        tw, th = struct.unpack_from(">II", data, ispe[0] + 4)
        if min(tw, th) < MIN_TILE:
            return "native", (f"a grid of {tw} x {th} tiles, which libheif "
                              f"decodes on several threads as fast")
        if types.get(coded) != b"hvc1":
            return "native", "grid tiles are not HEVC"
    elif kind != b"hvc1":
        return "native", f"primary image is {kind!r}, not HEVC"
    mine = properties(primary)
    if b"imir" in mine:
        return "native", "the image is mirrored (imir)"
    if b"irot" in mine and data[mine[b"irot"][0]] & 3:
        return "native", "the image is rotated (irot)"
    if b"clap" in mine and not _clap_is_identity(data, mine):
        return "native", "the image is cropped (clap)"
    hvcc = properties(coded).get(b"hvcC")
    if hvcc is None or hvcc[1] - hvcc[0] < 19:
        return "native", "no HEVC configuration"
    s = hvcc[0]
    chroma, luma, chroma_bits = data[s + 16] & 3, data[s + 17] & 7, data[s + 18] & 7
    if chroma == 0:
        return "native", "monochrome"
    if luma or chroma_bits:
        return "native", f"{8 + luma}-bit samples"
    for k, src, dst in refs:
        if k == b"auxl" and primary in dst:
            auxc = properties(src).get(b"auxC")
            # auxC is a full box; its aux_type is a NUL-terminated URN.
            urn = (b"" if auxc is None else
                   bytes(data[auxc[0] + 4:auxc[1]]).split(b"\0")[0])
            if auxc is None or urn in _ALPHA_URNS:
                return "native", "the image has an alpha plane"
    return "imageio", ""


# --------------------------------------------------------------- decode

def decode(data, *, out=None):
    """Decode the primary image with ImageIO; None if ImageIO's pixels are
    not 8-bit color (the caller then uses libheif)."""
    api = load()
    buf = np.frombuffer(data, np.uint8)
    cfdata = api.CFDataCreateWithBytesNoCopy(
        None, buf.ctypes.data, buf.size, api.kCFAllocatorNull)
    source = image = copied = None
    try:
        source = api.CGImageSourceCreateWithData(cfdata, None)
        if not source or api.CGImageSourceGetCount(source) < 1:
            raise OpenCodecsError("heif decode: ImageIO cannot read this file")
        image = api.CGImageSourceCreateImageAtIndex(
            source, api.CGImageSourceGetPrimaryImageIndex(source), None)
        if not image:
            raise OpenCodecsError("heif decode: ImageIO could not decode "
                                  "the primary image")
        width = api.CGImageGetWidth(image)
        height = api.CGImageGetHeight(image)
        info = api.CGImageGetBitmapInfo(image)
        alpha = info & _ALPHA_MASK
        if (api.CGImageGetBitsPerComponent(image) != 8
                or api.CGImageGetBitsPerPixel(image) != 32
                or info & _FLOAT
                or alpha not in (_ALPHA_NONE_SKIP_LAST, _ALPHA_NONE_SKIP_FIRST)
                or api.CGColorSpaceGetModel(api.CGImageGetColorSpace(image))
                != _COLOR_MODEL_RGB):
            return None
        copied = api.CGDataProviderCopyData(api.CGImageGetDataProvider(image))
        if not copied:
            raise OpenCodecsError("heif decode: ImageIO returned no pixels")
        row_bytes = api.CGImageGetBytesPerRow(image)
        if api.CFDataGetLength(copied) < row_bytes * height:
            raise OpenCodecsError("heif decode: ImageIO returned too few bytes")
        shape = (height, width, 3)
        if out is None:
            result = np.empty(shape, np.uint8)
        else:
            from ..core.buffers import array_output
            result = array_output(out)
            if result.shape != shape or result.dtype != np.uint8:
                raise ValueError(
                    f"heif decode: out= is {result.shape} {result.dtype}; "
                    f"the image is {shape} uint8")
        little = (info & _ORDER_MASK) == _ORDER_32_LITTLE
        first = alpha == _ALPHA_NONE_SKIP_FIRST
        if little and first:        # memory order B G R X
            convert = api.vImageConvert_BGRA8888toRGB888
        elif first:                 # X R G B
            convert = api.vImageConvert_ARGB8888toRGB888
        elif not little:            # R G B X
            convert = api.vImageConvert_RGBA8888toRGB888
        else:
            return None             # X B G R: no vImage converter
        src = _VImageBuffer(api.CFDataGetBytePtr(copied), height, width,
                            row_bytes)
        dst = _VImageBuffer(result.ctypes.data, height, width, width * 3)
        err = convert(C.byref(src), C.byref(dst), 0)
        if err:
            raise OpenCodecsError(f"heif decode: vImage error {err}")
        return result
    finally:
        for ref in (copied, image, source, cfdata):
            if ref:
                api.CFRelease(ref)


def decode_heif(data, native_decode, *, out=None, index=None,
                photometric=None, **native_kw):
    """The heif codec's ``backend="imageio"`` decode: ImageIO where
    :func:`route` says it helps, libheif (``native_decode``) otherwise."""
    load()
    path, reason = route(data, index=index, photometric=photometric)
    if path == "imageio":
        result = decode(data, out=out)
        if result is not None:
            return result
        reason = "ImageIO returned pixels that are not 8-bit RGB"
    _log.debug("heif decode, backend='imageio': using libheif (%s)", reason)
    return native_decode(data, out=out, index=index, photometric=photometric,
                         **native_kw)


# --------------------------------------------------------------- encode

def encode(data, *, level=None, lossless=None, bit_depth=None,
           iccprofile=None, color=None) -> bytes:
    """HEIC from (H, W, 3) uint8 with the hardware HEVC encoder.

    Lossy only, 8-bit 4:2:0. ``level`` (0 to 100) is required and becomes
    ImageIO's quality ``level / 100``.
    """
    api = load()
    if lossless:
        raise ValueError("heif encode: backend='imageio' is lossy only; "
                         "use backend=None for lossless")
    if level is None:
        raise ValueError(
            "heif encode: backend='imageio' is lossy only and the codec's "
            "default is lossless; pass level= (0 to 100) to accept loss")
    if not 0 <= float(level) <= 100:
        raise ValueError(f"heif encode: level={level} is not in 0..100 for "
                         f"backend='imageio'")
    if bit_depth not in (None, 8):
        raise ValueError("heif encode: backend='imageio' writes 8 bits only")
    if iccprofile is not None or color is not None:
        raise ValueError("heif encode: backend='imageio' does not take "
                         "iccprofile= or color=; it writes sRGB")
    arr = np.ascontiguousarray(data)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"heif encode: backend='imageio' takes (H, W, 3) uint8, not "
            f"{arr.shape} {arr.dtype}")
    height, width = arr.shape[:2]
    quality = C.c_double(float(level) / 100.0)
    space = provider = image = number = props = output = dest = None
    try:
        space = api.CGColorSpaceCreateWithName(api.kSRGB)
        provider = api.CGDataProviderCreateWithData(
            None, arr.ctypes.data, arr.nbytes, None)
        image = api.CGImageCreate(width, height, 8, 24, width * 3, space,
                                  _ALPHA_NONE, provider, None, False, 0)
        if not image:
            raise OpenCodecsError("heif encode: CGImageCreate failed")
        number = api.CFNumberCreate(None, _CF_DOUBLE, C.byref(quality))
        keys = (C.c_void_p * 1)(api.kQuality)
        values = (C.c_void_p * 1)(number)
        props = api.CFDictionaryCreate(None, keys, values, 1,
                                       api.key_callbacks, api.value_callbacks)
        output = api.CFDataCreateMutable(None, 0)
        dest = api.CGImageDestinationCreateWithData(output, api.heic, 1, None)
        if not dest:
            raise OpenCodecsError("heif encode: this macOS has no HEIC encoder")
        api.CGImageDestinationAddImage(dest, image, props)
        if not api.CGImageDestinationFinalize(dest):
            raise OpenCodecsError("heif encode: ImageIO could not encode")
        return C.string_at(api.CFDataGetBytePtr(output),
                           api.CFDataGetLength(output))
    finally:
        for ref in (dest, output, props, number, image, provider, space):
            if ref:
                api.CFRelease(ref)
