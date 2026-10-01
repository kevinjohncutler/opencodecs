"""QuantizeCodec: lossy float quantization filters (netCDF-C BitRound and kin).

Quantization is a lossy *filter*: it rounds float values to a coarser
representation so that downstream byte/bit-level compressors (zstd,
floatpred+zstd, lerc) reach much better ratios. The output is still
the same dtype and shape as the input — quantize doesn't change the
wire format, just reduces the number of distinct values.

Modes. The first four are imagecodecs' ``quantize_encode`` modes, with
its parameter ``nsd``, and give imagecodecs' bits for float32 and
float64 except for the zeros, NaN and infinities it alters (below):

* ``"bitround"``: netCDF-C BitRound (Klower et al. 2021): keep ``nsd``
  explicit mantissa bits with round-to-nearest (add half, then mask);
  a value whose rounding carries past the largest exponent becomes an
  infinity, as in netCDF-C.
  ``bitspersample=`` is accepted as another name for the bit count.
* ``"bitgroom"``: netCDF-C BitGroom: keep enough bits for ``nsd``
  decimal digits, ceil(nsd * log2(10)) + 1, zeroing the rest of each
  even-indexed value and setting them in each odd-indexed one.
* ``"granularbr"`` (or ``"gbr"``): netCDF-C Granular BitRound: keep
  ``nsd`` decimal digits, with the bit count worked out per value.
* ``"scale"``: ``round(x * 2**b) / 2**b`` with ``2**b`` the power of two
  just above ``10**nsd`` (bcolz's and numcodecs' Quantize). Rounding is
  half away from zero, as C's ``round`` in imagecodecs.

The netCDF-C modes leave values equal to netCDF's default fill value
(9.9692099683868690e+36), ``+0.0``, ``-0.0`` and NaN unchanged, as
netCDF-C's ``nc4_convert_type`` does in all three ("Do not quantize
_FillValue, +/- zero, or NaN"). Infinities are kept as well. BitRound
leaves them unchanged by its own arithmetic; netCDF-C's BitGroom sets
the dropped bits of an odd-indexed infinity, which turns it into NaN,
and its Granular BitRound takes the logarithm of the infinity and
converts the result to ``int``, which C leaves undefined.

``nsd`` must be a whole number, at least 1 for the netCDF-C modes, as
``nc_def_var_quantize`` requires. BitRound takes at most the type's
mantissa bits (23 for float32, 52 for float64, 10 for float16).
BitGroom and Granular BitRound take at most the digits whose BitGroom
bit count, ceil(nsd * log2(10)) + 1, fits in the mantissa: 6 for
float32 and 15 for float64, netCDF-C's NC_QUANTIZE_MAX_FLOAT_NSD and
NC_QUANTIZE_MAX_DOUBLE_NSD, and 2 for float16. Beyond these the modes
have no bits left to drop (netCDF-C rejects the value; imagecodecs
returns the data unchanged or garbled), so they raise ``ValueError``.
``"scale"`` takes any ``nsd`` from 0, as imagecodecs does.

* ``"nsd"``: an opencodecs extension with no imagecodecs or netCDF
  counterpart: each value correctly rounded to ``nsd`` significant
  decimal digits (ties to even), then the value of the input type
  nearest that decimal; so a float32 1234.5678 at ``nsd=1`` is 1000,
  not 999.99994. Zeros, inf and NaN are kept, as is a value whose
  rounding would overflow the type.

float16 is accepted by every mode (10 mantissa bits); imagecodecs takes
float32 and float64 only.

Decode is the identity: quantize is irreversible by design, so the
"decoder" returns the data unchanged. ``mode`` and ``nsd`` are accepted
there for symmetry with encode and checked as encode checks them, so a
misspelled mode raises rather than passing unnoticed. (imagecodecs'
``quantize_decode`` raises for every mode instead.)
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from ._filter_args import reject_unknown


# Mantissa-bits-per-dtype lookup (IEEE 754).
_MANTISSA_BITS = {
    np.dtype("float16"): 10,
    np.dtype("float32"): 23,
    np.dtype("float64"): 52,
}

# netCDF-C's NC_FILL_FLOAT / NC_FILL_DOUBLE: the netCDF modes skip it.
_NC_FILL = 9.9692099683868690e+36

# netCDF-C's quantize mode numbers (NC_QUANTIZE_*), and imagecodecs' 100.
_MODES = {
    "bitgroom": "bitgroom", 1: "bitgroom",
    "granularbr": "granularbr", "gbr": "granularbr", 2: "granularbr",
    "bitround": "bitround", 3: "bitround",
    "scale": "scale", 100: "scale",
    "nsd": "nsd",
}

_BIT_PER_DGT = math.log(10) / math.log(2)
_DGT_PER_BIT = math.log(2) / math.log(10)


def _uint_view(arr: np.ndarray) -> np.ndarray:
    return arr.view(np.dtype(f"u{arr.dtype.itemsize}"))


def _quantizable(arr: np.ndarray) -> np.ndarray:
    """Mask of values the netCDF modes may change.

    netCDF-C skips the fill value, +0.0, -0.0 and NaN in every mode
    (``val != fill && val != 0.0 && !isnan(val)``). A bit pattern test
    for zero misses -0.0, which BitGroom's set step then turned into a
    negative subnormal. Infinities are skipped too: BitRound leaves them
    alone by itself, and BitGroom and Granular BitRound would turn them
    into NaN or into undefined C behavior.
    """
    with np.errstate(invalid="ignore"):      # signaling NaN payloads
        sel = np.isfinite(arr) & (arr != 0)
        if arr.dtype.itemsize != 2:          # float16 has no netCDF fill
            sel &= arr != arr.dtype.type(_NC_FILL)
    return sel


def _masks(arr: np.ndarray, zero_bits):
    """netCDF-C's shave mask, set mask and half-shave bit for ``zero_bits``."""
    udt = np.dtype(f"u{arr.dtype.itemsize}")
    ones = np.array(~udt.type(0), udt)
    shave = (ones << np.asarray(zero_bits).astype(udt)).astype(udt)
    keep_low = (~shave).astype(udt)
    half = (keep_low & (shave >> udt.type(1))).astype(udt)
    return shave, keep_low, half


def _bitround(arr: np.ndarray, keepbits: int) -> np.ndarray:
    """netCDF-C BitRound: add half of the dropped part, then mask it off."""
    dt = arr.dtype
    mantissa_bits = _MANTISSA_BITS.get(dt)
    if mantissa_bits is None:
        raise ValueError(
            f"quantize bitround: unsupported dtype {dt}")
    _check_nsd("bitround", keepbits, dt)
    out = np.ascontiguousarray(arr).copy()
    if keepbits == mantissa_bits:
        return out
    shave, _, half = _masks(arr, mantissa_bits - keepbits)
    bits = _uint_view(out)
    sel = _quantizable(out)
    bits[sel] = (bits[sel] + half) & shave        # wraps like C unsigned
    return out


def _bitgroom(arr: np.ndarray, nsd: int) -> np.ndarray:
    """netCDF-C BitGroom: shave even-indexed values, set odd-indexed ones."""
    mantissa_bits = _MANTISSA_BITS[arr.dtype]
    keep = math.ceil(nsd * _BIT_PER_DGT) + 1
    out = np.ascontiguousarray(arr).copy()
    zero_bits = mantissa_bits - keep
    if zero_bits <= 0:                # _check_nsd keeps nsd below this
        raise ValueError(f"quantize bitgroom: nsd {nsd} is too large")
    shave, keep_low, _ = _masks(arr, zero_bits)
    bits = _uint_view(out).reshape(-1)
    sel = _quantizable(out).reshape(-1)
    even, odd = bits[0::2], bits[1::2]
    sel_even, sel_odd = sel[0::2], sel[1::2]
    even[sel_even] &= shave
    odd[sel_odd] |= keep_low
    return out


def _granularbr(arr: np.ndarray, nsd: int) -> np.ndarray:
    """netCDF-C Granular BitRound: BitRound with a per-value bit count."""
    mantissa_bits = _MANTISSA_BITS[arr.dtype]
    out = np.ascontiguousarray(arr).copy()
    bits = _uint_view(out).reshape(-1)
    with np.errstate(invalid="ignore"):          # signaling NaN payloads
        vals = out.reshape(-1).astype(np.float64)
    sel = _quantizable(out).reshape(-1)
    idx = np.flatnonzero(sel)
    if not idx.size:
        return out
    mnt, xpn = np.frexp(vals[idx])                 # DGG19 (8)
    mnt_log10 = np.log10(np.abs(mnt))
    dgt = np.floor(xpn * _DGT_PER_BIT + mnt_log10).astype(np.int64) + 1
    qnt_pwr = np.floor(_BIT_PER_DGT * (dgt - nsd)).astype(np.int64)
    keep = np.abs(np.floor(xpn - _BIT_PER_DGT * mnt_log10).astype(np.int64)
                  - qnt_pwr) - 1             # netCDF-C keeps one bit fewer
    zero_bits = mantissa_bits - keep
    if np.any((keep < 0) | (zero_bits <= 0)):
        # Not reached for the nsd _check_nsd allows: keep stayed within
        # 0 to 20 bits for float32, 0 to 50 for float64 and 0 to 8 for
        # float16 over every float16 value and two million random float32
        # and float64 bit patterns. netCDF-C would shift by a negative
        # count here.
        raise ValueError(f"quantize granularbr: nsd {nsd} is out of range")
    shave, _, half = _masks(arr, zero_bits)
    bits[idx] = (bits[idx] + half) & shave
    return out


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """C's ``round``: halves away from zero (np.round goes to even)."""
    mag = np.abs(x)
    whole = np.floor(mag)
    whole += (mag - whole) >= 0.5
    return np.copysign(whole, x)


def _scale(arr: np.ndarray, nsd: int) -> np.ndarray:
    """imagecodecs' ``scale``: round to a power-of-two grid finer than 10**-nsd."""
    exp = math.log10(10.0 ** -nsd)
    exp = math.floor(exp) if exp < 0.0 else math.ceil(exp)
    scale = 2.0 ** math.ceil(math.log2(10.0 ** -exp))
    vals = arr.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        return (_round_half_away(vals * scale) / scale).astype(arr.dtype)


# Powers of ten that float64 holds exactly: 10**0 .. 10**22.
_EXACT_POW10 = 22


def _decimal_round_one(x: float, nsd: int) -> float:
    """``x`` correctly rounded to ``nsd`` significant decimal digits.

    Python formats a float from its exact binary value, rounding half to
    even, and parses the digits back to the nearest float.
    """
    return float(f"{x:.{nsd - 1}e}")


def _nsd_round(arr: np.ndarray, nsd: int) -> np.ndarray:
    """Round to ``nsd`` significant decimal digits (opencodecs extension).

    The result is the value of the input type nearest to ``x`` correctly
    rounded to ``nsd`` decimal digits (ties to even). With
    ``e = floor(log10|x|) - nsd + 1`` and ``p = 10**|e|`` exact, the
    quotient ``x / p`` (or ``x * p`` for negative ``e``) is rounded to an
    integer ``r`` and ``r * p`` (or ``r / p``) is one correctly rounded
    operation on exact operands. The quotient itself can round, which
    only matters when it lies within two units in the last place of a
    half; those values, and values whose ``|e|`` exceeds the exact powers
    of ten, are rounded one at a time from their decimal digits. The
    scale used to be the power ``10**(nsd - 1 - floor(log10|x|))``,
    inexact for large ``|x|``, so 1175103902.8858647 at one digit gave
    999999999.9999999, and for float32 input it was computed in float32.
    The float64 found is cast to a narrower input type, and the few casts
    that would round a second time are decided from the decimal.
    Zeros, infinities and NaN pass through unchanged, as does a value
    whose rounding would overflow the type.
    """
    if nsd < 1:
        raise ValueError(f"quantize nsd: nsd must be >= 1, got {nsd}")
    out = arr.copy()
    if nsd >= 17:
        # 17 significant digits name every float64 (and so every float32
        # and float16) uniquely: rounding to them gives the value back.
        return out
    with np.errstate(invalid="ignore"):         # signaling NaN payloads
        vals = arr.astype(np.float64)
    sel = (vals != 0) & np.isfinite(vals)
    if not np.any(sel):
        return out
    v = vals[sel]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.floor(np.log10(np.abs(v))).astype(np.int64) - (nsd - 1)
        exact = np.abs(e) <= _EXACT_POW10
        p = np.power(10.0, np.where(exact, np.abs(e), 0))
        q = np.where(e >= 0, v / p, v * p)
        r = np.round(q)
        rounded = np.where(e >= 0, r * p, r / p)
        # A quotient within two ulps of a half may sit on the wrong side
        # of it; log10 can also be one off right at a power of ten, which
        # shows as a quotient outside [10**(nsd-1), 10**nsd].
        frac = np.abs(np.abs(q) - np.floor(np.abs(q)) - 0.5)
        unsure = ~exact | (frac <= 2 * np.spacing(np.abs(q)))
        unsure |= (np.abs(q) < 10.0 ** (nsd - 1)) | (np.abs(q) >= 10.0 ** nsd)
        unsure |= np.abs(q) >= 2.0 ** 53        # r would not be exact
    for i in np.flatnonzero(unsure):
        rounded[i] = _decimal_round_one(float(v[i]), nsd)
    with np.errstate(over="ignore"):
        cast = rounded.astype(arr.dtype)
    if arr.dtype != np.float64:
        _fix_double_rounding(cast, rounded, v, nsd)
    out[sel] = np.where(np.isfinite(cast), cast, arr[sel])
    return out


def _fix_double_rounding(cast, rounded, v, nsd) -> None:
    """Correct ``cast`` (float32 or float16) where it was rounded twice.

    ``rounded`` is the float64 nearest the decimal. Casting it to a
    narrower type picks the value nearest the decimal unless ``rounded``
    lies exactly halfway between two values of that type while the
    decimal itself does not; the cast then breaks a tie the decimal does
    not have (float32 7.0385307e-26 at 7 digits is 7.038531e-26, nearest
    7.0385307e-26, but the cast gives 7.0385313e-26). Those values are
    decided from the exact decimal.
    """
    mag = np.abs(rounded)
    with np.errstate(over="ignore", invalid="ignore"):
        c = mag.astype(cast.dtype)
        back = c.astype(np.float64)
        below = np.where(back > mag, np.nextafter(c, c.dtype.type(0)), c)
        low = below.astype(np.float64)
        step = np.spacing(below).astype(np.float64)
        # Past the largest finite value, the step to the overflow threshold.
        step = np.where(np.isinf(step),
                        low - np.nextafter(below, c.dtype.type(0)), step)
        half = step / 2
        tie = (back != mag) & (low + half == mag)
    for i in np.flatnonzero(tie):
        exact = Fraction(f"{abs(float(v[i])):.{nsd - 1}e}")
        if exact == Fraction(float(mag[i])):
            continue                            # a true tie: the cast is right
        pick = below[i] if exact < Fraction(float(mag[i])) else np.nextafter(
            below[i], cast.dtype.type(np.inf))
        cast[i] = -pick if rounded[i] < 0 else pick


_NETCDF_MODES = ("bitround", "bitgroom", "granularbr")


def _max_nsd(mode: str, dtype: np.dtype) -> int | None:
    """The largest ``nsd`` ``mode`` takes for ``dtype`` (None: no limit).

    BitRound: the mantissa bits. BitGroom and Granular BitRound: the most
    digits whose BitGroom bit count ceil(nsd * log2(10)) + 1 fits in the
    mantissa, which is netCDF-C's NC_QUANTIZE_MAX_FLOAT_NSD (6) and
    NC_QUANTIZE_MAX_DOUBLE_NSD (15).
    """
    bits = _MANTISSA_BITS.get(np.dtype(dtype).newbyteorder("="))
    if bits is None or mode not in _NETCDF_MODES:
        return None
    if mode == "bitround":
        return bits
    nsd = 0
    while math.ceil((nsd + 1) * _BIT_PER_DGT) + 1 <= bits:
        nsd += 1
    return nsd


def _check_nsd(mode: str, nsd: int, dtype) -> None:
    """Raise for an ``nsd`` netCDF-C's nc_def_var_quantize would reject
    (and, for the ``"nsd"`` extension, below one digit)."""
    if (mode in _NETCDF_MODES or mode == "nsd") and nsd < 1:
        raise ValueError(
            f"quantize {mode}: nsd must be >= 1, got {nsd}")
    top = _max_nsd(mode, dtype)
    if top is not None and nsd > top:
        what = "bits" if mode == "bitround" else "digits"
        raise ValueError(
            f"quantize {mode}: {np.dtype(dtype).name} holds at most {top} "
            f"{what} for this mode, got nsd={nsd}")


def _as_whole(value, name: str) -> int:
    """``value`` as an int; a fractional value raises (it was truncated)."""
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"quantize: {name} must be a whole number, got {value!r}")
    raise TypeError(f"quantize: {name} must be an integer, got {value!r}")


def _resolve_params(mode, nsd, bitspersample):
    """Check ``mode``, ``nsd`` and ``bitspersample``; return (mode, nsd)."""
    key = mode.lower() if isinstance(mode, str) else mode
    resolved = _MODES.get(key) if isinstance(key, (str, int)) else None
    if resolved is None:
        raise ValueError(
            f"quantize: unsupported mode {mode!r}; expected 'bitround', "
            f"'bitgroom', 'granularbr' (or 'gbr'), 'scale' or 'nsd'")
    if bitspersample is not None:
        bitspersample = _as_whole(bitspersample, "bitspersample")
        if resolved != "bitround":
            raise ValueError(
                f"quantize {resolved}: bitspersample= is the BitRound "
                f"bit count; use nsd=")
        if nsd is not None and _as_whole(nsd, "nsd") != bitspersample:
            raise ValueError(
                f"quantize bitround: nsd={nsd} and bitspersample="
                f"{bitspersample} disagree")
        nsd = bitspersample
    if nsd is not None:
        nsd = _as_whole(nsd, "nsd")
        if nsd < 0:
            raise ValueError(
                f"quantize {resolved}: nsd must be >= 0, got {nsd}")
    return resolved, nsd


class QuantizeCodec(Codec):
    """Lossy float quantization filter (netCDF-C modes, scale, nsd)."""

    name = "quantize"
    aliases = ()
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.float16, np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # transparent filter — no magic

    def encode(
        self,
        data: Any,
        mode="bitround",
        nsd: int | None = None,
        *,
        dest=None,
        bitspersample: int | None = None,
        **opts,
    ) -> bytes | None:
        """Quantize ``data``; ``mode`` and ``nsd`` as in imagecodecs.

        They may be given by position, ``encode(data, "bitround", 7)``,
        as imagecodecs' ``quantize_encode(data, mode, nsd)`` takes them.
        """
        reject_unknown("quantize encode", opts)
        arr = np.ascontiguousarray(data)
        native = arr.dtype.newbyteorder("=")
        if arr.dtype.kind != "f" or native not in _MANTISSA_BITS:
            raise ValueError(
                f"quantize: requires a float16, float32 or float64 dtype, "
                f"got {arr.dtype}")
        resolved, nsd = _resolve_params(mode, nsd, bitspersample)
        if nsd is None:
            raise ValueError(f"quantize {resolved}: nsd= is required")
        _check_nsd(resolved, nsd, native)
        work = arr.astype(native, copy=False)
        if resolved == "bitround":
            out = _bitround(work, nsd)
        elif resolved == "bitgroom":
            out = _bitgroom(work, nsd)
        elif resolved == "granularbr":
            out = _granularbr(work, nsd)
        elif resolved == "scale":
            out = _scale(work, nsd)
        else:
            out = _nsd_round(work, nsd)
        return _write_dest(out.astype(arr.dtype, copy=False).tobytes(), dest)

    def decode(
        self,
        src: Any,
        mode=None,
        nsd=None,
        *,
        dtype=None,
        shape=None,
        out=None,
        bitspersample=None,
        **opts,
    ) -> np.ndarray:
        """Return the quantized values unchanged (quantization is final).

        ``mode``, ``nsd`` and ``bitspersample`` are accepted for symmetry
        with encode and do not change the result, but they are checked as
        encode checks them: an unknown mode, or an ``nsd`` encode would
        reject for ``dtype``, raises.
        An ndarray ``src`` gives ``dtype`` and ``shape`` by default; bytes
        need ``dtype``.
        """
        reject_unknown("quantize decode", opts)
        checked = None
        if mode is not None or nsd is not None or bitspersample is not None:
            checked = _resolve_params("bitround" if mode is None else mode,
                                      nsd, bitspersample)
        if isinstance(src, np.ndarray):
            if dtype is None:
                dtype = src.dtype
            if shape is None and np.dtype(dtype) == src.dtype:
                shape = src.shape
        if dtype is None:
            raise ValueError("quantize decode: dtype= is required")
        if checked is not None and checked[1] is not None:
            _check_nsd(checked[0], checked[1], dtype)
        buf = _read_src(src)
        arr = np.frombuffer(buf, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"quantize decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError(
                    "quantize decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        # Caller must own the buffer; frombuffer's result is read-only over
        # an immutable bytes input — copy so the result is writable.
        return arr.copy()


__all__ = ["QuantizeCodec"]
