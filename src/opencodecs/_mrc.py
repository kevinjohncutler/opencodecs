"""MRC / CCP4 map reader (MRC2014).

MRC is the volume format of cryo-electron microscopy and the map format
of CCP4 crystallography. Everything downstream of a detector speaks it:
motion-corrected micrographs, particle stacks, class averages, tomograms,
and the density maps deposited in EMDB. ``opencodecs`` already reads what
Falcon detectors write (see :mod:`opencodecs._eer_reader`); this reads
what the rest of the pipeline converts that into.

The format is deliberately simple, which is why it has outlived several
attempts to replace it: a fixed 1024-byte header, an optional extended
header, then the voxels, uncompressed and in axis order. That makes a
plane or a sub-volume a pure offset computation, so this reader is built
on the same ``read_at`` contract as the TIFF and FITS readers and never
needs the whole file resident.

Reference: the MRC2014 specification, Cheng et al. 2015, J. Struct. Biol.
192(2):146-150, and the CCP4/CCP-EM header definition it formalizes. No
code is derived from any implementation; the header layout below is
transcribed from that published specification.
"""

from __future__ import annotations

import os
import struct
from typing import Any, Callable

import numpy as np

HEADER_SIZE = 1024

# MODE -> dtype. The gaps are real: the specification assigns 3 and 4 to
# complex types and 101 to a 4-bit packed mode, and leaves the rest
# unassigned.
_MODES: dict[int, np.dtype] = {
    0: np.dtype("i1"),      # signed 8-bit. See _dtype_for_mode: some
                            # writers mean unsigned here.
    1: np.dtype("i2"),
    2: np.dtype("f4"),
    3: np.dtype("i2"),      # complex, stored as (real, imag) int16 pairs
    4: np.dtype("c8"),      # complex float32
    6: np.dtype("u2"),
    12: np.dtype("f2"),     # half float, added in MRC2014 for detectors
}

_COMPLEX_MODES = frozenset({3, 4})
_MODE_NAMES = {
    0: "int8", 1: "int16", 2: "float32", 3: "complex int16",
    4: "complex float32", 6: "uint16", 12: "float16", 101: "4-bit packed",
}


class MrcError(Exception):
    """Raised when a file is not valid MRC, or uses a mode we cannot read."""


class MrcStream:
    """Lazy reader over one MRC / CCP4 map file.

    Nothing but the 1024-byte header is read on open. ``asarray()`` pulls
    the whole volume; ``plane(i)`` pulls one z-section, which is the
    access pattern that matters for tomograms and movie stacks too large
    to hold at once.
    """

    def __init__(self, src: Any, *,
                 read_at: Callable[[int, int], bytes] | None = None):
        self._src = src
        if read_at is None and callable(src) and not isinstance(
                src, (str, os.PathLike, bytes, bytearray, memoryview)):
            read_at = src
            self._src = None
        if read_at is not None:
            self._read = read_at
            self._owns_fd = False
            self._fh = None
        else:
            self._read, self._owns_fd, self._fh = self._open_read_at(src)

        head = self._read(0, HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            raise MrcError(
                f"MRC: file is {len(head)} bytes, shorter than the "
                f"{HEADER_SIZE}-byte header")
        self._header = self._parse_header(head)

    # -- header ------------------------------------------------------

    @staticmethod
    def _byteorder(head: bytes) -> str:
        """Endianness, from the machine stamp, with a sanity fallback.

        MRC2014 puts 0x44 0x44 at bytes 212-213 for a little-endian
        writer and 0x11 0x11 for big-endian. Plenty of older files leave
        the stamp zeroed, so fall back to whichever byte order makes NX
        look like a plausible dimension rather than a number in the
        billions.
        """
        stamp = head[212:214]
        if stamp in (b"\x44\x44", b"\x44\x41"):
            return "<"
        if stamp == b"\x11\x11":
            return ">"
        nx_le = struct.unpack_from("<i", head, 0)[0]
        nx_be = struct.unpack_from(">i", head, 0)[0]
        if 0 < nx_le <= 1 << 20:
            return "<"
        if 0 < nx_be <= 1 << 20:
            return ">"
        raise MrcError(
            "MRC: cannot determine byte order; machine stamp is "
            f"{stamp!r} and neither reading of NX is plausible "
            f"({nx_le} / {nx_be})")

    def _parse_header(self, head: bytes) -> dict[str, Any]:
        # "MAP " at bytes 208-211 is the format identifier. It is absent
        # from some pre-2014 files, so its absence is not fatal on its
        # own; the dimension and mode checks below carry the weight.
        bo = self._byteorder(head)
        i = lambda off: struct.unpack_from(bo + "i", head, off)[0]  # noqa: E731
        f = lambda off: struct.unpack_from(bo + "f", head, off)[0]  # noqa: E731

        nx, ny, nz, mode = i(0), i(4), i(8), i(12)
        if min(nx, ny, nz) < 0:
            raise MrcError(f"MRC: negative dimension in header "
                           f"({nx}, {ny}, {nz})")
        if mode not in _MODES:
            name = _MODE_NAMES.get(mode, "unknown")
            raise MrcError(f"MRC: unsupported MODE {mode} ({name})")

        nsymbt = i(92)
        if nsymbt < 0:
            raise MrcError(f"MRC: negative extended header size {nsymbt}")

        return {
            "byteorder": bo,
            "nx": nx, "ny": ny, "nz": nz,
            "mode": mode,
            "nxstart": i(16), "nystart": i(20), "nzstart": i(24),
            "mx": i(28), "my": i(32), "mz": i(36),
            "cella": (f(40), f(44), f(48)),
            "cellb": (f(52), f(56), f(60)),
            "mapc": i(64), "mapr": i(68), "maps": i(72),
            "dmin": f(76), "dmax": f(80), "dmean": f(84),
            "ispg": i(88),
            "nsymbt": nsymbt,
            "exttyp": head[104:108].decode("ascii", "replace").strip(),
            "nversion": i(108),
            "origin": (f(196), f(200), f(204)),
            "map": head[208:212].decode("ascii", "replace"),
            "machst": head[212:216],
            "rms": f(216),
            "nlabl": i(220),
        }

    @property
    def header(self) -> dict[str, Any]:
        return dict(self._header)

    @property
    def is_mrcz(self) -> bool:
        """MRCZ stores blosc-compressed blocks and is not plain MRC.

        It reuses the header but the voxels are compressed, so a reader
        that trusts the size arithmetic below would return noise. We
        detect it and refuse rather than guess.
        """
        return self._header["exttyp"].upper().startswith("MRCZ")

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Angstroms per voxel, from the cell dimensions over the grid.

        This is the number the rest of a cryo-EM pipeline cares about,
        and it is derived rather than stored: CELLA divided by the grid
        sampling MX/MY/MZ.
        """
        h = self._header
        cella, grid = h["cella"], (h["mx"], h["my"], h["mz"])
        return tuple(float(c) / g if g else 0.0
                     for c, g in zip(cella, grid))          # type: ignore

    @property
    def axis_order(self) -> tuple[int, int, int]:
        """Crystallographic axis of each stored axis, slowest first.

        ``(maps, mapr, mapc)``: which of x=1, y=2, z=3 the sections,
        rows and columns correspond to. The common value is ``(3, 2, 1)``
        so that the array is already (z, y, x), but it is not guaranteed
        and real deposits do permute: EMD-3001 stores ``(2, 1, 3)``.
        """
        h = self._header
        return (h["maps"], h["mapr"], h["mapc"])

    @property
    def is_canonical(self) -> bool:
        """True when the stored order is already (z, y, x)."""
        return self.axis_order == (3, 2, 1)

    def _canonical_permutation(self) -> tuple[int, int, int] | None:
        """Transpose that takes the stored order to (z, y, x), if valid."""
        order = self.axis_order
        if sorted(order) != [1, 2, 3]:
            return None                    # header is not self-consistent
        if order == (3, 2, 1):
            return None                    # already canonical
        return tuple(order.index(axis) for axis in (3, 2, 1))   # type: ignore

    # -- data --------------------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        h = self._header
        return _MODES[h["mode"]].newbyteorder(h["byteorder"])

    @property
    def shape(self) -> tuple[int, ...]:
        """``(nz, ny, nx)``, or ``(ny, nx)`` when there is one section.

        Numpy convention: slowest axis first. A single-section map is
        returned 2-D because that is what callers of an image reader
        expect, and nz == 1 is how MRC spells "this is an image".
        """
        h = self._header
        if h["nz"] == 1:
            return (h["ny"], h["nx"])
        return (h["nz"], h["ny"], h["nx"])

    @property
    def n_planes(self) -> int:
        return max(1, self._header["nz"])

    @property
    def data_offset(self) -> int:
        return HEADER_SIZE + self._header["nsymbt"]

    def _plane_bytes(self) -> int:
        h = self._header
        n = h["ny"] * h["nx"] * self.dtype.itemsize
        return n * 2 if h["mode"] == 3 else n

    def extended_header(self) -> bytes:
        """Raw extended header. FEI/Thermo store per-frame metadata here."""
        n = self._header["nsymbt"]
        return self._read(HEADER_SIZE, n) if n else b""

    def plane(self, index: int) -> np.ndarray:
        """Read one z-section as ``(ny, nx)``."""
        if self.is_mrcz:
            raise MrcError(
                "MRC: this file is MRCZ (blosc-compressed voxels); "
                "plain MRC reading would return noise")
        n = self.n_planes
        if not 0 <= index < n:
            raise IndexError(f"plane {index} out of range (0..{n - 1})")
        h = self._header
        size = self._plane_bytes()
        raw = self._read(self.data_offset + index * size, size)
        if len(raw) < size:
            raise MrcError(
                f"MRC: truncated file; plane {index} needs {size} bytes, "
                f"got {len(raw)}")
        arr = np.frombuffer(raw, dtype=self.dtype)
        if h["mode"] == 3:
            arr = arr.reshape(h["ny"], h["nx"], 2)
        else:
            arr = arr.reshape(h["ny"], h["nx"])
        return arr

    def asarray(self, *, out: np.ndarray | None = None,
                canonical: bool = False) -> np.ndarray:
        """Read the whole map.

        Reads in one call rather than per plane: the data is contiguous,
        so a single large read is one syscall instead of nz of them, and
        over an HTTP data source it is one request instead of nz.

        ``canonical=True`` reorients to (z, y, x) using MAPC/MAPR/MAPS
        instead of returning the stored order. Off by default because
        returning what the file contains is the honest default and is
        what every other MRC reader does, but a caller who assumes
        (z, y, x) on a permuted map like EMD-3001 gets a transposed
        volume with no error, which is the kind of bug that survives to
        publication.
        """
        if self.is_mrcz:
            raise MrcError(
                "MRC: this file is MRCZ (blosc-compressed voxels); "
                "plain MRC reading would return noise")
        h = self._header
        total = self._plane_bytes() * self.n_planes
        raw = self._read(self.data_offset, total)
        if len(raw) < total:
            raise MrcError(
                f"MRC: truncated file; data needs {total} bytes, "
                f"got {len(raw)}")
        arr = np.frombuffer(raw, dtype=self.dtype)
        if h["mode"] == 3:
            arr = arr.reshape(*self.shape, 2)
        else:
            arr = arr.reshape(self.shape)
        if canonical:
            perm = self._canonical_permutation()
            if perm is not None:
                if h["mode"] == 3:
                    perm = perm + (3,)     # keep the complex pair trailing
                arr = np.ascontiguousarray(np.transpose(arr, perm))
        if out is not None:
            if out.shape != arr.shape:
                raise ValueError(
                    f"out has shape {out.shape}, expected {arr.shape}")
            out[...] = arr
            return out
        return arr

    # -- plumbing ----------------------------------------------------

    def _open_read_at(self, src: Any):
        if isinstance(src, (str, os.PathLike)):
            fh = open(src, "rb")

            def read_at(off: int, n: int, _f=fh) -> bytes:
                _f.seek(off)
                return _f.read(n)
            return read_at, True, fh
        if isinstance(src, (bytes, bytearray, memoryview)):
            buf = bytes(src)

            def read_at(off: int, n: int, _b=buf) -> bytes:
                return _b[off:off + n]
            return read_at, False, None
        if hasattr(src, "read") and hasattr(src, "seek"):
            fh = src

            def read_at(off: int, n: int, _f=fh) -> bytes:
                _f.seek(off)
                return _f.read(n)
            return read_at, False, None
        raise TypeError(
            f"MRC: unsupported src type {type(src).__name__}; pass a path, "
            f"bytes, file-like, or a read_at callable")

    def close(self) -> None:
        if self._owns_fd and self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "MrcStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["MrcStream", "MrcError", "HEADER_SIZE"]
