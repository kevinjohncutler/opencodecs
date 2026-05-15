"""Minimal OLE2 / Compound File Binary reader.

Read-only parser for the Microsoft Compound File Binary container
format (MS-CFB). Backs OibCodec's native path so opencodecs can open
Olympus OIB files without depending on the ``olefile`` package.

Implements just enough of MS-CFB v3 / v4 to:
  * Parse the 512-byte file header
  * Walk the DIFAT / FAT to find every regular-sector chain
  * Walk the MiniFAT for streams smaller than the mini-cutoff
  * Decode the directory tree (storage + stream entries)
  * Read any named stream as ``bytes``

Not implemented: write support, transactions, encrypted streams.
That's an order of magnitude more code and we don't need it.

Reads via a :class:`~opencodecs.core.io.DataSource` so the same code
path works on local files, mmap, and ``HTTPDataSource`` — meaning
OIB on cloud storage gets range-read partial decode for free.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.io import DataSource, coerce_data_source


_OLE_MAGIC = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"

# Special FAT sector values
_FREESECT  = 0xFFFFFFFF
_ENDOFCHAIN = 0xFFFFFFFE
_FATSECT    = 0xFFFFFFFD
_DIFSECT    = 0xFFFFFFFC

# Directory entry types (byte at offset 0x42 in each 128-byte entry)
_OBJ_INVALID = 0
_OBJ_STORAGE = 1
_OBJ_STREAM  = 2
_OBJ_ROOT    = 5

# Directory entries are always 128 bytes
_DIRSIZE = 128


@dataclass
class OleDirEntry:
    """One entry in the OLE2 directory (storage, stream, or root)."""
    name: str
    obj_type: int   # 1=storage, 2=stream, 5=root
    left_id: int    # sibling tree (red-black)
    right_id: int
    child_id: int   # for storages, points to its first child
    start_sect: int   # for streams + root: first sector of data
    size: int          # stream size in bytes (root: total mini-stream size)


class OleReader:
    """Random-access reader for an OLE2 compound file.

    Construction reads only the 512-byte header + DIFAT + FAT + Directory
    + MiniFAT — typically a few KB even for multi-megabyte OIBs. Each
    stream's contents are decoded on demand via :meth:`read_stream`.
    """

    def __init__(self, src: Any):
        # OLE2 doesn't need the size up front (header is at offset 0),
        # but the shared helper does the path-vs-DataSource coercion
        # and the lifecycle bookkeeping in one place.
        self._src, self._owns_src, _ = coerce_data_source(src)
        self._parse_header()
        self._read_fat()
        self._read_minifat()
        self._read_directory()
        # The mini-stream is stored as a regular stream pointed to by
        # the root directory entry (entry 0). We pre-read it here so
        # mini-FAT-resident streams can be served quickly.
        root = self._directory[0]
        if root.size > 0:
            self._mini_stream = self._read_chain(root.start_sect, root.size)
        else:
            self._mini_stream = b""

    # ----- header / FAT / directory -----

    def _parse_header(self) -> None:
        header = self._src.read_at(0, 512)
        if header[:8] != _OLE_MAGIC:
            raise ValueError(
                f"OLE2: bad magic at offset 0: {header[:8].hex()}")
        (
            self._minor_version,
            self._major_version,
            byte_order,
            sector_shift,
            mini_sector_shift,
        ) = struct.unpack_from("<HHHHH", header, 24)
        if byte_order != 0xFFFE:
            raise ValueError(
                f"OLE2: unsupported byte order 0x{byte_order:04X} "
                f"(expected little-endian 0xFFFE)")
        self._sector_size = 1 << sector_shift
        self._mini_sector_size = 1 << mini_sector_shift
        # Per MS-CFB §2.2:
        #   0x28 (40): n_dir_sectors    — 0 for v3, real count for v4
        #   0x2C (44): n_fat_sectors
        #   0x30 (48): first_dir_sect
        #   0x34 (52): transaction sig
        #   0x38 (56): mini_cutoff      — typically 0x1000
        #   0x3C (60): first_mini_fat
        #   0x40 (64): n_mini_fat
        #   0x44 (68): first_difat
        #   0x48 (72): n_difat
        #   0x4C (76): DIFAT[0..108]    — 109 × u32 inline in the header
        (
            self._n_dir_sectors,
            self._n_fat_sectors,
            self._first_dir_sect,
            self._tx_sig,
            self._mini_cutoff,
            self._first_mini_fat,
            self._n_mini_fat,
            self._first_difat,
            self._n_difat,
        ) = struct.unpack_from("<IIIIIIIII", header, 40)
        self._header_difat = struct.unpack_from("<109I", header, 76)

    def _sector_offset(self, sector: int) -> int:
        """Byte offset of a regular sector in the file. The 512-byte
        header is padded to a full ``sector_size`` boundary before
        sectors begin — so sector 0 starts at offset ``sector_size``
        for v4 (sector_size=4096), and at offset 512 for v3
        (sector_size=512). The unified formula:
        ``offset = (sector + 1) * sector_size`` when sector_size >= 512.
        """
        return self._sector_size + sector * self._sector_size

    def _read_sector(self, sector: int) -> bytes:
        return self._src.read_at(
            self._sector_offset(sector), self._sector_size)

    def _read_chain(self, start: int, length: int) -> bytes:
        """Walk the FAT from ``start`` collecting up to ``length``
        bytes. Returns ``length`` bytes (truncated chains are an
        OLE2-corrupt error).

        Coalesces contiguous runs of sectors into a single
        ``read_at(offset, run_size)`` call — important for HTTP
        DataSources where each call is an HTTP round trip.

        Detects cycles in the FAT and caps the iteration count at
        the number of FAT entries — guards against malformed files
        that would otherwise loop forever.
        """
        if length == 0 or start == _ENDOFCHAIN:
            return b""
        # First pass: walk the chain to get the ordered list of
        # sector numbers covering `length` bytes. No I/O yet.
        sectors: list[int] = []
        remaining = length
        sect = start
        seen: set[int] = set()
        max_iter = len(self._fat) + 1
        for _ in range(max_iter):
            if remaining <= 0 or sect == _ENDOFCHAIN:
                break
            if sect in seen:
                raise ValueError(
                    f"OLE2: FAT cycle detected at sector {sect}")
            seen.add(sect)
            if sect >= len(self._fat):
                raise ValueError(
                    f"OLE2: FAT chain reaches sector {sect} but FAT "
                    f"has only {len(self._fat)} entries")
            sectors.append(sect)
            remaining -= self._sector_size
            sect = self._fat[sect]
        else:
            raise ValueError(
                f"OLE2: FAT chain exceeded iteration limit "
                f"({max_iter}); likely corrupt")
        if not sectors:
            return b""
        # Coalesce runs of consecutive sector numbers, one I/O each.
        chunks: list[bytes] = []
        run_start = sectors[0]
        run_count = 1
        for s in sectors[1:]:
            if s == run_start + run_count:
                run_count += 1
            else:
                chunks.append(self._src.read_at(
                    self._sector_offset(run_start),
                    run_count * self._sector_size))
                run_start = s
                run_count = 1
        chunks.append(self._src.read_at(
            self._sector_offset(run_start),
            run_count * self._sector_size))
        data = b"".join(chunks)
        return data[:length]

    def _read_fat(self) -> None:
        """Read the regular-stream FAT by following DIFAT.

        The first 109 FAT sectors are listed in the header. If
        ``n_difat > 0``, additional DIFAT sectors chain from
        ``first_difat`` — each contains 127 FAT-sector pointers + 1
        next-DIFAT pointer.
        """
        fat_sectors: list[int] = []
        for s in self._header_difat:
            if s != _FREESECT:
                fat_sectors.append(s)
            if len(fat_sectors) >= self._n_fat_sectors:
                break
        # Walk extra DIFAT sectors if needed
        difat = self._first_difat
        per_difat = (self._sector_size // 4) - 1
        for _ in range(self._n_difat):
            if difat == _ENDOFCHAIN:
                break
            buf = self._read_sector(difat)
            entries = struct.unpack_from(
                f"<{per_difat}I", buf, 0)
            for s in entries:
                if s != _FREESECT:
                    fat_sectors.append(s)
                if len(fat_sectors) >= self._n_fat_sectors:
                    break
            difat = struct.unpack_from("<I", buf, per_difat * 4)[0]
        # Concatenate the actual FAT sectors → flat fat[i] table
        fat: list[int] = []
        per_sector = self._sector_size // 4
        for s in fat_sectors:
            buf = self._read_sector(s)
            fat.extend(struct.unpack_from(f"<{per_sector}I", buf, 0))
        self._fat = fat

    def _read_minifat(self) -> None:
        """Read the mini-FAT (for streams < mini_cutoff bytes)."""
        if self._first_mini_fat == _ENDOFCHAIN or self._n_mini_fat == 0:
            self._minifat = []
            return
        # The mini-FAT is itself a chain in the regular FAT.
        mini_fat_bytes = self._read_chain(
            self._first_mini_fat,
            self._n_mini_fat * self._sector_size,
        )
        n_entries = len(mini_fat_bytes) // 4
        self._minifat = list(
            struct.unpack_from(f"<{n_entries}I", mini_fat_bytes, 0))

    def _read_directory(self) -> None:
        """Read every directory entry. Entries are 128 bytes each."""
        # The directory is a regular-FAT chain starting at
        # _first_dir_sect. The chain's total length isn't recorded
        # explicitly, but we know each sector holds sector_size / 128
        # entries — walk the chain until it ends.
        chunks: list[bytes] = []
        sect = self._first_dir_sect
        seen: set[int] = set()
        max_iter = len(self._fat) + 1
        for _ in range(max_iter):
            if sect == _ENDOFCHAIN:
                break
            if sect in seen or sect >= len(self._fat):
                break
            seen.add(sect)
            chunks.append(self._read_sector(sect))
            sect = self._fat[sect]
        else:
            raise ValueError(
                f"OLE2: directory FAT chain exceeded iteration limit "
                f"({max_iter}); likely corrupt")
        dir_bytes = b"".join(chunks)
        n_entries = len(dir_bytes) // _DIRSIZE
        entries: list[OleDirEntry] = []
        for i in range(n_entries):
            entry = dir_bytes[i * _DIRSIZE:(i + 1) * _DIRSIZE]
            name_len = struct.unpack_from("<H", entry, 0x40)[0]
            obj_type = entry[0x42]
            if obj_type == _OBJ_INVALID:
                entries.append(OleDirEntry(
                    name="", obj_type=0, left_id=-1, right_id=-1,
                    child_id=-1, start_sect=0, size=0))
                continue
            # Name is 64 bytes of UTF-16-LE; name_len includes the
            # trailing null and is given in BYTES.
            if name_len >= 2:
                name = entry[:name_len - 2].decode(
                    "utf-16-le", errors="replace")
            else:
                name = ""
            left = struct.unpack_from("<i", entry, 0x44)[0]
            right = struct.unpack_from("<i", entry, 0x48)[0]
            child = struct.unpack_from("<i", entry, 0x4C)[0]
            start_sect = struct.unpack_from("<I", entry, 0x74)[0]
            # Stream size: 8 bytes at 0x78. For v3 only the low 4
            # bytes are valid (high 4 = 0); for v4 it's a full u64.
            size_lo, size_hi = struct.unpack_from("<II", entry, 0x78)
            size = size_lo | (size_hi << 32)
            entries.append(OleDirEntry(
                name=name, obj_type=obj_type, left_id=left,
                right_id=right, child_id=child,
                start_sect=start_sect, size=size,
            ))
        self._directory = entries

    # ----- public read API -----

    def list_streams(self) -> list[str]:
        """Return the names of every stream (leaf) in the directory.

        Storage hierarchy is flattened — storage paths are NOT
        encoded into the returned names. OIB files store everything
        under the root storage, so this works fine.
        """
        return [
            e.name for e in self._directory
            if e.obj_type == _OBJ_STREAM and e.name
        ]

    def get_size(self, name: str) -> int:
        for e in self._directory:
            if e.obj_type == _OBJ_STREAM and e.name == name:
                return e.size
        raise KeyError(f"OLE2: no stream named {name!r}")

    def read_stream(self, name: str, length: int | None = None) -> bytes:
        """Read the named stream's contents as bytes.

        ``length=None`` (default) reads the entire stream.
        ``length=N`` reads only the first N bytes — useful for
        peeking at headers (TIFF magic + first IFD) without
        fetching the whole stream over HTTP.
        """
        for e in self._directory:
            if e.obj_type == _OBJ_STREAM and e.name == name:
                n = e.size if length is None else min(length, e.size)
                if e.size < self._mini_cutoff:
                    return self._read_mini_chain(e.start_sect, n)
                return self._read_chain(e.start_sect, n)
        raise KeyError(f"OLE2: no stream named {name!r}")

    def _read_mini_chain(self, start: int, length: int) -> bytes:
        """Walk the mini-FAT for small streams that live in the
        mini-stream rather than the regular sector pool."""
        if length == 0:
            return b""
        chunks: list[bytes] = []
        remaining = length
        sect = start
        while remaining > 0 and sect != _ENDOFCHAIN:
            offset = sect * self._mini_sector_size
            chunk = self._mini_stream[offset:offset + self._mini_sector_size]
            take = min(self._mini_sector_size, remaining)
            chunks.append(chunk[:take])
            remaining -= take
            if sect >= len(self._minifat):
                raise ValueError(
                    f"OLE2: mini-FAT chain reaches sector {sect} but "
                    f"mini-FAT has only {len(self._minifat)} entries")
            sect = self._minifat[sect]
        if remaining > 0:
            raise ValueError(
                f"OLE2: mini-FAT chain ended {remaining} bytes short")
        return b"".join(chunks)

    def close(self) -> None:
        if self._owns_src and hasattr(self._src, "close"):
            self._src.close()

    def __enter__(self) -> "OleReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


__all__ = ["OleReader", "OleDirEntry"]
