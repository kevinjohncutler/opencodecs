"""VSI (Olympus CellSens) codec — TIFF-backed index reader.

VSI is a multi-file format produced by Olympus CellSens / Evident
microscope software. The top-level ``foo.vsi`` is a **TIFF**
(``II*\\0`` magic) containing:

  * A thumbnail / overview image (typically a 256x256 RGB jpg-in-TIFF)
  * Olympus-specific metadata in private IFD tags
  * Pointers into the sibling ``_foo_/stack<N>/frame_t.ets`` directory
    that holds the full-resolution pyramid data

Our native TIFF reader already handles the top-level ``.vsi`` file
end-to-end — it returns the thumbnail and exposes the IFD tags.
What we DON'T yet have is the ``.ets`` parser for full-resolution
tile data. That's a future native upgrade.

This codec wires VSI into the registry so:

  * ``oc.read("foo.vsi")`` returns the thumbnail (uses TIFF reader)
  * ``oc.get_codec("vsi").open(...)`` returns a TiffStream
  * ``codec.signature(head)`` detects VSI by extension hint + TIFF magic
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Codec, Reader


class VsiCodec(Codec):
    """Olympus VSI (CellSens virtual slide) — delegates to TIFF.

    The top-level VSI is a TIFF; full-resolution data lives in a
    sibling ``_NAME_/stackN/frame_t.ets`` companion tree we don't
    yet read natively. For typical "what's in this slide?" use cases
    (thumbnail + metadata) this still works.
    """

    name = "vsi"
    file_extensions = (".vsi",)
    aliases = ()

    has_native = True   # TIFF reader handles the top-level container
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.uint16, np.float32,
    )
    supports_color = True

    def signature(self, head: bytes) -> bool:
        """VSI files start with the standard TIFF magic. There's no
        VSI-specific magic in the header — the format is detected by
        the .vsi extension; we accept any TIFF-magic bytes here so
        codec_for_bytes() can still route a .vsi blob correctly."""
        return len(head) >= 4 and head[:4] in (b"II*\x00", b"MM\x00*")

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read()

    def open(self, src: Any, *, backend: str = "auto",
             mode: str | None = None, **opts) -> Reader:
        """Open a VSI for reading.

        ``backend`` (named ``mode`` in older callers — see compat note
        below):
          * ``"auto"`` (default): if a sibling ``_NAME_/stackN/frame_t.ets``
            tree exists, decode the full-resolution stack natively.
            Otherwise (or for the second IFD of the index), return
            the TIFF thumbnail reader.
          * ``"thumbnail"``: always return the TIFF thumbnail.
          * ``"ets"``: always decode the .ets stack(s); fails when
            no companion is present.

        Both ``backend`` and ``mode`` accept the same three values;
        ``backend`` is the cross-codec kwarg name (matches
        ``Nd2Codec.open`` / ``LifCodec.open`` / ``OibCodec.open``) and
        is the documented form. ``mode`` is kept as a deprecated alias
        for back-compat with the earlier VSI-only API.
        """
        if mode is not None:
            backend = mode
        from ._tiff_codec import TiffStream
        if backend == "thumbnail":
            return TiffStream(src, **opts)
        if not isinstance(src, (str, Path)):
            # Bytes / file-like don't have a sibling .ets path
            return TiffStream(src, **opts)
        # Look for a companion _NAME_/stack*/frame_t_*.ets tree
        p = Path(src)
        companion = p.parent / f"_{p.stem}_"
        ets_files = []
        if companion.is_dir():
            for sd in sorted(companion.iterdir()):
                if sd.is_dir():
                    ets_files.extend(sorted(sd.glob("frame_t_*.ets")))
        if not ets_files:
            if backend == "ets":
                raise FileNotFoundError(
                    f"VSI: no .ets companion found at {companion}")
            return TiffStream(src, **opts)
        # ETS mode (default when companion present)
        return VsiNativeReader(p, ets_files)

    def info(self, src: Any) -> dict:
        """Partial-parse the VSI index + every .ets companion in the
        sibling ``_NAME_/stackN/`` tree. Returns geometry info
        without decoding pixel data. Useful for inspecting what's
        in a VSI before committing to a full decode."""
        from pathlib import Path
        from ._ets import parse_ets
        p = Path(src)
        out: dict = {"vsi_path": str(p)}
        # Top-level TIFF
        from ._tiff_codec import TiffStream
        try:
            with TiffStream(str(p)) as stream:
                page0 = stream.page(0)
                out["index_shape"] = page0.shape
                out["index_dtype"] = str(page0.dtype)
        except Exception as e:
            out["index_error"] = str(e)
        # Sibling _NAME_/stackN/frame_t_*.ets
        companion = p.parent / f"_{p.stem}_"
        stacks = []
        if companion.is_dir():
            for sd in sorted(companion.iterdir()):
                if not sd.is_dir():
                    continue
                for ets_path in sorted(sd.glob("frame_t_*.ets")):
                    info = parse_ets(ets_path)
                    stacks.append({
                        "stack": sd.name,
                        "path": str(ets_path),
                        "file_size": info.file_size,
                        "width": info.width,
                        "height": info.height,
                        "n_components": info.n_components,
                        "level_count": info.level_count,
                        "magic_ok": info.magic_ok,
                    })
        out["ets_stacks"] = stacks
        return out


class VsiNativeReader(Reader):
    """Native VSI reader: lazily decodes planes from each
    ``frame_t_N.ets`` companion. Per-plane reads pass through the
    DataSource abstraction, so HTTP-backed VSI sources only fetch
    the planes the caller actually iterates."""

    def __init__(self, vsi_path: Path, ets_files: list[Path]):
        from ._ets import parse_ets, decode_ets_plane, decode_ets
        self._vsi_path = vsi_path
        self._ets_files = list(ets_files)
        self._decode_ets = decode_ets
        self._decode_plane = decode_ets_plane
        # Probe the first .ets to learn geometry; assume all match.
        first_info = parse_ets(str(ets_files[0]))
        self._plane_height = int(first_info.height)
        self._plane_width = int(first_info.width)
        self._info_per_file = [first_info]
        # Per-file plane count is derived from the payload region.
        plane_bytes = first_info.height * first_info.width * 2
        first_planes = (first_info.sub_chunk_offsets[1] - 292) // plane_bytes
        self._planes_per_file = [first_planes]
        for ets_path in ets_files[1:]:
            inf = parse_ets(str(ets_path))
            self._info_per_file.append(inf)
            self._planes_per_file.append(
                (inf.sub_chunk_offsets[1] - 292) // plane_bytes)
        self.n_frames = sum(self._planes_per_file)
        self.shape = (
            self.n_frames, self._plane_height, self._plane_width)
        self.dtype = np.dtype("<u2")
        self.is_chunked = False

    def _locate(self, frame_index: int) -> tuple[int, int]:
        """Map a global frame index → (file_index, plane_in_file)."""
        f = 0
        i = frame_index
        for cnt in self._planes_per_file:
            if i < cnt:
                return f, i
            i -= cnt
            f += 1
        raise IndexError(
            f"VSI: frame {frame_index} out of range [0, {self.n_frames})")

    def __getitem__(self, idx) -> np.ndarray:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(
                "VsiNativeReader: only int frame indexing supported")
        f, p = self._locate(int(idx))
        return self._decode_plane(str(self._ets_files[f]), p)

    def iter_frames(self) -> Iterator[np.ndarray]:
        for ets_path in self._ets_files:
            stack = self._decode_ets(str(ets_path))
            for j in range(stack.shape[0]):
                yield stack[j]

    def read(self) -> np.ndarray:
        chunks = [self._decode_ets(str(p)) for p in self._ets_files]
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        self._info_per_file = []

    def __enter__(self) -> "VsiNativeReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


__all__ = ["VsiCodec", "VsiNativeReader"]
