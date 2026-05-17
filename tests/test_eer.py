"""EER (Electron Event Representation) decoder tests.

The reference test vector ``b'\\x03\\x1b\\xfc\\xb1\\x35\\xfb'`` is taken
straight from the EER format specification and matches imagecodecs'
own test cases. Event positions are pre-computed in the spec, so we
assert against absolute expected values rather than cross-validate.

Additional fuzz cross-validation against imagecodecs.eer_decode (when
present) covers parameter combinations on random bitstreams.
"""

from __future__ import annotations

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._eer")
decode = mod.decode
EerError = mod.EerError

# Test vector from the EER specification
SPEC_ENCODED = b"\x03\x1b\xfc\xb1\x35\xfb"


def test_eer_spec_linear():
    """Linear 1x312 frame; expected event positions: 3, 17, 233, 311."""
    im = decode(SPEC_ENCODED, (1, 312), 7, 1, 1)
    hits = np.where(im[0])[0].tolist()
    assert hits == [3, 17, 233, 311]


def test_eer_spec_2d_grid():
    """Same events folded into 20x16."""
    im = decode(SPEC_ENCODED, (20, 16), 7, 1, 1)
    assert im[0, 3]
    assert im[1, 1]
    assert im[14, 9]
    assert im[19, 7]
    assert im.sum() == 4


def test_eer_spec_superres():
    """Super-resolution mode upsamples to 40x32 with sub-pixel hits."""
    im = decode(SPEC_ENCODED, (40, 32), 7, 1, 1, superres=1)
    assert im[0, 7]
    assert im[3, 3]
    assert im[29, 18]
    assert im[39, 14]
    assert im.sum() == 4


def test_eer_uint16_accumulator():
    """Passing a uint16 ``out`` accumulates instead of overwriting."""
    out = np.ones((40, 32), np.uint16)
    decode(SPEC_ENCODED, (40, 32), 7, 1, 1, superres=1, out=out)
    # Each of the four hits adds 1 to the pre-existing 1.
    assert int(out[0, 7]) == 2
    assert int(out[3, 3]) == 2
    assert int(out[29, 18]) == 2
    assert int(out[39, 14]) == 2
    # Background should be the original 1.
    assert int(out.sum()) == (40 * 32) + 4


def test_eer_rejects_shape_too_small():
    """Output shape too small to hold all events -> EerError."""
    with pytest.raises(EerError):
        decode(SPEC_ENCODED, (19, 15), 7, 1, 1)


def test_eer_superres_requires_compatible_shape():
    """In super-resolution mode the output shape must be divisible by
    the super-resolution factor."""
    with pytest.raises(EerError):
        decode(SPEC_ENCODED, (40, 33), 7, 1, 1, superres=1)


def test_eer_rejects_invalid_params():
    with pytest.raises(EerError):
        # skipbits=0 invalid
        decode(SPEC_ENCODED, (16, 16), 0, 1, 1)
    with pytest.raises(EerError):
        # horzbits=0 invalid
        decode(SPEC_ENCODED, (16, 16), 7, 0, 1)


def test_eer_imagecodecs_cross_validate():
    """Random bitstreams must decode identically to imagecodecs.

    EER's "skip" field can advance past the end of small canvases on
    random input — we use a 1024x1024 canvas so most parameter combos
    decode cleanly. When *either* implementation errors we just
    cross-check that *both* implementations error on the same input.
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    if not getattr(imagecodecs, "EER", None) or not imagecodecs.EER.available:
        pytest.skip("imagecodecs EER backend unavailable")

    rng = np.random.default_rng(42)
    data = rng.bytes(4096)
    matched = 0
    for sb in (7, 8, 10):
        for hb in (2, 3):
            for vb in (2, 3):
                if not (8 < sb + hb + vb < 17):
                    continue
                for sr in (0, 1, 2):
                    shape = (1024, 1024)
                    try:
                        ours = decode(data, shape, sb, hb, vb, superres=sr)
                    except EerError:
                        with pytest.raises(Exception):
                            imagecodecs.eer_decode(
                                data, shape, sb, hb, vb, superres=sr
                            )
                        continue
                    theirs = imagecodecs.eer_decode(
                        data, shape, sb, hb, vb, superres=sr
                    )
                    np.testing.assert_array_equal(
                        ours, theirs,
                        err_msg=(
                            f"divergence at sb={sb} hb={hb} vb={vb} sr={sr}"
                        ),
                    )
                    matched += 1
    assert matched > 0, "no parameter combo decoded cleanly"


def test_eer_in_tiff_dispatch_via_tiffstream():
    """A synthetic EER-in-TIFF file with compression=65002 + private
    tags 65007/8/9 must decode through our TiffStream reader's EER
    compression-tag dispatch (no need for tifffile)."""
    import struct
    from opencodecs._tiff_codec import TiffStream

    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    shape = (20, 16)
    expected = decode(encoded, shape, 7, 1, 1)

    # Hand-roll a minimal classic-TIFF file with the EER strip + tags.
    bo = "<"
    out = bytearray()
    out += b"II"
    out += struct.pack(bo + "H", 42)
    out += struct.pack(bo + "I", 0)   # patched below
    pixel_off = 8
    out += encoded
    if len(out) % 2:
        out += b"\x00"
    ifd_start = len(out)

    entries = [
        (256, 4, 1, shape[1]),     # ImageWidth
        (257, 4, 1, shape[0]),     # ImageLength
        (258, 3, 1, 8),            # BitsPerSample
        (259, 3, 1, 65002),        # Compression = EER v2
        (262, 3, 1, 1),            # Photometric
        (273, 4, 1, pixel_off),    # StripOffsets
        (277, 3, 1, 1),            # SamplesPerPixel
        (278, 4, 1, shape[0]),     # RowsPerStrip
        (279, 4, 1, len(encoded)), # StripByteCounts
        (65007, 3, 1, 7),          # EER SKIPBITS
        (65008, 3, 1, 1),          # EER HORZBITS
        (65009, 3, 1, 1),          # EER VERTBITS
    ]
    entries.sort(key=lambda e: e[0])
    out[4:8] = struct.pack(bo + "I", ifd_start)
    out += struct.pack(bo + "H", len(entries))
    for tag, tc, count, value in entries:
        out += struct.pack(bo + "HHI", tag, tc, count)
        out += (struct.pack(bo + "HH", value, 0) if tc == 3
                else struct.pack(bo + "I", value))
    out += struct.pack(bo + "I", 0)   # next IFD = 0

    with TiffStream(bytes(out)) as r:
        page = r.page(0)
        assert page.compression == 65002
        arr = page.asarray()
    np.testing.assert_array_equal(arr, expected)


# ---------------------------------------------------------------------------
# EerReader (file-level wrapper) tests
# ---------------------------------------------------------------------------


def _build_multi_frame_eer_tiff(
    frames: list[bytes], shape: tuple[int, int],
    skipbits: int = 7, horzbits: int = 1, vertbits: int = 1,
) -> bytes:
    """Hand-roll a multi-page TIFF file where each page is one EER
    frame. Used by the EerReader tests to avoid bringing in a real
    EER acquisition just to exercise the wrapper."""
    import struct
    bo = "<"
    out = bytearray()
    out += b"II"
    out += struct.pack(bo + "H", 42)
    out += struct.pack(bo + "I", 0)   # patched below to point at IFD 0

    # Write strip payloads first; record their offsets.
    strip_offsets = []
    for blob in frames:
        if len(out) % 2:
            out += b"\x00"
        strip_offsets.append(len(out))
        out += blob

    # Write one IFD per frame, chained by NextIFDOffset.
    first_ifd_offset = None
    prev_next_ifd_field = 4   # patches the header's first-IFD pointer
    for i, blob in enumerate(frames):
        if len(out) % 2:
            out += b"\x00"
        ifd_start = len(out)
        if first_ifd_offset is None:
            first_ifd_offset = ifd_start
        # Patch the previous "next IFD" field to point here.
        out[prev_next_ifd_field:prev_next_ifd_field + 4] = struct.pack(
            bo + "I", ifd_start
        )

        entries = [
            (256, 4, 1, shape[1]),
            (257, 4, 1, shape[0]),
            (258, 3, 1, 8),
            (259, 3, 1, 65002),         # Compression = EER v2
            (262, 3, 1, 1),
            (273, 4, 1, strip_offsets[i]),
            (277, 3, 1, 1),
            (278, 4, 1, shape[0]),
            (279, 4, 1, len(blob)),
            (65007, 3, 1, skipbits),
            (65008, 3, 1, horzbits),
            (65009, 3, 1, vertbits),
        ]
        entries.sort(key=lambda e: e[0])
        out += struct.pack(bo + "H", len(entries))
        for tag, tc, count, value in entries:
            out += struct.pack(bo + "HHI", tag, tc, count)
            out += (struct.pack(bo + "HH", value, 0) if tc == 3
                    else struct.pack(bo + "I", value))
        prev_next_ifd_field = len(out)
        out += struct.pack(bo + "I", 0)   # NextIFD — patched on next iter

    return bytes(out)


def test_eer_reader_iter_frames(tmp_path):
    """EerReader walks the IFD chain and decodes each frame to the
    same array a direct ``decode()`` call would."""
    from opencodecs._eer_reader import EerReader

    # Two frames of synthetic EER bitstream (same payload twice — fine
    # for testing the wrapper plumbing).
    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    shape = (20, 16)
    frames = [encoded, encoded]
    blob = _build_multi_frame_eer_tiff(frames, shape)
    path = tmp_path / "synth.eer"
    path.write_bytes(blob)

    expected = decode(encoded, shape, 7, 1, 1)
    with EerReader(str(path)) as r:
        assert r.n_frames == 2
        assert r.shape == shape
        assert r.dtype == np.uint8
        seen = list(r.iter_frames())
    assert len(seen) == 2
    for f in seen:
        np.testing.assert_array_equal(f, expected)


def test_eer_reader_sum_accumulates_events(tmp_path):
    """``sum()`` accumulates events across a frame range — the
    dose-corrected-average primitive cryo-EM users want. Three
    identical frames should sum to 3x one frame's counts."""
    from opencodecs._eer_reader import EerReader

    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    shape = (20, 16)
    blob = _build_multi_frame_eer_tiff([encoded] * 3, shape)
    path = tmp_path / "synth.eer"
    path.write_bytes(blob)

    one = decode(encoded, shape, 7, 1, 1).astype(np.uint16)
    with EerReader(str(path)) as r:
        total = r.sum(dtype=np.uint16)
        partial = r.sum(start=0, stop=2, dtype=np.uint16)
    np.testing.assert_array_equal(total, one * 3)
    np.testing.assert_array_equal(partial, one * 2)


def test_eer_reader_sum_validates_range(tmp_path):
    from opencodecs._eer_reader import EerReader
    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    blob = _build_multi_frame_eer_tiff([encoded], (20, 16))
    path = tmp_path / "synth.eer"
    path.write_bytes(blob)
    with EerReader(str(path)) as r:
        with pytest.raises(ValueError):
            r.sum(start=2, stop=1)
        with pytest.raises(ValueError):
            r.sum(start=0, stop=99)


def test_eer_codec_registered_and_dispatches(tmp_path):
    """``oc.has_codec('eer')`` is True and ``oc.get_codec('eer').open()``
    returns an EerReader. Confirms the registration wiring works."""
    import opencodecs as oc
    from opencodecs._eer_reader import EerReader

    assert oc.has_codec("eer")
    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    blob = _build_multi_frame_eer_tiff([encoded], (20, 16))
    path = tmp_path / "synth.eer"
    path.write_bytes(blob)
    with oc.get_codec("eer").open(str(path)) as r:
        assert isinstance(r, EerReader)
        assert r.n_frames == 1


def test_eer_codec_open_via_extension(tmp_path):
    """``oc.open('foo.eer')`` should route through the registry to
    EerReader by file-extension alone."""
    import opencodecs as oc

    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    blob = _build_multi_frame_eer_tiff([encoded, encoded], (20, 16))
    path = tmp_path / "scan.eer"
    path.write_bytes(blob)
    with oc.open(str(path)) as r:
        # Should be an EerReader (not TiffStream) — same file but
        # the extension-based dispatcher picked the EER codec.
        from opencodecs._eer_reader import EerReader
        assert isinstance(r, EerReader)
        assert r.n_frames == 2


def test_eer_codec_encode_raises():
    """EER is a detector-only format — encode should not be silently
    a no-op or a TIFF passthrough."""
    import opencodecs as oc
    with pytest.raises(NotImplementedError):
        oc.get_codec("eer").encode(np.zeros((4, 4), dtype=np.uint8))
