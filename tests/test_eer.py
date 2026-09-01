"""EER (Electron Event Representation) decoder tests.

The reference test vector ``b'\\x03\\x1b\\xfc\\xb1\\x35\\xfb'`` is taken
straight from the EER format specification and matches imagecodecs'
own test cases. Event positions are pre-computed in the spec, so we
assert against absolute expected values rather than cross-validate.

Additional fuzz cross-validation against imagecodecs.eer_decode (when
present) covers parameter combinations on random bitstreams.
"""

from __future__ import annotations

import pathlib

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


def _encode_frame(events, ncells, sb, hb, vb):
    """Build a well-formed EER frame: the events, then a final skip that
    lands exactly on the last cell, then ones padding to a byte edge.

    Random bytes are NOT a valid frame -- they have no terminator, so two
    decoders can legitimately disagree about whether the last few bits of
    padding form one more event. Generating real frames tests decoding
    instead of tail-handling trivia.
    """
    bits = []

    def put(value, n):
        for i in range(n):
            bits.append((value >> i) & 1)

    maxskip = (1 << sb) - 1
    prev = 0
    for pos, horz, vert in events:
        gap = pos - prev
        while gap >= maxskip:            # continuations for long gaps
            put(maxskip, sb)
            gap -= maxskip
        put(gap, sb)
        put(horz, hb)
        put(vert, vb)
        prev = pos + 1
    gap = ncells - prev                  # terminate by exact fill
    while gap >= maxskip:
        put(maxskip, sb)
        gap -= maxskip
    put(gap, sb)
    while len(bits) % 8:
        bits.append(1)
    return bytes(sum(b << j for j, b in enumerate(bits[i:i + 8]))
                 for i in range(0, len(bits), 8))


def test_eer_decodes_generated_frames():
    """Round-trip well-formed frames we build ourselves. No imagecodecs."""
    rng = np.random.default_rng(7)
    for sb in (7, 8):
        for hb in (1, 2):
            for vb in (1, 2):
                shape = (64, 64)
                ncells = shape[0] * shape[1]
                pos = sorted(rng.choice(ncells - 1, size=40, replace=False).tolist())
                events = [(int(p), int(rng.integers(0, 1 << hb)),
                           int(rng.integers(0, 1 << vb))) for p in pos]
                data = _encode_frame(events, ncells, sb, hb, vb)
                im = decode(data, shape, sb, hb, vb, superres=0)
                got = sorted(int(r) * shape[1] + int(c)
                             for r, c in zip(*np.nonzero(im)))
                assert got == [p for p, _, _ in events], (
                    f"sb={sb} hb={hb} vb={vb}")


# Real Falcon 4 data. Downloaded by tests/download_test_corpus.sh into the
# gitignored .test_data/, so this skips cleanly in CI and on a fresh clone.
# EMPIAR-10568, Krios G4 apoferritin, 4096x4096, 721 frames, compression
# 65001 (7-bit RLE with 2+2 sub-pixel bits), CC0.
_REAL_EER = (pathlib.Path(__file__).parent.parent
             / ".test_data" / "eer" / "empiar10568_falcon4.eer")


@pytest.mark.skipif(not _REAL_EER.is_file(),
                    reason="run tests/download_test_corpus.sh --eer")
@pytest.mark.parametrize("superres", [0, 1, 2])
def test_eer_real_falcon4_matches_imagecodecs(superres):
    """Cross-validate on genuine camera output, not synthetic bitstreams.

    Random bytes are not a valid EER frame: they carry no terminator, so
    two decoders can legitimately disagree about whether trailing padding
    forms one more event. Real frames end by walking the position exactly
    onto the last cell, which is unambiguous.
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    tifffile = pytest.importorskip("tifffile")
    if not getattr(imagecodecs, "EER", None) or not imagecodecs.EER.available:
        pytest.skip("imagecodecs EER backend unavailable")

    factor = 1 << superres
    shape = (4096 * factor, 4096 * factor)
    with tifffile.TiffFile(_REAL_EER) as tif:
        fh = tif.filehandle
        for index in range(4):
            page = tif.pages[index]
            fh.seek(page.dataoffsets[0])
            raw = fh.read(page.databytecounts[0])
            ours = decode(raw, shape, 7, 2, 2, superres=superres)
            theirs = imagecodecs.eer_decode(raw, shape, 7, 2, 2,
                                            superres=superres)
            np.testing.assert_array_equal(
                ours, theirs, err_msg=f"frame {index}, superres={superres}")


@pytest.mark.skipif(not _REAL_EER.is_file(),
                    reason="run tests/download_test_corpus.sh --eer")
def test_eer_real_falcon4_frame_is_exactly_filled():
    """Every real frame terminates by landing exactly on the last cell,
    which is why exact fill is normal termination and overshoot is not."""
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffFile(_REAL_EER) as tif:
        fh = tif.filehandle
        for index in range(4):
            page = tif.pages[index]
            fh.seek(page.dataoffsets[0])
            raw = fh.read(page.databytecounts[0])
            im = decode(raw, (4096, 4096), 7, 2, 2, superres=0)
            # ~197k electrons per 3 ms frame at this dose rate
            assert 150_000 < int(im.sum()) < 250_000


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


def test_eer_reader_sum_weighted(tmp_path):
    """``sum(weights=...)`` applies a per-frame dose curve so the
    accumulator can down-weight high-drift frames or boost peak-dose
    frames. Identical input frames + weights [0.5, 1.0, 2.0] must
    give 3.5x one frame's counts."""
    from opencodecs._eer_reader import EerReader

    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    shape = (20, 16)
    blob = _build_multi_frame_eer_tiff([encoded] * 3, shape)
    path = tmp_path / "synth.eer"
    path.write_bytes(blob)

    one = decode(encoded, shape, 7, 1, 1).astype(np.float64)
    expected = one * 0.5 + one * 1.0 + one * 2.0
    with EerReader(str(path)) as r:
        weighted = r.sum(weights=np.array([0.5, 1.0, 2.0]))
    assert weighted.dtype == np.float64
    np.testing.assert_allclose(weighted, expected)


def test_eer_reader_sum_weighted_rejects_wrong_length(tmp_path):
    from opencodecs._eer_reader import EerReader
    encoded = b"\x03\x1b\xfc\xb1\x35\xfb"
    blob = _build_multi_frame_eer_tiff([encoded] * 3, (20, 16))
    path = tmp_path / "synth.eer"
    path.write_bytes(blob)
    with EerReader(str(path)) as r:
        with pytest.raises(ValueError, match="weights"):
            r.sum(weights=np.array([1.0, 1.0]))  # 2 weights for 3 frames
        with pytest.raises(ValueError, match="weights"):
            # 2 weights for a 1-frame range
            r.sum(start=0, stop=1, weights=np.array([1.0, 1.0]))


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


# ---------------------------------------------------------------------------
# Sub-pixel inversion, and where imagecodecs gets it wrong
# ---------------------------------------------------------------------------


def _one_event(field, nbits, skipbits=7, vertbits=1):
    """A one-event stream at base position 0 with horz=field, vert=0."""
    bits = []

    def put(value, n):
        for i in range(n):
            bits.append((value >> i) & 1)

    put(0, skipbits)
    put(field, nbits)
    put(0, vertbits)
    while len(bits) % 8:          # EER frames are byte aligned; pad with ones
        bits.append(1)
    return bytes(sum(b << j for j, b in enumerate(bits[i:i + 8]))
                 for i in range(0, len(bits), 8))


@pytest.mark.parametrize("horzbits", [1, 2, 3, 4])
def test_eer_subpixel_inversion_is_width_independent(horzbits):
    """The top bit of each sub-pixel field is inverted, at every width.

    Ground truth is RELION's renderEER.cpp, which XORs the packed symbol:

        s = ((chunk >> 7) & 15) ^ 0x0A;   // 2 horz + 2 vert bits
        s = ((chunk >> 7) &  3) ^ 3;      // 1 horz + 1 vert bits

    0x0A is 0b1010 and 0x03 is 0b11. Both flip exactly the MSB of each
    field and leave the low bits alone, so the rule does not depend on
    the field width and we apply it uniformly.

    imagecodecs applies the inversion at widths 1 and 2 but not 3 and 4.
    Real Falcon hardware only emits 1 or 2, so that does not affect real
    data, but it does mean this test is deliberately NOT cross-validated
    against imagecodecs.
    """
    cols = []
    for field in range(1 << horzbits):
        im = decode(_one_event(field, horzbits), (8, 16), 7, horzbits, 1,
                    superres=1)
        cols.append(int(np.nonzero(im)[1][0]))
    assert cols == [1 - (f >> (horzbits - 1)) for f in range(1 << horzbits)]


def test_eer_frame_terminates_on_exact_fill_not_overshoot():
    """A frame ends when its final skip lands exactly on the last cell.

    Real frames always end that way and carry a trailing footer, so
    "bits remain" cannot mean the shape was wrong. Overshooting the
    frame is what indicates an undersized output.
    """
    # 20x16 holds the spec vector's four events, the last at 311
    assert int(decode(SPEC_ENCODED, (20, 16), 7, 1, 1).sum()) == 4
    with pytest.raises(EerError):
        decode(SPEC_ENCODED, (19, 15), 7, 1, 1)     # 285 cells, overshoots
