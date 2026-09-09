"""Region and tile decode for JPEG 2000.

Taking a window out of a large image without expanding the rest is
most of why this format exists for satellite and slide imagery, and
until now opencodecs decoded the whole codestream every time. The
``reduce`` work gave the resolution half of that; this is the spatial
half, and the two compose.

Both halves of every claim get checked: the pixels have to be the
RIGHT window -- a fast call returning the wrong crop is worse than a
slow one -- and it has to actually cost less, because a region call
that decodes everything and slices is correct and pointless.
"""

from __future__ import annotations

import pathlib
import subprocess
import time

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("jpeg2k"), reason="jpeg2k not built here")

# openjpeg's own CLI, the only thing here that writes a tiled
# codestream: neither this package's encoder nor imagecodecs does.
_OPJ = next(
    (p for p in pathlib.Path("/opt/homebrew/Cellar").glob(
        "openjpeg/*/bin/opj_compress") if p.is_file()),
    None) if pathlib.Path("/opt/homebrew/Cellar").is_dir() else None


@pytest.fixture(scope="module")
def image():
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return np.clip(128 + 90 * np.sin(yy / 30) * np.cos(xx / 23),
                   0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def blob(image):
    return oc.get_codec("jpeg2k").encode(image)


@pytest.mark.parametrize("box", [
    (0, 128, 0, 128),            # a corner
    (256, 384, 256, 384),        # the middle
    (896, 1024, 896, 1024),      # the far corner
    (0, 1024, 0, 64),            # a tall strip
    (0, 64, 0, 1024),            # a wide strip
])
def test_a_region_is_the_right_pixels(box, blob, image):
    y0, y1, x0, x1 = box
    got = oc.get_codec("jpeg2k").decode_region(blob, y0, y1, x0, x1)
    want = np.asarray(oc.get_codec("jpeg2k").decode(blob))[y0:y1, x0:x1]
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_a_region_costs_less_than_the_whole_image(blob):
    """Correctness alone would be satisfied by decoding everything.

    The threshold is loose because this is not a benchmark: a 128x128
    window of a 1024x1024 image measured around 13x cheaper, so 2x is
    a floor that separates a real region decode from a crop.
    """
    codec = oc.get_codec("jpeg2k")

    def best(fn, reps=3):
        fn()
        b = 1e9
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            b = min(b, time.perf_counter() - s)
        return b

    whole = best(lambda: codec.decode(blob))
    window = best(lambda: codec.decode_region(blob, 128, 256, 128, 256))
    assert window < whole / 2, (
        f"a 128x128 window took {window * 1e3:.1f} ms against "
        f"{whole * 1e3:.1f} ms for the full image; that is a crop, not "
        f"a region decode")


@pytest.mark.parametrize("reduce", [0, 1, 2, 3])
def test_region_and_reduce_compose(reduce, blob):
    """Coordinates stay on the full-resolution grid at every zoom.

    openjpeg applies cp_reduce to the region itself, which an earlier
    version of this code did not know: it shifted the coordinates
    first and halved the window twice, returning 128 pixels where 256
    were asked for. A caller picks a window once and changes the zoom.
    """
    got = oc.get_codec("jpeg2k").decode_region(
        blob, 256, 512, 256, 512, reduce=reduce)
    assert got.shape[:2] == (256 >> reduce, 256 >> reduce)


def test_an_empty_or_backwards_region_is_refused(blob):
    codec = oc.get_codec("jpeg2k")
    for box in [(10, 10, 0, 50), (0, 50, 10, 10), (50, 10, 0, 50)]:
        with pytest.raises(ValueError, match="empty region"):
            codec.decode_region(blob, *box)


def test_tile_zero_of_an_untiled_codestream_is_the_image(blob, image):
    """Our encoder writes single-tile codestreams, so this is the tile
    path's floor: tile 0 exists everywhere and is the whole image."""
    got = oc.get_codec("jpeg2k").decode_tile(blob, 0)
    assert np.array_equal(got, np.asarray(oc.get_codec("jpeg2k").decode(blob)))


def test_a_missing_tile_is_refused(blob):
    codec = oc.get_codec("jpeg2k")
    with pytest.raises(ValueError, match="must be >= 0"):
        codec.decode_tile(blob, -1)
    with pytest.raises(Exception, match="fewer tiles"):
        codec.decode_tile(blob, 999)


@pytest.mark.skipif(_OPJ is None, reason="opj_compress not available")
def test_each_tile_of_a_tiled_codestream(tmp_path, image):
    """The real tile test, on a codestream that actually has tiles.

    Neither this package's encoder nor imagecodecs writes them, so
    without openjpeg's CLI the tile path can only be called, not
    tested -- and a call that returns without raising proves nothing
    about which pixels came back.
    """
    n, tile = image.shape[0], 256
    pgm = tmp_path / "in.pgm"
    with pgm.open("wb") as fh:
        fh.write(f"P5\n{n} {n}\n255\n".encode())
        fh.write(image.tobytes())
    out = tmp_path / "tiled.j2k"
    r = subprocess.run(
        [str(_OPJ), "-i", str(pgm), "-o", str(out),
         "-t", f"{tile},{tile}", "-r", "1"],
        capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        pytest.skip(f"opj_compress failed: {r.stderr[:120]}")

    data = out.read_bytes()
    codec = oc.get_codec("jpeg2k")
    per_row = n // tile
    assert np.array_equal(np.asarray(codec.decode(data)), image), \
        "the tiled codestream did not round-trip losslessly"

    for idx in (0, 1, per_row, per_row * per_row - 1):
        got = codec.decode_tile(data, idx)
        row, col = divmod(idx, per_row)
        want = image[row * tile:(row + 1) * tile,
                     col * tile:(col + 1) * tile]
        assert got.shape == want.shape, idx
        assert np.array_equal(got, want), f"tile {idx} is the wrong pixels"
