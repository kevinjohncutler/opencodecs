"""AVIF YUV to RGB in row bands must equal one whole-frame conversion.

decode() converts YUV to RGB on several threads by cutting the frame
into bands of rows, each converted through a view of the image. For
4:2:0 with bilinear chroma upsampling (libavif's default) a row also
reads the chroma row above or below its own, which a view clamps at its
edge, so the rows beside every band edge are converted again with the
rows they need. These tests compare every thread count against
numthreads=1, which is a single avifImageYUVToRGB call over the whole
frame, for each chroma layout and bit depth, with and without alpha, at
heights that do not divide evenly into bands.

Content varies from row to row in every channel, so a band edge that
read the wrong chroma row changes the output. Disabling the re-conversion
of the edge rows makes the 4:2:0 cases here fail at exactly those rows.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("avif"), reason="libavif not built here")

THREADS = (2, 3, 4, 7, 16)


def _content(h, w, c, maxval, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    planes = [np.clip(0.5 + 0.3 * np.sin(y / (2.0 + k) + x / 7.0)
                      + rng.normal(0, 0.08, (h, w)), 0, 1) for k in range(c)]
    a = np.round(np.stack(planes, -1) * maxval)
    return a.astype(np.uint8 if maxval == 255 else np.uint16)


def _same_at_every_thread_count(data):
    codec = oc.get_codec("avif")
    whole = codec.decode(data, numthreads=1)
    for n in THREADS:
        got = codec.decode(data, numthreads=n)
        assert got.dtype == whole.dtype and got.shape == whole.shape
        bad = np.unique(np.nonzero(got != whole)[0])
        assert bad.size == 0, f"numthreads={n}: rows {bad[:12]} differ"
    with codec.open(data, numthreads=5) as reader:
        assert np.array_equal(reader.frame(0), whole)
    out = np.empty_like(whole)
    assert codec.decode(data, numthreads=6, out=out) is out
    assert np.array_equal(out, whole)
    return whole


@pytest.mark.parametrize("yuv", ["420", "422", "444"])
@pytest.mark.parametrize("depth", [8, 10, 12])
@pytest.mark.parametrize("alpha", [False, True], ids=["rgb", "rgba"])
@pytest.mark.parametrize("hw", [(257, 45), (514, 33), (771, 20)],
                         ids=lambda hw: f"{hw[0]}x{hw[1]}")
def test_color_bands_match_whole_frame(yuv, depth, alpha, hw):
    h, w = hw
    maxval = 255 if depth == 8 else (1 << depth) - 1
    img = _content(h, w, 4 if alpha else 3, maxval, depth + h)
    data = oc.get_codec("avif").encode(
        img, level=60, speed=10, yuv_format=yuv, bit_depth=depth)
    _same_at_every_thread_count(data)


@pytest.mark.parametrize("alpha", [False, True], ids=["gray", "gray-alpha"])
def test_monochrome_bands_match_whole_frame(alpha):
    img = _content(389, 31, 2 if alpha else 1, 255, 5)
    if not alpha:
        img = img[..., 0]
    data = oc.get_codec("avif").encode(img, level=60, speed=10)
    _same_at_every_thread_count(data)


def test_lossless_bands_round_trip():
    img = _content(301, 29, 3, 255, 9)
    data = oc.get_codec("avif").encode(img, speed=10)
    assert np.array_equal(_same_at_every_thread_count(data), img)


# Limited-range 4:2:0, written by libavif's avifenc (1.4.1, -q 40 -y 420
# -r limited): 34 x 259, 8-bit with alpha, and 10-bit. The encoder here
# writes full range only.
_LIMITED_8BIT_ALPHA = (
    "AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAAGGbWV0YQAAAAAAAAAhaGRscgAA"
    "AAAAAAAAcGljdAAAAAAAAAAAAAAAAAAAAAAOcGl0bQAAAAAAAQAAACxpbG9jAAAAAEQAAAIA"
    "AQAAAAEAAAMEAAACHwACAAAAAQAAAa4AAAFWAAAAQmlpbmYAAAAAAAIAAAAaaW5mZQIAAAAA"
    "AQAAYXYwMUNvbG9yAAAAABppbmZlAgAAAAACAABhdjAxQWxwaGEAAAAAGmlyZWYAAAAAAAAA"
    "DmF1eGwAAgABAAEAAADDaXBycAAAAJ1pcGNvAAAAFGlzcGUAAAAAAAAAIgAAAQMAAAAQcGl4"
    "aQAAAAADCAgIAAAADGF2MUOBAAwAAAAAE2NvbHJuY2x4AAEADQAGAAAAAA5waXhpAAAAAAEI"
    "AAAADGF2MUOBABwAAAAAOGF1eEMAAAAAdXJuOm1wZWc6bXBlZ0I6Y2ljcDpzeXN0ZW1zOmF1"
    "eGlsaWFyeTphbHBoYQAAAAAeaXBtYQAAAAAAAAACAAEEAQKDBAACBAEFhgcAAAN9bWRhdBIA"
    "CgYYFiGBMKgyyQIUwCCAgLtijch39p7r/k+vv/luZvoRRIy4hkpnW3hbTE93JVV052yrQnjz"
    "hIcP8C81+GNZki8CFSy5Iys59ImL8xGNz9hcOoauy9aJLx8OqFlKs130BKxrePyctYfs1k6J"
    "yvQ29g77/DkVXnoqUAnXc/ZwC7yhrtPr6DLEMzPgkBb0iEA/9liGwPV1Rnv8BC1K4U5qEt+W"
    "u9Gcnhe9gxvdU2/WpU6pzLTWjuC5JoLiITSIp2P4KMiryOADvKm2pc5zv8tWfzcrfC6jBxLQ"
    "05CyvD1B9HNv21wrPgMg367d/uGxO0M18qfq1e3fmzC5DL6LROt7gP0yYg7/FF8eEArh3IFv"
    "zXehHKd88e00wst3jrSKCynP7uiWz6+L7XJZ08CgL/EkU+SFPVvrmL/ZUQWaMYby+35J8Bmn"
    "hd/PbYz5JmN0y3s23tGPgBIACgkYFiGBNEBDQYEyjwQVY8PC8OYYYYFCikDOU/se/3ys39co"
    "/0+W1/UOajyAKkIjWHXxag2wgZJRLZx+Bm6Mq+MVbp3g4NxhJtZccVYVRtfE35dgYEe9zUg0"
    "wghYeAJA0Za+VekRDyMRkRG0BbOxz/9skAPMEEEAqGXLLf8CNuR+zwzw3LzGAHaJcLnPmP7d"
    "UfqOZduEryc8NYRp4ryq606pi6ZaZzcvZJWibXN/qIRwFpgezhEYxp2qxND+PRe5B911zBB1"
    "rfbVYa3hhBfqWC3RfiBNdOiLbuYpAcaVRvHycJYmno5jCmna08Lj9gLdZWPxpyx/AumH0D6l"
    "U+FIqQSue5v4MTvKP0YCyTAQmxpXW6cT4GIXebbKJBwEAJ+ZHC0xufCAXmI9sCI6ynlJmuzB"
    "BJBcMXcj2Ng6dxHs8Ef+Znxp2xXQblwg1Byoe7wYs9pnfXoyAZ09SeFHw5LyJq2F58o0NNlF"
    "7xBgrr+64gKGOEmlqIO85CFCLO4wjazV/ua5Q1tFQXRfsBcMgqejlauApISug+lFvGlfWHXF"
    "Hgbvi/XanBpTU6fskM4zmIb6UaBatJRSGZQWDwYV6Ab/QjYfKN+weE2SSbJAfjJlW+EjVPoQ"
    "DOhGVGwMRZsR9GUFRYZYlNZ98XoaeDLeiDMbSe6acIfH4+Cpn3oh27yOzOQH8nKVoRQsW0O7"
    "R3MOoHiRkXB3OhlduTarCEU/oA=="
)
_LIMITED_10BIT = (
    "AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADrbWV0YQAAAAAAAAAhaGRscgAA"
    "AAAAAAAAcGljdAAAAAAAAAAAAAAAAAAAAAAOcGl0bQAAAAAAAQAAAB5pbG9jAAAAAEQAAAEA"
    "AQAAAAEAAAETAAACDgAAAChpaW5mAAAAAAABAAAAGmluZmUCAAAAAAEAAGF2MDFDb2xvcgAA"
    "AABqaXBycAAAAEtpcGNvAAAAFGlzcGUAAAAAAAAAIgAAAQMAAAAQcGl4aQAAAAADCgoKAAAA"
    "DGF2MUOBAEwAAAAAE2NvbHJuY2x4AAEADQAGAAAAABdpcG1hAAAAAAAAAAEAAQQBAoMEAAAC"
    "Fm1kYXQSAAoJGBYhgTVAQ0GBMv4DFWPDwvDnnnnhQooAz6NLOQl3pE6W4H0NedElYfv2iuEe"
    "mEugJeCt0Zp4qGOFmgQZgwVtQ9pBu1WGcgb32ofqF03AW3Sat/XExKJbSag0ZezuIPn/z9cL"
    "uyYXZUjFM7nLEHuCI02vgcquxORM8eEuETr4yK2wJ/kfUmlfmLt2ZpxSsPFL1Mdw+e/RvW1p"
    "wllW+NTThuv4dxkTXVssBJI555otrpgNAeU8GaranyIqcvuHpCP93qCnfDs3ppqtQ2yJBXz2"
    "+WWdYhy7gNOTsiGBeFPnNiX7+D8I882bSN4MbRaJpFvH9VN93WsjtScc0KJL1PXfNi5M/avg"
    "FWOMGap1gGzZWJXaSpNt+5pm+G7DZuDB8zZldJVDOpTpHrS/utIeRgUnOYx4NgX6mObLeNJJ"
    "MtiSJa9J4DC62S2drvN5hQGhPgvcB1qvbe7XHMBFFA4ulSYXFYIViBXGexhd7a7rH2TCXsDS"
    "YRaZIilMZQjx3JY8U2G1lvjbITuqJHc6Tcpxoknzb9V8SAAByXP4S2O87DVX5bgJZNeeBglW"
    "2+K/Yeal9SioKcgpA7GG22VbEyJMd2LruI+AVmRJ9pRsDwJohOBuxi7jqX4puv9W4L9ujVQ+"
    "1GY7ErtfDZSd4PmS/w9CeWhHjAhGWQMfYc+vkYqsygxhMgqwZRDC6tHCRfwg"
)


@pytest.mark.parametrize("fixture", ["8bit-alpha", "10bit"])
def test_limited_range_bands_match_whole_frame(fixture):
    data = base64.b64decode(
        _LIMITED_8BIT_ALPHA if fixture == "8bit-alpha" else _LIMITED_10BIT)
    whole = _same_at_every_thread_count(data)
    assert whole.shape == ((259, 34, 4) if fixture == "8bit-alpha"
                           else (259, 34, 3))
