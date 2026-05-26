"""Verify a built wheel contains every required codec extension.

Run after cibuildwheel produces a wheel and before the publish step.
Catches silent extension drops caused by missing codec libs at build
time (e.g. setup.py's required-header probe dropping a codec when its
library prefix isn't set, which is how v0.1.2 shipped Windows wheels
that the changelog claimed had _sperr/_brunsli but actually didn't).

Usage:
    python ci/check_wheel_contents.py <wheel_path>

The wheel filename determines the platform; required codecs come from
MUST_SHIP_ALL_PLATFORMS plus any platform-specific extras.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path


# Codec extensions that MUST ship on every wheel cell. Failure to
# include any of these fails the build and blocks the PyPI publish.
# The point is to catch silent drops at build time; a codec being
# listed here is a promise the project makes in its README/changelog.
MUST_SHIP_ALL_PLATFORMS = {
    # Marquee
    "_jxl",
    "_tiff",
    # Tier 1 scientific compressors — the v0.1.2 → v0.1.3 motivating set
    "_sz3",
    "_pcodec",
    "_sperr",
    "_brunsli",
    "_aec",
    "_lerc",
    "_zfp",
    "_bitshuffle",
    "_blosc2",
    "_b2nd",
    # Codec building blocks
    "_zstd",
    "_deflate",
    "_brotli",
    "_lz4",
    # Image codecs
    "_png",
    "_jpeg",
    "_jpeg2k",
    "_mozjpeg",
    "_webp",
    "_avif",
    "_heif",
    "_qoi",
    "_bmp",
    "_rgbe",
    # TIFF dispatch + adjacent
    "_ndtiff",
    "_eer",
    "_hcomp",
    "_plio",
    "_rcomp",
    "_bcdec",
    "_bytetools",
}


def platform_from_wheel_name(wheel_name: str) -> str:
    """Return one of: linux_x86_64, linux_aarch64, macosx_arm64, win_amd64."""
    name = wheel_name.lower()
    if "win_amd64" in name:
        return "win_amd64"
    if "manylinux" in name and "x86_64" in name:
        return "linux_x86_64"
    if "manylinux" in name and "aarch64" in name:
        return "linux_aarch64"
    if "macosx" in name and "arm64" in name:
        return "macosx_arm64"
    raise ValueError(f"unrecognized wheel platform: {wheel_name}")


def codec_extensions_in_wheel(wheel_path: Path) -> set[str]:
    """Return the set of codec names (e.g. '_jxl') with a .pyd/.so inside."""
    found: set[str] = set()
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if not name.startswith("opencodecs/codecs/"):
                continue
            base = Path(name).name
            # Extension files look like '_jxl.cpython-312-x86_64-linux-gnu.so'
            # or '_jxl.cp312-win_amd64.pyd'.
            if not (base.endswith(".so") or base.endswith(".pyd")):
                continue
            codec = base.split(".", 1)[0]
            if codec.startswith("_"):
                found.add(codec)
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <wheel_path>", file=sys.stderr)
        return 2
    wheel = Path(argv[1])
    if not wheel.is_file():
        print(f"wheel not found: {wheel}", file=sys.stderr)
        return 2

    platform = platform_from_wheel_name(wheel.name)
    required = MUST_SHIP_ALL_PLATFORMS

    present = codec_extensions_in_wheel(wheel)
    missing = required - present
    extras = present - required

    print(f"wheel:    {wheel.name}")
    print(f"platform: {platform}")
    print(f"present:  {sorted(present)}")
    if extras:
        print(f"extras:   {sorted(extras)}  (not required, fine)")
    if missing:
        print(f"MISSING:  {sorted(missing)}")
        print(f"FAIL — every codec in MUST_SHIP_ALL_PLATFORMS must have a "
              f".pyd/.so in opencodecs/codecs/. setup.py's header probe "
              f"likely dropped a codec because its library prefix wasn't "
              f"on the search path. Check the build log for "
              f"'opencodecs: skipping extensions'.")
        return 1
    print("OK — every required codec present")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
