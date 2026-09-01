# Third-party code and licenses

opencodecs itself is BSD-3-Clause (see [LICENSE](LICENSE)). It also
vendors source from other projects, derives some Cython declaration
files from another project, and links against a set of codec libraries
that are bundled into the binary wheels. Each of those retains its own
license. This file is the inventory.

## 1. Source vendored in this repository

Each directory under `3rdparty/` contains the upstream license text
alongside the source.

| Component | Path | Upstream | License |
|---|---|---|---|
| oc_eer | `3rdparty/oc_eer/` | opencodecs authors | MIT (`LICENSE`) |
| bcdec | `3rdparty/bcdec/` | Sergii Kudlai | MIT or public domain, dual (`LICENSE`) |
| bitshuffle | `3rdparty/bitshuffle/` | Kiyoshi Masui | MIT (`LICENSE`) |
| cfitsio (rice, hcompress, plio) | `3rdparty/cfitsio/` | NASA / HEASARC | Permissive NASA notice (`License.txt`) |

The `cfitsio` files track upstream **cfitsio-4.7.0**. `fits_hcompress.c`
is byte-identical to it and `fits_hdecompress.c` differs only in
comments. `pliocomp.c` is taken verbatim from that release, which added
the `srclen` bounds checks. `ricecomp.c` keeps the error-code return
convention an earlier vendoring introduced, since our binding depends
on it, with upstream's buffer-size guard applied on top.

Re-check these against upstream periodically rather than assuming a
vendored copy is current: both fixes above landed upstream after the
version originally vendored here, and one of them was a crash.
| libspng | `3rdparty/libspng/` | Randy | BSD-2-Clause (`LICENSE`) |
| qoi | `3rdparty/qoi/` | Dominic Szablewski | MIT (`LICENSE`) |
| rgbe | `3rdparty/rgbe/` | Bruce Walter, after Greg Ward | No formal license statement; see the disclaimer at the top of `rgbe.c` and the notes in `rgbe.txt` |

`rgbe` is not ours and is not a single author's work. The chain is Greg
Ward's original scheme, Bruce Walter's 1995 C reference, fixes by Denis
Mentey, then fixes by Christoph Gohlke (buffer overflow and partial-line
handling in `rgbe_stream_gets`, stream instead of FILE, const
qualifiers), then an opencodecs change to the exponent conversion. All
four are credited in the file header, and Gohlke's fixes are the reason
this copy is safe; do not drop them.

There is no newer upstream to move to. The format was frozen in 1991 and
the maintained implementations all descend from the same reference:
OpenImageIO's `hdrinput.cpp` says so in its own header ("Based on source
code that originally came from ... Bruce Walter ... modified very
heavily"), and three.js's RGBELoader is adapted from the same file.

Checked against OpenImageIO, which is the most actively maintained C++
version, rather than assuming: its RLE validation is check-for-check the
same as ours (scanline width bounds, magic bytes, width match, and
`count == 0 || count > ptr_end - ptr` on both the run and literal
paths). Its one advantage was converting the exponent through a
compile-time table instead of a per-pixel `ldexp`, which we have now
adopted, with entry 0 set to zero so the nonzero-pixel branch folds away
too.

| oc_giflzw | `3rdparty/oc_giflzw/` | opencodecs authors | MIT (`LICENSE`) |
| oc_tifflzw | `3rdparty/oc_tifflzw/` | opencodecs authors | MIT (`LICENSE`) |

`oc_giflzw`, `oc_tifflzw` and `oc_eer` are original opencodecs work, not
derived from any of the above. `oc_tifflzw` holds both directions of TIFF LZW;
the encoder replaced a vendored excerpt of imagecodecs' `imcd.c` and is
written against TIFF 6.0 section 13.

The `rgbe` files carry no explicit license grant upstream. They are
distributed here in the form they have been circulated in since 1997.
Anyone with better provenance information is invited to open an issue.

## 2. Relationship to imagecodecs

[imagecodecs](https://github.com/cgohlke/imagecodecs) is BSD-3-Clause,
Copyright (c) 2008-2026 Christoph Gohlke. opencodecs began as a fork of
it. No code from it remains: the two vendored `imcd.c` excerpts, the
TIFF LZW encoder and the EER decoder, have both been replaced by our
own implementations.

Nothing in this repository is derived from it. Two `.pxd`
files had been copied from imagecodecs; `libjxl.pxd` has been rewritten
from the libjxl 0.11.2 public headers and now declares only the subset
opencodecs calls, and `libultrahdr.pxd` was unused and has been
deleted. Several `.pxd` files still resemble their imagecodecs
counterparts, because both declare the same upstream C APIs and the
headers fix the names, signatures and enum orderings. Each such file
says so in its own header comment.

The `.pyx` implementations, the pure-Python package and the test suite
are original work. Where a comment says a default or a call sequence
"matches imagecodecs", it documents deliberate behavioral parity so
output is interchangeable, not shared source.

## 3. Codec libraries linked into the binary wheels

`bench/build_codec_libs.sh` builds these from upstream source at the
pinned versions below, and the wheels bundle the resulting shared
libraries. Licenses are as declared by each upstream project at the
pinned tag; the authoritative copy is the license file installed with
each library.

| Library | Version | License |
|---|---|---|
| zlib | 1.3.1 | zlib |
| zstd | 1.5.7 | BSD-3-Clause or GPL-2.0, dual |
| lz4 | 1.10.0 | BSD-2-Clause (library) |
| brotli | 1.1.0 | MIT |
| giflib | 5.2.2 | MIT |
| libdeflate | 1.23 | MIT |
| libpng | 1.6.50 | libpng-2.0 |
| libjpeg-turbo | 3.1.2 | IJG, BSD-3-Clause and zlib |
| libwebp | 1.6.0 | BSD-3-Clause |
| openjpeg | 2.5.5 | BSD-2-Clause |
| mozjpeg | 4.1.5 | IJG and BSD-3-Clause |
| c-blosc2 | 2.23.0 | BSD-3-Clause |
| dav1d | 1.5.1 | BSD-2-Clause |
| SVT-AV1 | 3.1.2 | BSD-3-Clause and AOM patent license |
| libavif | 1.3.0 | BSD-2-Clause |
| libde265 | 1.0.16 | LGPL-3.0-or-later |
| libheif | 1.21.0 | LGPL-3.0-or-later |
| libultrahdr | 1.4.0 | Apache-2.0 |
| libaec | 1.1.6 | BSD-2-Clause |
| lerc | 4.1.0 | Apache-2.0 |
| zfp | 1.0.1 | BSD-3-Clause |
| SZ3 | 3.3.1 | BSD-3-Clause |
| SPERR | 0.8.5 | Apache-2.0 |
| pcodec | 1.0.2 | Apache-2.0 |
| brunsli | master | MIT |
| CharLS | 2.4.3 | BSD-3-Clause |
| libjxl | 0.11.2 | BSD-3-Clause |

Two notes on this list:

**libaom and x265 are not in the published wheels.** The build script
enables both by default for local development, but the wheel workflow
sets `ENABLE_AOM=0` and `ENABLE_X265=0`. x265 is GPL-2.0 and is
deliberately excluded. Wheels therefore ship decode-only AVIF (via
dav1d) and decode-only HEIF (via libde265). A wheel built locally with
the defaults is **not** redistributable under BSD-3-Clause terms.

**libheif and libde265 are LGPL-3.0-or-later.** They are dynamically
linked and bundled unmodified. Redistributing these wheels carries the
LGPL's obligations for those two libraries, including conveying the
license text and permitting relinking against a modified version.
