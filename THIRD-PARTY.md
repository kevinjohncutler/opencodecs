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
| imcd (EER decoder excerpt) | `3rdparty/imcd_eer/` | imagecodecs, Christoph Gohlke | BSD-3-Clause (`LICENSE`) |
| imcd (TIFF LZW encoder excerpt) | `3rdparty/imcd_lzw/` | imagecodecs, Christoph Gohlke | BSD-3-Clause (`LICENSE`) |
| bcdec | `3rdparty/bcdec/` | Sergii Kudlai | MIT or public domain, dual (`LICENSE`) |
| bitshuffle | `3rdparty/bitshuffle/` | Kiyoshi Masui | MIT (`LICENSE`) |
| cfitsio (rice, hcompress, plio) | `3rdparty/cfitsio/` | NASA / HEASARC | Permissive NASA notice (`License.txt`) |
| libspng | `3rdparty/libspng/` | Randy | BSD-2-Clause (`LICENSE`) |
| qoi | `3rdparty/qoi/` | Dominic Szablewski | MIT (`LICENSE`) |
| rgbe | `3rdparty/rgbe/` | Bruce Walter, after Greg Ward | No formal license statement; see the disclaimer at the top of `rgbe.c` and the notes in `rgbe.txt` |
| oc_giflzw | `3rdparty/oc_giflzw/` | opencodecs authors | MIT (`LICENSE`) |
| oc_tifflzw | `3rdparty/oc_tifflzw/` | opencodecs authors | MIT (`LICENSE`) |

`oc_giflzw` and `oc_tifflzw` are original opencodecs work, not derived
from any of the above.

The `rgbe` files carry no explicit license grant upstream. They are
distributed here in the form they have been circulated in since 1997.
Anyone with better provenance information is invited to open an issue.

## 2. Relationship to imagecodecs

[imagecodecs](https://github.com/cgohlke/imagecodecs) is BSD-3-Clause,
Copyright (c) 2008-2026 Christoph Gohlke. opencodecs began as a fork of
it, and two vendored C components under `3rdparty/` are still excerpts
of its `imcd.c` (see section 1).

No Cython source in this repository is derived from it. Two `.pxd`
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
