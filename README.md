<!-- markdownlint-disable MD060 -->

# opencodecs

[![Tests](https://github.com/kevinjohncutler/opencodecs/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinjohncutler/opencodecs/actions/workflows/tests.yml)
[![Build wheels](https://github.com/kevinjohncutler/opencodecs/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/kevinjohncutler/opencodecs/actions/workflows/build_wheels.yml)

Native, parallel-decode codecs for scientific imaging. One unified Codec
/ Reader / Writer API across compression streams, single images,
multi-frame stacks, and chunked containers.

Built for fast modern storage (NVMe, 10 G NAS) where the bottleneck is
codec dispatch and per-tile parallelism, not raw I/O bandwidth. Native
implementations of every codec — no runtime delegation to
[imagecodecs](https://github.com/cgohlke/imagecodecs) — though we use
its excellent test suite as a parity reference.

```python
import opencodecs as oc

arr = oc.read("scan.czi")              # auto-detect by extension
arr = oc.read("photo.jxl")
arr = oc.read(blob)                    # auto-detect by magic bytes

oc.write("out.jxl", arr, lossless=True)
oc.write("out.zst", b"...payload...", level=10)

# Streaming reader for multi-frame / chunked formats
with oc.get_codec("czi").open(path) as r:
    print(r.shape, r.dtype, r.n_frames)
    for tile in r:                     # iter_frames
        ...
    tile5 = r[5]                       # random access

# Discovery
oc.list_codecs()                       # capability table
oc.has_codec("avif")
```

## Codec capability matrix

All codecs below are native implementations linking against system or
vendored C libraries. Build skips cleanly when an optional system
library is missing — see [INSTALL.md](INSTALL.md).

### Compression (bytes → bytes)

| Codec | Encode | Decode | Backing library | Extension |
| --- | :-: | :-: | --- | --- |
| `zstd` | ✓ | ✓ | system libzstd | `.zst` |
| `lz4` | ✓ | ✓ | system liblz4 (frame) | `.lz4` |
| `brotli` | ✓ | ✓ | system libbrotli | `.br` |
| `blosc2` | ✓ | ✓ | system c-blosc2 | `.b2` |
| `deflate` | ✓ | ✓ | system zlib | `.zlib` |

### Single-image codecs

| Codec | Encode | Decode | Color | Backing library | Extension |
| --- | :-: | :-: | --- | --- | --- |
| `qoi` | ✓ | ✓ | RGB / RGBA | vendored qoi.h | `.qoi` |
| `bmp` | ✓ | ✓ | gray / RGB / RGBA | pure Python+numpy | `.bmp`, `.dib` |
| `png` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit | system libspng | `.png` |
| `jpeg` | ✓ | ✓ | gray / RGB | libjpeg-turbo (TJ v3) | `.jpg`, `.jpeg` |
| `webp` | ✓ | ✓ | RGB / RGBA, lossy + lossless | system libwebp | `.webp` |
| `jpeg2k` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit, lossless + lossy | OpenJPEG | `.jp2`, `.j2k`, `.jpx`, `.jpc` |
| `avif` | ✓ | ✓ | RGB / RGBA, lossy + lossless (YUV444+identity) | libavif | `.avif` |
| `heif` | ✓ | ✓ | RGB / RGBA, lossy (HEVC) | libheif (+ aomenc) | `.heif`, `.heic` |
| `jxl` | ✓ | ✓ | gray / RGB / RGBA, P3, HDR, multi-frame | vendored libjxl 0.11.2 | `.jxl` |

### Multi-frame / chunked formats

| Codec | Decode | Container | Notes |
| --- |:-:| --- |---|
| `jxl` | ✓ | ISO BMFF (frame index) | Streaming + parallel multi-frame decode |
| `czi` | ✓ | Zeiss ZISRAW | mmap + parallel zstd; metadata accessor |
| `hdf5` | ✓ | HDF5 | Wraps `h5py.Dataset` |

`czi` decodes types 0 (uncompressed) and 6 (ZSTDHDR) — the entire
modern Zen archive. JPEG-XR sub-blocks (rare in 2022+ output) raise
`NotImplementedError`; native jxrlib support is tracked for v0.2.

`czi` exposes `reader.metadata_bytes` and `reader.metadata_xml` as
lazy zero-copy accessors so downstream parsers (e.g. hiprpy's Cython
`metadata_summary`) can consume the XML without a `bytes → str → bytes`
round trip.

### zarr v3 codecs

`opencodecs._zarr_codecs` registers our compressors as zarr v3
`BytesBytesCodec`s:

```python
import zarr
from opencodecs._zarr_codecs import OcZstd, OcLz4, OcBlosc2, OcBrotli, OcDeflate

z = zarr.create_array(
    store=..., shape=..., dtype=..., chunks=...,
    compressors=[OcZstd(level=10)],
    zarr_format=3,
)
```

## Performance

Scientific microscopy CZI (66 MB, 14 sub-blocks of 2000×2000 uint16,
ZSTDHDR), single-file warm cache:

| Reader        | Mac M3 | Threadripper x86_64 |
|---------------|-------:|--------------------:|
| czifile (Python ref) | 148 ms | 414 ms       |
| aicspylibczi (C++)   |  17 ms | 140 ms       |
| **opencodecs**       |  **15 ms** | **46 ms**  |

64-core Threadripper benefits dramatically more from parallel decode
than the 12-core M3.

Pipeline benchmark (8 CZIs back-to-back, 795 MB total, NAS):

| Reader        | Total  | Per-file | Throughput |
|---------------|-------:|---------:|-----------:|
| czifile       | 1603 ms | 229 ms  | 0.50 GB/s  |
| **opencodecs** | **198 ms** | **22 ms** | **4.01 GB/s** |
| aicspylibczi  | 191 ms |  22 ms   | 4.16 GB/s  |

JXL multi-frame parallel decode on Mac arm64 (16-frame uint16 stack):

| Approach                 | Time    |
|--------------------------|--------:|
| Sequential `iter_frames` | 68 ms   |
| `decode_frames_parallel(n_workers=16)` | 24 ms (2.8×) |

See [docs/io_patterns.md](docs/io_patterns.md) for the lessons learned
about coalesced I/O, mmap vs pread, persistent thread pools, and where
parallelism actually pays off.

## Public API

### Top-level dispatch

```python
oc.read(src, *, format=None, **opts) -> ndarray | bytes
oc.write(dest, data, *, format=None, **opts) -> bytes | None
oc.codec_for_path(path) -> Codec | None
oc.codec_for_bytes(head) -> Codec | None
```

`src` and `dest` accept paths, file-like objects, bytes, and
memoryview / mmap slices (zero-copy through the codec).

### Codec registry

```python
oc.list_codecs() -> list[Codec]
oc.has_codec(name_or_alias) -> bool
oc.get_codec(name_or_alias) -> Codec
```

### Codec interface

Each codec exposes:

```python
codec.name            # "czi"
codec.file_extensions # (".czi",)
codec.has_native      # True for everything we ship
codec.can_encode / codec.can_decode
codec.multi_frame / codec.chunked / codec.streaming_decode / codec.parallel_decode
codec.supported_dtypes / codec.supports_color

codec.signature(head_bytes) -> bool
codec.encode(data, *, dest=None, **opts) -> bytes | None
codec.decode(src, **opts) -> ndarray | bytes
codec.open(src, **opts) -> Reader        # multi-frame / chunked
```

### Reader interface (multi-frame / chunked)

```python
reader.shape       # (n_frames, *frame_shape)
reader.dtype
reader.n_frames
reader.is_chunked  # True if [idx] random access works
reader.iter_frames()
reader.read()      # full eager decode
reader[idx]        # random access (chunked formats only)
```

CZI reader additionally exposes:

```python
reader.entries                  # list[CziSubBlockEntry] — sub-block metadata
reader.metadata_bytes           # raw UTF-8 bytes (lazy + cached)
reader.metadata_xml             # decoded str (lazy + cached)
reader.subblock_metadata_bytes(i)
```

HDF5 reader additionally exposes:

```python
reader.dataset_names            # all numeric datasets in the file
reader.select(name)             # switch to a different dataset
```

## Install

See [INSTALL.md](INSTALL.md) for system dependencies per platform and
build instructions. Short version:

```sh
# macOS
brew install jpeg-turbo webp libavif libheif openjpeg libtiff hdf5 c-blosc2

# Ubuntu / Debian
sudo apt install -y libturbojpeg0-dev libwebp-dev libavif-dev libheif-dev \
                    libopenjp2-7-dev libblosc2-dev libcharls-dev \
                    liblz4-dev libspng-dev libtiff-dev libhdf5-dev

# Build
cd opencodecs
pip install -e .
# or
python setup.py build_ext --inplace
```

The build skips cleanly for any system library that's missing — useful
extensions still build, missing ones print a one-line notice.

libjxl 0.11.2 is vendored via `bench/build_libjxl.sh` (auto-builds + caches
to `~/Library/Caches/opencodecs/libjxl/` on Mac, `~/.cache/opencodecs/libjxl/`
on Linux). See INSTALL.md for the rationale (Homebrew/apt builds are
0.5-0.7× slower than a tuned `-O3 + LTO` build).

## Status

- Core API stable
- 76 parity tests passing on Mac
- 66 parity tests + 9 graceful skips on Linux x86_64 (Threadripper)
- CZI native reader benchmarked against czifile + aicspylibczi
- JXL native reader with frame-index parallel decode

What's not yet shipped:

- Native JPEG-XR (jxrlib) — small minority of older CZI files only
- Native LERC, JPEG-LS — no current users
- Native TIFF parser — `tifffile` already wins on this hardware; not worth
  reimplementing. `opencodecs.tifffile_patch` provides an opt-in shim
  that reroutes tifffile's codec dispatch through opencodecs.
- Wheels / PyPI release — install from source for now

## License

BSD-3-Clause. Vendored components retain their original licenses (see
3rdparty/).
