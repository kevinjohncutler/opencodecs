<!-- markdownlint-disable MD060 -->

# opencodecs

[![PyPI](https://img.shields.io/pypi/v/opencodecs.svg)](https://pypi.org/project/opencodecs/)
[![Tests](https://github.com/kevinjohncutler/opencodecs/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinjohncutler/opencodecs/actions/workflows/tests.yml)
[![Build wheels](https://github.com/kevinjohncutler/opencodecs/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/kevinjohncutler/opencodecs/actions/workflows/build_wheels.yml)

Native, parallel, cloud-aware codecs for scientific imaging. One
unified Codec / Reader / Writer API across compression streams,
single images, multi-frame stacks, and chunked containers — with
HTTP range-fetch and per-chunk parallelism wired in at the bottom
of the stack, not bolted on.

Built for fast modern storage (NVMe, 10 G NAS, S3) where the
bottleneck is codec dispatch and per-tile parallelism, not raw I/O
bandwidth. Native implementations of every codec — no runtime
delegation to [imagecodecs](https://github.com/cgohlke/imagecodecs) —
though we use its excellent test suite as a parity reference.

```sh
pip install opencodecs
```

```python
import opencodecs as oc

# 1. Look at any scientific image file
arr = oc.read("scan.czi")              # auto-detect by extension
arr = oc.read("photo.jxl")
arr = oc.read(blob)                    # auto-detect by magic bytes

# 2. Write with the right codec for the data
oc.write("out.jxl", arr, lossless=True)
oc.write("out.zst", b"...payload...", level=10)

# 3. Stream multi-frame / chunked formats
with oc.get_codec("czi").open(path) as r:
    print(r.shape, r.dtype, r.n_frames)
    for tile in r:                     # iter_frames
        ...
    tile5 = r[5]                       # random access

# 4. Fetch tiles of a remote pyramidal TIFF over HTTPS by range request
with oc.open_pyramid("https://example.com/slide.svs") as p:
    region = p.read_region(level=2, y=(1024, 2048), x=(1024, 2048))
    # → 2-3 HTTP Range requests, not a full slide download

# Discovery
oc.list_codecs()                       # capability table
oc.has_codec("avif")
```

## Why opencodecs

| Need | What you get |
|---|---|
| **Decode regions of cloud-hosted TIFF/Zarr/HDF5 without downloading the whole file** | Native `HTTPDataSource` with range-coalescing + adaptive read-ahead, wired into the TIFF/NDTiff/HDF5/Zarr/FITS pyramid readers |
| **Per-chunk parallel decode of CZI/OME-TIFF/NDTiff stacks** | Built-in `ThreadPoolExecutor` orchestration with nogil-released codec calls; 3–10× over single-threaded reference readers on large stacks |
| **Modern codec coverage (JPEG XL, AVIF, HEIF, JPEG-LS, Brunsli, Ultra HDR, OME-Zarr v3 sharded)** | All shipped, all with native bindings — no `pip install ten-other-packages` |
| **Tier-1 scientific compressors (LERC, ZFP, SZ3, SPERR, pcodec, bitshuffle, blosc2, libaec)** | All shipped, source-built with `-O3 + LTO + hidden-visibility` for Pareto wins over distro builds |
| **Lossless drop-in replacement for `imagecodecs`** | `tifffile_patch` opt-in shim reroutes tifffile's codec dispatch through opencodecs without changing your tifffile code |

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
| `blosc2` | ✓ | ✓ | source-built c-blosc2 2.23 | `.b2` |
| `deflate` | ✓ | ✓ | libdeflate / zlib-ng / zlib (auto-selected at build time) | `.zlib` |
| `gzip` | ✓ | ✓ | stdlib gzip | `.gz` |
| `none` | ✓ | ✓ | identity (filter-chain placeholder) | — |
| `bz2` | ✓ | ✓ | stdlib bz2 | `.bz2` |
| `lzma` | ✓ | ✓ | stdlib lzma | `.xz` |
| `snappy` | ✓ | ✓ | system snappy | `.sz` |
| `bitshuffle` | ✓ | ✓ | vendored bitshuffle (filter) | — |

`bitshuffle` is a *filter*, not a stand-alone compressor: bit-level
transpose that radically improves LZ77 ratios on typed numerical data.
Output size equals input size; pair with `zstd` / `lz4`. Aliases:
`bshuf`.

`deflate` aliases: `zlib`, `zlibng`. Pass `backend="isal"` to opt into
Intel ISA-L's igzip (~4× faster encode on x86_64; opt-in because
output is ~19% bigger). The default backend is auto-selected at build
time: libdeflate when present (fastest at default level), else
zlib-ng-compat, else the stdlib zlib.

### Scientific / numerical-array codecs (ndarray ↔ bytes, self-describing)

These four codecs target *typed multidimensional arrays* rather than
images or raw bytes. The encoded blob carries shape and dtype in its
header, so `decode(blob)` reconstructs the full ndarray without
out-of-band metadata.

| Codec | Encode | Decode | Lossless | Lossy modes | Backing library | Extension |
| --- | :-: | :-: | :-: | --- | --- | --- |
| `b2nd` | ✓ | ✓ | ✓ | — | system c-blosc2 (NDim API) | `.b2nd` |
| `aec` | ✓ | ✓ | ✓ | — | system libaec (CCSDS 121.0-B-2) | `.aec` |
| `lerc` | ✓ | ✓ | ✓ | `max_z_error` | system liblerc (Esri) | `.lerc` |
| `zfp` | ✓ | ✓ | ✓ (reversible) | rate / precision / accuracy | system libzfp | `.zfp` |
| `sz3` | ✓ | ✓ | — | abs / rel / psnr / norm | source-built SZ3 | `.sz3` |
| `pcodec` | ✓ | ✓ | ✓ | — | source-built pcodec (Rust) | `.pco` |

Quick guidance:

- `pcodec` — modern lossless numerical compressor; often beats `zstd`
  by 1.5–3× on float / int arrays without a pre-filter.
- `b2nd` — c-blosc2's multidim layer with shuffle/bitshuffle filters
  built in; great when you already use blosc2 elsewhere.
- `aec` — entropy coder used by NetCDF-4 SZIP; lossless integers.
- `lerc` — fast (lossy or lossless) raster codec used in
  Cloud-Optimized GeoTIFF, Esri MRF.
- `zfp` — fast 1D-4D float / int compression with multiple lossy modes
  (predictable size, accuracy, or precision).
- `sz3` — error-bounded prediction-based scientific compressor;
  often beats `zfp` at the same error budget on simulation snapshots.
  *Float only* (the SZ3 v3 C API doesn't dispatch integer types).

### Single-image codecs

| Codec | Encode | Decode | Color | Backing library | Extension |
| --- | :-: | :-: | --- | --- | --- |
| `qoi` | ✓ | ✓ | RGB / RGBA | vendored qoi.h | `.qoi` |
| `bmp` | ✓ | ✓ | gray / RGB / RGBA | pure Python+numpy | `.bmp`, `.dib` |
| `png` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit | vendored libspng + libdeflate | `.png` |
| `jpeg` | ✓ | ✓ | gray / RGB | libjpeg-turbo (TJ v3) | `.jpg`, `.jpeg` |
| `mozjpeg` | ✓ | ✓ | gray / RGB, 8/12-bit | system mozjpeg (TJ v2) | `.jpg` |
| `webp` | ✓ | ✓ | RGB / RGBA, lossy + lossless | system libwebp | `.webp` |
| `jpeg2k` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit, lossless + lossy | OpenJPEG | `.jp2`, `.j2k`, `.jpx`, `.jpc` |
| `htj2k` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit, lossless + lossy | OpenJPH 0.31.0 (source-built) | `.j2c` |
| `jpegls` | ✓ | ✓ | gray / RGB / RGBA, 2-16 bit, lossless + near-lossless | system CharLS | `.jls` |
| `avif` | ✓ | ✓ | RGB / RGBA, lossy + lossless (YUV444+identity) | libavif | `.avif` |
| `heif` | ✓ | ✓ | RGB / RGBA, lossless + lossy (HEVC) | libheif (+ aomenc) | `.heif`, `.heic` |
| `jxl` | ✓ | ✓ | gray / RGB / RGBA, P3, HDR, multi-frame | vendored libjxl 0.11.2 | `.jxl` |
| `bcdec` | — | ✓ | BC1-7 / DXT / BPTC GPU textures | vendored bcdec.h | `.dds` |
| `rgbe` | ✓ | ✓ | float32 RGB HDR (Radiance) | vendored rgbe.c | `.hdr` |
| `ultrahdr` | ✓ | ✓ | float16 / uint8 / uint16 RGBA HDR + SDR | system libultrahdr 1.4.x | `.jpg` (gainmap) |

`htj2k` is JPEG-2000 Part 15 (High-Throughput) — same DWT front end
as classic JPEG-2000 but ~10-20× faster entropy coding. Used by
modern DICOM and remote-sensing pipelines.

`jpegls` (CharLS) is the lossless / near-lossless predictive JPEG
variant standardized as ISO/IEC 14495-1 — the dominant codec in
medical-imaging DICOM workflows.

`mozjpeg` is Mozilla's libjpeg-turbo fork; ~10-15% smaller files
than libjpeg-turbo at the same quality. Built only when MozJPEG is
on the system (keg-only on Homebrew so it doesn't collide with
plain libjpeg-turbo).

`rgbe` is the canonical Radiance HDR format — float32 RGB shared-
exponent encoding for high-dynamic-range photography and physically-
based rendering output. `ultrahdr` is the ISO 21496 gainmap-JPEG
format — Android Camera's default since A14 and what iOS 18+ reads
natively. Decode dtype controls the output: `float16` returns linear
BT.2100 HDR; `uint8` returns the SDR-tonemapped base JPEG.

### Multi-frame / chunked formats

| Codec | Read | Write | Container | Notes |
| --- |:-:|:-:| --- |---|
| `jxl` | ✓ | ✓ | ISO BMFF (frame index) | Streaming + parallel multi-frame decode |
| `czi` | ✓ | ✓ | Zeiss ZISRAW | mmap + parallel zstd; metadata accessor; parallel bulk HTTP fetch via `CziReader.from_http(max_workers=N)` |
| `tiff` | ✓ | ✓ | TIFF 6.0 + BigTIFF | Native reader + writer; tiled or strip; parallel encode; LZW encode; streaming write to unseekable sinks; EER cryo-EM dispatch |
| `ndtiff` | ✓ | ✓ | Micro-Manager / Pycro-Manager NDTiff | Streaming writer; `os.writev` hot path; cross-platform (POSIX + Windows-NTFS-safe pre-allocation) |
| `hdf5` | ✓ | ✓ | HDF5 | Wraps `h5py.Dataset`. Remote HDF5 via `open_remote_hdf5(url)` — slices stream chunks over HTTP Range with one-shot parallel prefetch |
| `eer` | ✓ | — | Thermo Fisher EER (cryo-EM event-list) | Native bitstream decoder + TIFF compression-tag dispatch (codes 65000-65002) |
| `dicomweb` | ✓ | — | WADO-RS HTTP frame retrieval | Multipart/related parser; transfer-syntax dispatch through opencodecs's codec layer (JPEG-LS / HTJ2K / JPEG-2000 / RLE / raw) |
| `fits` | ✓ | — | FITS (astronomy) | Multi-HDU walk; BITPIX 8/16/32/64/-32/-64; BZERO unsigned-int trick; compressed images (RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1, NOCOMPRESS) with per-tile ZSCALE/ZZERO quantization. HTTP-range friendly — opening a 50 GB cube reads kilobytes. |
| `mrc` | ✓ | — | MRC2014 / CCP4 map (cryo-EM volumes, EMDB deposits) | Read-only. MODE 0/1/2/6/12 plus complex; both byte orders; extended header; `plane(i)` for one z-section; `canonical=True` reorients a permuted MAPC/MAPR/MAPS to (z, y, x). |
| `nifti` | ✓ | — | NIfTI-1 / NIfTI-2 (neuroimaging volumes) | Read-only. Both header versions and byte orders; transparent gzip, since almost every NIfTI in the wild is `.nii.gz`; scl_slope/scl_inter applied when they change anything and skipped when they do not, so an unscaled integer volume stays integer. |
| `n5` | ✓ | — | N5 (Janelia / Saalfeld chunked arrays) | Read-only, via `opencodecs.N5Array`. Local directory, http(s) URL or a fetch callable, so an N5 on S3 reads like one on disk. raw/gzip/bzip2/xz plus blosc, lz4 and zstd through our own codecs; column-major dimensions reversed to C order; big-endian per-block headers; absent blocks read as zeros the way sparse datasets expect. |
| `imaris` | ✓ | — | Imaris `.ims` (Bitplane, HDF5-based) | Read-only, via `opencodecs.ImarisReader` and `open_pyramid`. Resolution pyramid, timepoints and channels; crops the padding Imaris leaves in the stored array using each level's own ImageSize attributes; decodes the character-array attribute convention. Needs `h5py`. |

#### TIFF writer specifics

```python
from opencodecs._tiff_writer import TiffWriter

# Classic TIFF (<4 GiB)
with TiffWriter("out.tif") as w:
    w.write_page(arr, tile=(256, 256), compression="zstd")

# BigTIFF (>4 GiB; magic=43, 64-bit offsets)
with TiffWriter("huge.tif", bigtiff=True) as w:
    w.write_pyramid(levels, compression="zstd", subifds=True)

# COG-style streaming to an unseekable sink (pipe, S3 multipart, HTTP body)
with TiffWriter(sink, streaming=True) as w:
    w.write_stream(pages, total_pages=N, tile=(256, 256), compression="zstd")
```

Supported encode-side compressions: none, deflate (libdeflate /
zlib-ng / zlib auto-detect), zstd, LZW, JPEG, JPEG2000, WebP, JXL,
LERC. Horizontal predictor on byte-stream codecs.

#### OME-TIFF metadata

```python
from opencodecs._ome_xml import write_ome_tiff, Channel

write_ome_tiff(
    "scan.ome.tif", arr_5d, axes="TCZYX",
    physical_size_um=(0.108, 0.108, 0.5),
    channels=[Channel(name="DAPI", emission_wavelength_nm=460),
              Channel(name="GFP",  emission_wavelength_nm=520)],
)
```

Round-trips through tifffile / Bio-Formats / QuPath. For schema
elements outside the 80%-case subset, hand-author OME-XML and pass
via TiffWriter's `metadata=` kwarg.

#### Remote HDF5

```python
from opencodecs._hdf5_http import open_remote_hdf5, prefetch_hdf5_chunks

with open_remote_hdf5("https://bucket.s3.amazonaws.com/big.h5") as f:
    prefetch_hdf5_chunks(f["img"], np.s_[:1024, :1024])  # 1 syscall, N HTTP
    arr = f["img"][:1024, :1024]                          # all from cache
```

`czi` decodes types 0 (uncompressed) and 6 (ZSTDHDR) — the entire
modern Zen archive. JPEG-XR sub-blocks (rare in 2022+ output) raise
`NotImplementedError`. The reader exposes `metadata_bytes` and
`metadata_xml` as lazy zero-copy accessors.

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

Headline numbers from the latest bench run (`bench/run_benchmarks.py
--fast`, macOS M1 Ultra, vs `imagecodecs` / `tifffile` / `ndstorage`):

| Workload | opencodecs | reference | ratio |
|---|---:|---:|---:|
| `tiff_random_tile_read` | 0.70 ms | 7.71 ms (tifffile) | **11×** |
| `tiff_pyramid_crop_from_fullres` | 0.47 ms | 8.60 ms | **18×** |
| `ndtiff_index_parse_synthetic_10k` | 4.61 ms | 28.0 ms (ndstorage) | **6.1×** |
| `h2h_jxl_4mp_rgb` (encode) | 130 ms | 3153 ms (imagecodecs) | **24×** |
| `h2h_blosc2_10mb` | 4.63 ms | 54.8 ms | **12×** |
| `h2h_deflate_10mb` (encode) | 109 ms | 296 ms | **2.7×** |
| `h2h_png_4mp_rgb` (encode) | 142 ms | 281 ms | **2.0×** |
| `h2h_png_kodak_photo` (encode) | 19 ms | 58 ms | **3.1×** |
| `h2h_png_filterbound_u16` (encode) | 2.0 ms | 3.7 ms | **1.8×** |
| `tiff_write_1gb` | 89 ms | 91 ms | parity, +14% on Windows |
| `ndtiff_write_1gb` (raw 800 MB) | 159 ms | 154 ms | parity (1.04× on macOS, 2.4× on Windows after NTFS-friendly pre-alloc) |

The PNG encode wins above stack two independent improvements:
the `libdeflate` IDAT accumulator (already shipped) collapses
zlib's per-scanline `deflate()` loop into a single one-shot call,
and a per-filter split of libspng's `filter_sum` hot path lets
the compiler autovectorize each branch into NEON/SSE — together
they make every PNG-encode workload 1.5–3.1× faster than
imagecodecs.

Remote-fetch workloads benefit from `read_many` (one batched HTTP
fan-out + Range coalescing) — on a loopback Range-supporting server,
1024-chunk HDF5 slices land in 7 HTTP requests instead of 1010 (a
~50× request-count reduction; on real-network RTT this translates
to 8× wall-clock).

Scientific microscopy CZI (66 MB, 14 sub-blocks of 2000×2000 uint16,
ZSTDHDR), single-file warm cache:

| Reader        | Mac M3 | Threadripper x86_64 |
|---------------|-------:|--------------------:|
| czifile (Python ref) | 148 ms | 414 ms       |
| aicspylibczi (C++)   |  17 ms | 140 ms       |
| **opencodecs**       |  **15 ms** | **46 ms**  |

See [docs/io_patterns.md](docs/io_patterns.md) for the lessons learned
about coalesced I/O, mmap vs pread, persistent thread pools, and where
parallelism actually pays off. The deflate path is libdeflate when
available → zlib-ng-compat → stdlib zlib, auto-detected at build time.

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

## Streaming-reader examples

### 1. Fetch a region of a remote Aperio whole-slide TIFF

```python
import opencodecs as oc

# Pyramidal SVS (Aperio) hosted on S3 / any HTTPS endpoint with Range support.
with oc.open_pyramid("https://example.com/slide.svs") as p:
    print(p.levels)               # [(80000, 60000, 3), (40000, 30000, 3), ...]
    region = p.read_region(level=2, y=(1024, 3072), x=(2048, 4096))
    # Total HTTP traffic: ~6 Range requests covering only the tiles
    # that intersect this 2048×2048 bbox — typically 200 KB–2 MB,
    # not the 4 GB whole slide.
```

The pyramid reader auto-detects the best level for the requested
region, fetches only the intersecting TIFF tiles via HTTP Range,
and assembles the output in-memory. Works the same on local files,
NFS, SMB, S3, or any range-capable HTTP server.

### 2. Convert a multi-level pyramid to OME-Zarr v3 sharded

```python
import opencodecs as oc

with oc.open_pyramid("input.ome.tiff") as p:
    levels = [p.read_region(level=i) for i in range(len(p.levels))]

oc.write_omezarr_pyramid(
    "output.zarr",
    levels,
    chunks=(512, 512),
    shards=(2048, 2048),         # 16 chunks per shard, one file each
    compressor="zstd",
    zarr_format=3,
)
# 1 file per shard on disk instead of 1 file per chunk; per-chunk
# random access still works via Range fetches into the shard.
```

For data going to S3, sharded Zarr v3 cuts your `PUT` and `LIST`
costs by 1–2 orders of magnitude vs unsharded chunks while
preserving per-chunk random-access via HTTP Range — the reader
above understands the shard index automatically.

### 3. Fast JPEG XL thumbnails (native progressive decode)

```python
import opencodecs.jxl as jxl

# downsample=8 uses libjxl's native progressive decoder — stops at
# the DC pass without reconstructing full-resolution pixels.
thumb = jxl.read("scan.jxl", downsample=8, subsample="center")
# 4Kx4K input → 512x512 ndarray in ~28 ms on macOS arm64
# (vs ~40 ms for a full decode), positionally centroid-correct
# so SVG / GL renderers don't get a ½-block shift.

# For a partial JXL bitstream usable as a tiny browser-direct
# thumbnail (works in Safari + modern Chrome):
prefix = jxl.thumbnail_bytes("scan.jxl")
# → ~85 KB out of a 3.5 MB source for a 4Kx4K image
```

## Install

```sh
pip install opencodecs
```

Wheels are published for CPython 3.10–3.13 on macOS (arm64),
Linux (x86_64 + aarch64), and Windows (amd64). Each wheel
bundles libjxl, libavif, libheif, libwebp, libdeflate,
c-blosc2, and friends — no system dependencies needed.

For a source install, system development headers, or to build a
tuned local libjxl, see [INSTALL.md](INSTALL.md). Wheel publishing
runs through [docs/publishing.md](docs/publishing.md).

```sh
# Source install — auto-detects system libs, source-builds libjxl
git clone https://github.com/kevinjohncutler/opencodecs.git
cd opencodecs
pip install -e .
```

The build skips cleanly for any system library that's missing — useful
extensions still build, missing ones print a one-line notice. libjxl
0.11.2 is auto-built from source via `bench/build_libjxl.sh` and
cached under `~/Library/Caches/opencodecs/` (macOS) /
`~/.cache/opencodecs/` (Linux). See INSTALL.md for the rationale
(Homebrew/apt builds are 0.5-0.7× slower than a tuned `-O3 + LTO`
build).

## Status

- **v0.1.1** on PyPI (May 2026). Core API stable; **1066 tests passing**
  on Mac M1 Ultra + Linux x86_64/aarch64 + Windows VM
- Native readers + writers for the common scientific containers
  (TIFF, BigTIFF, OME-TIFF, CZI, NDTiff, HDF5, JXL, FITS,
  OME-Zarr v2 + v3 sharded)
- Cross-platform bench coverage: Mac arm64 (canonical), Windows 11 LTSC
  (libvirt VM), Linux x86_64 (Threadripper-class)
- Compression backend auto-detect (libdeflate → zlib-ng-compat → stdlib)
- Cloud I/O primitives (`HTTPDataSource` with covering-cache + adaptive
  read-ahead) wired into TIFF / HDF5 / DICOMweb / CZI / FITS / Zarr v3
  readers
- `tifffile_patch` opt-in shim reroutes tifffile's codec dispatch through
  opencodecs for users who want only a partial swap

Deferred work (see [`docs/TODO_DEFERRED.md`](docs/TODO_DEFERRED.md)):

- **Windows wheels currently miss `_sz3`, `_pcodec`, `_sperr`, `_brunsli`**
  — toolchain mismatch (conda's bash picks GCC over MSVC for CMake);
  v0.1.2 will restore them. macOS + Linux wheels have the full set.
- CCITT Fax3/Fax4 encode — legacy fax; zero scientific users
- JPEG-XR — abandoned format outside niche DICOM
- libspng `filter_sum` SIMD — off the bench-tracked workload (`h2h_png_4mp_rgb`
  is at 1.14× already); filter-bound PNG-encode users could see another 2-3×

## License

BSD-3-Clause; see [LICENSE](LICENSE).

Vendored source, the Cython declaration files derived from
[imagecodecs](https://github.com/cgohlke/imagecodecs) (BSD-3-Clause,
Copyright (c) 2008-2026 Christoph Gohlke), and the codec libraries
bundled into the binary wheels each retain their own license. The full
inventory is in [THIRD-PARTY.md](THIRD-PARTY.md).
