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
| `bitshuffle` | ✓ | ✓ | vendored bitshuffle (filter) | — |

`bitshuffle` is a *filter*, not a stand-alone compressor: bit-level
transpose that radically improves LZ77 ratios on typed numerical data.
Output size equals input size; pair with `zstd` / `lz4`. Aliases:
`bshuf`.

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
| `htj2k` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit, lossless + lossy | system OpenJPH | `.j2c` |
| `jpegls` | ✓ | ✓ | gray / RGB / RGBA, 2-16 bit, lossless + near-lossless | system CharLS | `.jls` |
| `avif` | ✓ | ✓ | RGB / RGBA, lossy + lossless (YUV444+identity) | libavif | `.avif` |
| `heif` | ✓ | ✓ | RGB / RGBA, lossy (HEVC) | libheif (+ aomenc) | `.heif`, `.heic` |
| `jxl` | ✓ | ✓ | gray / RGB / RGBA, P3, HDR, multi-frame | vendored libjxl 0.11.2 | `.jxl` |
| `bcdec` | — | ✓ | BC1-7 / DXT / BPTC GPU textures | vendored bcdec.h | `.dds` |

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

## Install

See [INSTALL.md](INSTALL.md) for system dependencies per platform and
build instructions, and [docs/publishing.md](docs/publishing.md) for
the wheel-publishing pipeline (TestPyPI / PyPI via Trusted Publishing).
Short version:

```sh
# macOS
brew install jpeg-turbo webp libavif libheif openjpeg libtiff hdf5 c-blosc2 \
             charls openjph libdeflate zlib-ng-compat

# Ubuntu / Debian
sudo apt install -y libturbojpeg0-dev libwebp-dev libavif-dev libheif-dev \
                    libopenjp2-7-dev libblosc2-dev libcharls-dev \
                    liblz4-dev libspng-dev libtiff-dev libhdf5-dev \
                    libdeflate-dev libopenjph-dev zlib1g-dev

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

- Core API stable; **1066 tests passing** (Mac M1 Ultra + Linux + Windows VM)
- Native readers + writers for the common scientific containers
  (TIFF, BigTIFF, OME-TIFF, CZI, NDTiff, HDF5, JXL)
- Cross-platform bench coverage: Mac arm64 (canonical), Windows 11 LTSC
  (libvirt VM), Linux x86_64 (Threadripper)
- Compression backend auto-detect (libdeflate → zlib-ng-compat → stdlib)
- Cloud I/O primitives (`HTTPDataSource.read_many`, range coalescing,
  HTTP/1.1 keep-alive) wired into TIFF / HDF5 / DICOMweb / CZI readers
- `tifffile_patch` opt-in shim reroutes tifffile's codec dispatch through
  opencodecs for users who want only a partial swap

Deferred work (see [`docs/TODO_DEFERRED.md`](docs/TODO_DEFERRED.md)):

- SPERR (error-bounded lossy scientific compression) — CMake build needed
- Brunsli (lossless JPEG transcoder) — source build needed; no brew formula
- CCITT Fax3/Fax4 encode — legacy fax; zero scientific users
- JPEG-XR — abandoned format outside niche DICOM
- libspng `filter_sum` SIMD — off the bench-tracked workload (`h2h_png_4mp_rgb`
  is at 1.14× already); filter-bound PNG-encode users could see another 2-3×
- Wheels / PyPI release — install from source for now

## License

BSD-3-Clause. Vendored components retain their original licenses (see
3rdparty/).
