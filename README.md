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
| **Files that interoperate with `imagecodecs` and `tifffile`** | Shared codecs write the formats imagecodecs writes (a published spec first, imagecodecs where none decides); `oc.read` and `oc.tiff_imwrite` read and write TIFF natively |

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
| `gzip` | ✓ | ✓ | the deflate engine above (decode: libdeflate when linked, else stdlib zlib) | `.gz` |
| `none` | ✓ | ✓ | identity (filter-chain placeholder) | — |
| `bz2` | ✓ | ✓ | stdlib bz2 | `.bz2` |
| `lzma` | ✓ | ✓ | stdlib lzma | `.xz` |
| `snappy` | ✓ | ✓ | system snappy (raw block) | (none) |
| `snappy_framed` | ✓ | ✓ | system snappy (framing format) | `.sz` |
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

### Scientific / numerical-array codecs (ndarray ↔ bytes)

These codecs target *typed multidimensional arrays* rather than
images or raw bytes. Each writes its library's own stream, which
records more for some than for others:

- `b2nd`, `lerc` and `zfp` streams carry the shape and dtype, so
  `decode(blob)` reconstructs the full ndarray.
- `aec` writes the bare CCSDS 121.0-B-2 stream, which records nothing
  about its data: `decode` takes the coding parameters it was written
  with, plus `dtype` and `shape` (or `out`).
- `sz3` records the dimensions but not reliably the data type, so
  `decode` needs `dtype` (or `out`).
- `pcodec` records the number type but not the shape: without `shape`
  (or `out`) `decode` returns a flat array.

| Codec | Encode | Decode | Lossless | Lossy modes | Backing library | Extension |
| --- | :-: | :-: | :-: | --- | --- | --- |
| `b2nd` | ✓ | ✓ | ✓ | none | system c-blosc2 (NDim API) | `.b2nd` |
| `aec` | ✓ | ✓ | ✓ | none | system libaec (CCSDS 121.0-B-2) | `.aec` |
| `lerc` | ✓ | ✓ | ✓ | `max_z_error` | system liblerc (Esri) | `.lerc` |
| `zfp` | ✓ | ✓ | ✓ (reversible) | rate / precision / accuracy | system libzfp | `.zfp` |
| `sz3` | ✓ | ✓ | ✓ (default, abs 0) | abs / rel / abs_or_rel / abs_and_rel | source-built SZ3 | `.sz3` |
| `pcodec` | ✓ | ✓ | ✓ | none | source-built pcodec (Rust) | `.pco` |

In fixed-rate mode `zfp` blocks are individually addressable, so
`decode_block(data, n)` reads one 4x4x4 block without touching the rest
(0.0010 ms against 0.209 ms for the whole stream) and a full decode
splits the block grid across threads (110 ms → 30 ms on a 67 MB volume).
The variable-rate modes have no computable block position and fall back
to a whole-stream decode.

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
| `gif` | ✓ | ✓ | 8-bit palette → RGB, or the palette indices; **animated** (decodes to a frame stack); encode takes palette indices | system giflib + vendored LZW decoder | `.gif` |
| `png` | ✓ | ✓ | gray / RGB / RGBA, 8/16-bit | vendored libspng + libdeflate | `.png` |
| `jpeg` | ✓ | ✓ | gray / RGB / CMYK (YCCK), 8/12-bit lossy, 2-16 bit lossless | libjpeg-turbo (TJ v3) | `.jpg`, `.jpeg` |
| `mozjpeg` | ✓ | ✓ | gray / RGB, 8-bit encode; decodes what `jpeg` does (12-bit and lossless through `jpeg`) | system mozjpeg (TJ v2) | `.jpg` |
| `webp` | ✓ | ✓ | RGB / RGBA, lossy + lossless; **animated** (decodes to a frame stack, like `gif`) | system libwebp (+ libwebpdemux) | `.webp` |
| `jpeg2k` | ✓ | ✓ | up to 4095 components on encode (as imagecodecs), 8/16-bit unsigned or signed (decode to 32-bit), lossless + lossy | OpenJPEG | `.jp2`, `.j2k`, `.jpx`, `.jpc` |
| `htj2k` | ✓ | ✓ | 1 to 16384 components, 8/16/32-bit unsigned or signed, float32, lossless + lossy | OpenJPH 0.31.0 (source-built) | `.j2c` |
| `jpegls` | ✓ | ✓ | gray / RGB / RGBA, 2-16 bit, lossless + near-lossless | system CharLS | `.jls` |
| `avif` | ✓ | ✓ | gray / gray+alpha (4:0:0) / RGB / RGBA, 8/10/12-bit, lossless + lossy (color YUV 4:4:4, identity matrix when lossless); **image sequences** (decode to a frame stack, like `gif`) | libavif | `.avif` |
| `heif` | ✓ | ✓ | gray / gray+alpha (monochrome) / RGB / RGBA, 8/10/12-bit, lossless + lossy (HEVC, color 4:4:4); **every top-level image**, not just the primary | libheif (+ aomenc) | `.heif`, `.heic` |
| `jxl` | ✓ | ✓ | gray / RGB / RGBA, P3, HDR, multi-frame | vendored libjxl 0.11.2 | `.jxl` |
| `bcdec` | — | ✓ | BC1-7 / DXT / BPTC GPU textures; band decode + threaded | vendored bcdec.h | `.dds` |
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

#### Opt-in hardware backends

`jpeg`, `jpeg2k` and `htj2k` can encode and decode on an NVIDIA GPU with
`backend="nvimgcodec"` (`pip install 'opencodecs[gpu]'`), and `heif` can
decode, or encode lossy, with Apple's media engine via
`backend="imageio"` on macOS. Neither is ever the default: the GPU path
costs 1 to 1.6 s of startup per process, and both lose on batches of
small tiles. Where they fit, an HTJ2K lossless decode of a 4096 x 4096
uint16 image ran 9x faster on an RTX 4090 (16x into pinned memory) and a
large untiled HEIC decoded 10 to 19x faster on an M1 Ultra. Lossless
decodes match the CPU path bit for bit. Details and every measurement:
[docs/hardware_backends.md](docs/hardware_backends.md).

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
| `mrc` | ✓ | ✓ | MRC2014 / CCP4 map (cryo-EM volumes, EMDB deposits) | Read and write. MODE 0/1/2/6/12 plus complex; both byte orders; extended header; `plane(i)` for one z-section; `canonical=True` reorients a permuted MAPC/MAPR/MAPS to (z, y, x). MRCZ (blosc-compressed voxels) decodes too, and an http(s) URL reads through range requests: opening a 4 MB volume moves 64 KB. |
| `nifti` | ✓ | ✓ | NIfTI-1 / NIfTI-2 (neuroimaging volumes) | Read both, write NIfTI-1. Both header versions and byte orders; transparent gzip, since almost every NIfTI in the wild is `.nii.gz`; scl_slope/scl_inter applied when they change anything and skipped when they do not, so an unscaled integer volume stays integer. |
| `n5` | ✓ | — | N5 (Janelia / Saalfeld chunked arrays) | Read-only, via `opencodecs.N5Array`. Local directory, http(s) URL or a fetch callable, so an N5 on S3 reads like one on disk. raw/gzip/bzip2/xz plus blosc, lz4 and zstd through our own codecs; column-major dimensions reversed to C order; big-endian per-block headers; absent blocks read as zeros the way sparse datasets expect. |
| `imaris` | ✓ | — | Imaris `.ims` (Bitplane, HDF5-based) | Read-only, via `opencodecs.ImarisReader` and `open_pyramid`. Resolution pyramid, timepoints and channels; crops the padding Imaris leaves in the stored array using each level's own ImageSize attributes; decodes the character-array attribute convention. Needs `h5py`. |
| `dicom` | ✓ | — | DICOM files (`.dcm`) | Read-only, via `opencodecs.DicomFile`. Explicit and implicit VR, big-endian, deflated; native and encapsulated Pixel Data; multi-frame. Frames route through the same transfer-syntax dispatch DICOMweb uses, so JPEG, JPEG-LS, JPEG 2000, HTJ2K and RLE all work. Reconciles a codestream's signedness with Pixel Representation. Frames are indexed by the Basic Offset Table and decode across threads. VL Whole Slide Microscopy series read as pyramids through `open_pyramid(dir, format="dicom")`. |
| `nrrd` | ✓ | — | NRRD / NHDR (3D Slicer, ITK) | Read-only, via `opencodecs.NrrdFile`. raw, gzip, bzip2, ascii and hex encodings; both byte orders; detached `.nhdr` + `.raw` pairs; `sizes` is fastest-axis-first so the numpy shape is reversed. |
| `dm` | ✓ | — | Gatan Digital Micrograph (`.dm3`, `.dm4`) | Read-only, via `opencodecs.DmFile`. Walks the tag tree; big-endian structure with little-endian samples; dm4's 64-bit counts; 2-D and 3-D stacks. The embedded thumbnail is identified from the file's own Thumbnails group and skipped, so image 0 is the acquisition. |
| `vsi` | ✓ | — | Olympus / Evident CellSens (`.vsi` + `.ets`) | Read-only. The `.vsi` is an index; the pixels are in sibling `_NAME_/stackN/frame_t*.ets` files, each holding one image as a tiled JPEG pyramid. `open_pyramid` decodes only the tiles a region covers: a 256x256 window of an 8022x9367 slide moves 0.5 MB of a 32.6 MB file over HTTP. Separate stacks are separate images, so a multi-stack `.vsi` needs `stack=`. Clean-room parser, no GPL reader consulted. |
| `emd` | ✓ | — | EMD (Berkeley/NCEM and Thermo Velox) | Read-only, via `opencodecs.EmdFile`. Two conventions share the extension, so the schema is detected from the structure rather than the filename. Berkeley `dimN` axis vectors are returned alongside the array; Velox JSON metadata is decoded. Arrays come back in stored order, so hyperspy's are the transpose. |

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

`czi` decodes compression types 0 (uncompressed), 5 (zstd), 6 (ZSTDHDR)
and 4 (JPEG XR). JPEG XR, which most Zeiss slide scans use, needs the
optional `_jpegxr` extension, built when jxrlib is installed (Homebrew
`jxrlib`, conda-forge `jxrlib`, Debian/Ubuntu `libjxr-dev`). The wheels
carry it from 0.3.0 on, statically linked on every platform. A JPEG XR
sub-block without it raises `CziError`. Pyramid levels are placed from the slide origin and scale, and
overlapping mosaic tiles compose with the higher mosaic index on top, as
libCZI and czifile do. The reader exposes `metadata_bytes` and
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

### TIFF against tifffile

Reading the same files, all written by tifffile, and requiring identical
pixels from both. "1 reader" is each library's default call, made from
the main thread as a script makes it; "1 thread" forces both to a single
thread (`numthreads=1`, `maxworkers=1`); "8 readers" is eight threads each
reading at once, both on defaults. Each
number is how many times faster opencodecs is, the median over fresh
processes alternating the two libraries, on a 20-core Apple silicon Mac
(tifffile 2026.3.3) and a 64-core x86-64 Linux workstation (tifffile
2026.5.2), with opencodecs 0.5.0 built as its CI wheels are.

| 4096 x 4096 uint16 | Mac, 1 reader | Mac, 1 thread | Mac, 8 readers | Linux, 1 reader | Linux, 1 thread | Linux, 8 readers |
|---|---:|---:|---:|---:|---:|---:|
| Tiled, deflate + predictor | 3.3x | 1.1x | 3.9x | 4.2x | 1.4x | 10.0x |
| Tiled, zstd | 4.1x | 1.1x | 5.0x | 4.9x | 1.7x | 9.3x |
| Tiled, LZW + predictor | 3.9x | 2.2x | 4.7x | 4.0x | 2.4x | 9.6x |
| Strips, deflate + predictor | 2.0x | 1.0x | 1.9x | 2.9x | 1.4x | 5.0x |
| Strips, uncompressed | 3.2x | 2.3x | 1.5x | 3.6x | 0.95x (slower) | 0.93x (slower) |
| 3072 x 3072 uint8, 144 tiles, deflate | 2.2x | 1.0x | 4.6x | 4.4x | 1.3x | 13.2x |

A tiled deflate image with the predictor takes 6.1 ms against 20.0 ms on
the Mac and 10.0 ms against 42.4 ms on Linux. Most of the gap is how the
work is spread: a batch of tiles decompresses, has its predictor undone
and lands in the output in one native call that releases the GIL once,
so threads spend their time decoding rather than queuing for it, and
concurrent readers share the machine instead of each taking all of it.
Uncompressed strips are a copy for both libraries; opencodecs splits it
into positioned reads on several threads, and for one thread's worth
does whichever its kernel does faster: macOS copies out of the file
mapping, Linux reads. The two Linux cells below 1x are that copy, and a
tie: alternating in one process, one thread reads the 32 MB in 9.27 ms
with opencodecs and 9.32 ms with tifffile, against 8.84 ms for a bare
`readinto` of a new array, so neither library has anything left to
gain there; across separate processes each cell's interval reaches 1.0.

### Codecs against imagecodecs

Each row is one codec operation at the same settings in both packages:
opencodecs 0.5.0 against imagecodecs 2026.8.16, each otherwise called with
its defaults, on a 20-core Apple silicon Mac, a 64-core x86-64 Linux
workstation and a 4-core x86-64 Windows laptop. On Linux every process is
pinned to one of the processor's 16 core groups (4 cores, 8 hardware
threads, one L3 cache): unpinned, a process lands on a different group
from run to run and one-thread timings move by up to 20% with it. The
threaded rows on Linux therefore use at most 8 hardware threads. Each cell is how many times
faster opencodecs is (imagecodecs' time over opencodecs'), the median over
fresh processes that alternate the two packages; below 1.00x, imagecodecs
is faster. Both decoders read the same bitstream and must return identical
output, and each encoder's output must decode back to its input where the
setting is lossless. The input is a 4096 x 4096 uint16 image with smooth
structure and noise (32 MB as bytes, 8 MB for lzma and bz2, 16 MB as uint8
for LZW and PackBits) and 2048 x 2048 crops of it as RGB uint8, uint16 and
float32. "(opencodecs threaded)" marks a default call that ran on several
threads where imagecodecs' ran on one.

| Codec | Settings | Operation | Mac | Linux | Windows |
|---|---|---|---:|---:|---:|
| **Compression** | | | | | |
| zstd | level 3 | encode | 0.95x | 0.99x | 0.99x |
| zstd | level 3 | decode | 0.96x | 1.18x | 0.81x |
| deflate | zlib stream, level 6 | encode | 2.69x | 2.58x | 2.61x |
| deflate | zlib stream, level 6 | decode | 1.90x | 1.54x | 1.67x |
| lz4 | frame format, default level | encode | 0.99x | 1.00x | 1.02x |
| lz4 | frame format, default level | decode | 1.25x | 1.00x | 1.00x |
| brotli | level 4 | encode | 0.93x | 0.85x | 1.02x |
| brotli | level 4 | decode | 1.02x | 0.97x | 0.94x |
| blosc2 | zstd, level 5, byte shuffle | encode | 0.89x | 0.96x | 0.87x |
| blosc2 | zstd, level 5, byte shuffle | decode | 0.87x | 1.02x | 0.87x |
| lzma | level 6 | encode | 0.93x | 0.98x | 1.05x |
| lzma | level 6 | decode | 0.97x | 0.98x | 0.85x |
| bz2 | level 9 | encode | 1.00x | 1.00x | 1.00x |
| bz2 | level 9 | decode | 1.00x | 1.00x | 0.99x |
| snappy |  | encode | 1.00x | 1.00x | 1.01x |
| snappy |  | decode | 1.26x | 1.00x | 1.01x |
| **TIFF compression** | | | | | |
| LZW | TIFF flavor | encode | 1.33x | 1.32x | 1.49x |
| LZW | TIFF flavor | decode | 3.33x | 3.07x | 2.94x |
| PackBits |  | decode | 3.32x | 2.81x | 1.52x |
| **Filters** | | | | | |
| delta | uint16, distance 1 | decode | 1.09x | 1.17x | 1.05x |
| XOR | uint16, distance 1 | decode | 1.08x | 1.28x | 0.93x |
| bitshuffle | 2-byte items | encode | 1.00x | 0.99x | 1.91x |
| bitshuffle | 2-byte items | decode | 1.00x | 0.99x | 1.75x |
| packed integers | 12-bit into uint16 | decode | 1.20x | 0.82x | 2.61x |
| **Image formats** | | | | | |
| PNG | RGB uint8, level 6 | encode | 3.45x | 3.68x | 1.12x |
| PNG | RGB uint8, level 6 | decode | 1.69x | 0.99x | 0.80x |
| JPEG | RGB uint8, quality 90 | encode | 2.68x | 3.91x | 0.97x |
| JPEG | RGB uint8, quality 90 | decode | 1.56x | 1.80x | 1.01x |
| WebP | RGB uint8, lossy quality 75, method 4 | encode | 0.92x | 1.03x | 0.98x |
| WebP | RGB uint8, lossy quality 75, method 4 | decode | 0.98x | 1.00x | 1.00x |
| QOI | RGB uint8 | encode | 0.99x | 1.00x | 1.02x |
| QOI | RGB uint8 | decode | 1.01x | 1.00x | 0.96x |
| JPEG 2000 | uint16, lossless | encode | 8.05x (opencodecs threaded) | 4.66x (opencodecs threaded) | 2.73x (opencodecs threaded) |
| JPEG 2000 | uint16, lossless | decode | 7.64x (opencodecs threaded) | 4.57x (opencodecs threaded) | 2.71x (opencodecs threaded) |
| JPEG-LS | uint16, lossless | encode | 0.95x | 1.00x | 1.08x |
| JPEG-LS | uint16, lossless | decode | 0.97x | 1.01x | 1.01x |
| JPEG XL | RGB uint8, distance 1, effort 5 | encode | 5.68x (opencodecs threaded) | 2.30x (opencodecs threaded) | 3.42x (opencodecs threaded) |
| JPEG XL | RGB uint8, distance 1, effort 5 | decode | 5.10x (opencodecs threaded) | 2.68x (opencodecs threaded) * | 3.81x (opencodecs threaded) |
| JPEG XL | uint16, lossless, effort 3 | encode | 10x (opencodecs threaded) | 2.46x (opencodecs threaded) | 3.03x (opencodecs threaded) |
| JPEG XL | uint16, lossless, effort 3 | decode | 6.44x (opencodecs threaded) | 4.11x (opencodecs threaded) | 3.79x (opencodecs threaded) |
| LERC | uint16, lossless | encode | 1.29x | 1.78x | 1.07x |
| LERC | uint16, lossless | decode | 0.92x | 0.96x | 0.95x |
| ZFP | float32, reversible | encode | 0.72x | 0.99x | 0.70x |
| ZFP | float32, reversible | decode | 0.95x | 1.07x | 1.03x |
| BC1 | 2048 x 2048, decode to RGBA | decode | 4.45x (opencodecs threaded) | 1.79x (opencodecs threaded) | 2.27x (opencodecs threaded) |
| BC7 | 2048 x 2048, decode to RGBA | decode | 11x (opencodecs threaded) | 5.47x (opencodecs threaded) | 4.09x (opencodecs threaded) |

\* Both packages use libjxl 0.12.0, but on Linux their lossy decodes of
the same file differ by 1 in about a third of the output values (never
more); on macOS and Windows they are identical. Lossless decodes match
everywhere.

opencodecs is ahead where it runs its own kernels (LZW, PackBits, BC1 and
BC7, and bitshuffle on Windows), builds PNG and deflate on libdeflate, and
decodes JPEG 2000 and JPEG XL on several threads by default. Where both
packages wrap the same library, some rows trail by more than 9%: blosc2
and ZFP encode on the Mac and Windows, zstd, lzma and PNG decode on
Windows, and brotli encode on Linux. Packed-integer decode, our own
kernel, trails on Linux. No other row is more than 9% behind. Regenerate the table with
`bench/bench_vs_imagecodecs.py`; the per-run medians are in
`bench/results/vs_imagecodecs/`.

### CZI against czifile and aicspylibczi

Scientific microscopy CZI (66 MB, 14 sub-blocks of 2000×2000 uint16,
ZSTDHDR), single-file warm cache:

| Reader        | Apple silicon Mac | x86-64 Linux |
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

Any of these accept an `http(s)` URL wherever they accept a path.
Formats that can reach storage by offset fetch only the bytes they
need by Range request; the whole-codestream formats fetch once, which
is the honest thing when every byte is needed anyway.

Which is which is not a list to keep in your head, or in this README
where it would rot: `capabilities.toml` records it per codec and
`ci/check_capabilities.py verify` re-derives every entry from the code
on each CI run, so it cannot quietly stop being true.

```python
import tomllib
caps = tomllib.load(open("capabilities.toml", "rb"))["codec"]
[c["name"] for c in caps if c["http"]]      # range-backed over HTTP
[c["name"] for c in caps if c["pyramid"]]   # codecs with a pyramid reader
```

The runtime manifest covers the codec registry. The companion
[pipeline catalog](pipeline_catalog.toml) also covers direct adapters such as
Imaris, OME-Zarr, NDTiff and N5, and maps shared optimization work across all
registered codecs. See the [implementation plan](docs/pipeline_optimization_plan.md)
for priorities, source evidence, correctness prerequisites and acceptance tests.
Run `python ci/check_pipeline_catalog.py report` for the current worklist.

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

`open_pyramid` dispatches on extension for TIFF/COG/SVS, OME-Zarr,
CZI, Imaris, JPEG, JPEG 2000 and HTJ2K, and two whole-slide formats
whose pyramid is not one file:

```python
# Olympus / Evident CellSens. The .vsi is an index; the tiles live in
# a sibling _NAME_/stackN/frame_t*.ets. Only the tiles the box covers
# are decoded.
with oc.open_pyramid("slide.vsi") as p:
    p.shapes            # ((9367, 8022, 3), (4684, 4011, 3), ... 6 levels)
    tile = p.read_region(0, y=(1000, 1256), x=(2000, 2256))

# DICOM VL Whole Slide Microscopy. A slide is a SERIES of instances,
# one per resolution, so this takes a directory rather than a file --
# no extension can imply that, hence the explicit format=.
with oc.open_pyramid("study/slide_dir", format="dicom") as p:
    overview = p.read_region(p.best_level_for(max_pixels_y=1024))
```

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

Wheels are published for CPython 3.10 to 3.13 on macOS 15+ (arm64),
Linux (x86_64 + aarch64, manylinux_2_28), and Windows (amd64). Every
platform ships the same 39 compiled extensions, bundling libjxl,
libavif, libheif, libwebp, libdeflate, c-blosc2, CharLS and friends,
so no system dependencies are needed. Intel Macs and macOS releases
older than 15 install from the sdist instead.

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

- **v0.5.0** on PyPI (October 2026). Every wheel carries the same 40
  compiled extensions, and `ci/check_wheel_contents.py` fails the
  release build if one goes missing.
- About 4,000 tests locally, including a 40-dataset conformance corpus;
  CI runs the corpus-independent suite on macOS, Linux and Windows for
  Python 3.10 and 3.13.
- Native readers and writers for the common scientific containers
  (TIFF, BigTIFF, OME-TIFF, CZI, NDTiff, HDF5, JXL, FITS, OME-Zarr v2 +
  v3 sharded), and pyramid readers for TIFF/COG/SVS, OME-Zarr, CZI,
  Imaris, DICOM VL Whole Slide Microscopy and Olympus VSI/ETS
- Shared source adapters support paths, buffers, file objects and remote URLs.
  Indexed readers use checked range requests, covering caches and adaptive
  read-ahead; whole-stream codecs retain their documented eager input paths.
- Multi-frame AVIF, animated WebP and GIF decode to a frame stack; HEIF
  exposes every top-level image
- Compression backend auto-detect (libdeflate → zlib-ng-compat → stdlib)
- `capabilities.toml` records what each codec actually supports, checked
  against the built extensions, with no open gaps

Deferred work (see [`docs/TODO_DEFERRED.md`](docs/TODO_DEFERRED.md)):

- CCITT Fax3/Fax4 encode: legacy fax, zero scientific users
- JPEG XR encode (decode exists for CZI; see above)

GIF and JPEG XL destination writers now drain encoded frame output before close.
The shared pipeline also supplies bounded independent-piece scheduling, seven
stateful byte-codec iterators, native PNG row sessions, selective native sources,
and checked caller-owned destinations.

Shared streaming, buffering, and overlap mechanisms and remaining integration
work are documented in [the pipeline catalog](docs/shared_pipeline_capabilities.md).

See [CHANGES.rst](CHANGES.rst) for release history.

## License

BSD-3-Clause; see [LICENSE](LICENSE).

Vendored source, the Cython declaration files derived from
[imagecodecs](https://github.com/cgohlke/imagecodecs) (BSD-3-Clause,
Copyright (c) 2008-2026 Christoph Gohlke), and the codec libraries
bundled into the binary wheels each retain their own license. The full
inventory is in [THIRD-PARTY.md](THIRD-PARTY.md).
