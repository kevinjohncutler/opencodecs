# Opt-in hardware backends

Three codec backends hand the work to dedicated hardware. None is ever
chosen by default: a call opts in with `backend=`, the same keyword
`deflate` uses for ISA-L, and nothing changes for code that does not.

| `backend=` | Codecs | Runs on | Needs |
|---|---|---|---|
| `"nvimgcodec"` | `jpeg`, `jpeg2k`, `htj2k`: encode and decode | NVIDIA GPU | CuPy and nvImageCodec (`pip install 'opencodecs[gpu]'`) |
| `"nvvideocodec"` | `heif`: decode and a lossy fast encode; `avif`: lossy fast encode | NVIDIA video engines (NVDEC, NVENC) | CuPy and PyNvVideoCodec (`pip install 'opencodecs[gpu]'`) |
| `"imageio"` | `heif`: decode, and a lossy fast encode | Apple media engine | macOS; no package (ImageIO through ctypes) |

```python
import opencodecs as oc

htj2k = oc.get_codec("htj2k")
img = htj2k.decode(blob, backend="nvimgcodec")          # same array as the CPU path
blob = htj2k.encode(img, backend="nvimgcodec")          # lossless by default

photo = oc.read("scan.heic", backend="nvvideocodec")    # NVDEC, same pixels as libheif
photo = oc.read("scan.heic", backend="imageio")         # macOS
fast = oc.write(None, photo, format="heif", level=80, backend="imageio")
```

`backend=None` or `"native"` is the CPU path. An unknown name raises
`ValueError`. A backend that cannot run here (package missing, no CUDA
device, not macOS) raises `opencodecs.backends.BackendUnavailable`, a
subclass of both `ImportError` and `OpenCodecsError`; there is no silent
fallback to the CPU. `opencodecs.backends.available(name)` asks first.

## Why opt-in only

The first nvImageCodec call in a process pays for importing CuPy and
nvImageCodec, creating the CUDA context and the decoder: 1.08 s before
the first HTJ2K decode returned, and the first JPEG decode after that
another 0.58 s (nvJPEG's own setup). The first NVDEC decode took 0.8 s
and the first NVENC encode 1.55 s. A script that decodes one image is
much slower with the GPU. Each backend is created once per process,
lazily, and reused, so the cost is paid once.

Importing `opencodecs` imports none of CuPy, nvImageCodec,
PyNvVideoCodec or any Apple framework (`tests/test_backend_selection.py` checks this in a fresh
interpreter); they load on the first call that asks for them.

There is no environment variable that switches the default either. Many
readers decode JPEG or JPEG 2000 internally, one tile at a time (TIFF,
CZI, DICOM, zarr chunks), and a batch of small tiles is exactly where
the GPU loses (below), so a process-wide switch would slow down the
workloads it reached most often.

## NVIDIA GPU: `backend="nvimgcodec"`

nvJPEG2000 decodes and encodes JPEG 2000 and HTJ2K on the GPU. JPEG goes
through nvJPEG: the hardware JPEG engine where the GPU has one (A100,
H100), otherwise the hybrid decoder, which does Huffman decoding on the
CPU and the rest on the GPU. Consumer GPUs have no JPEG engine, and
nvImageCodec's GPU-only JPEG decoder returns nothing on them. Its CPU
decoders are never enabled.

What comes back:

* The dtype and shape the CPU decoder returns, checked against the
  codestream header before decoding. For lossless JPEG 2000 and HTJ2K
  the pixels are the same, bit for bit (gray, RGB, RGBA; uint8, uint16,
  int16; 12-bit samples in uint16).
* Lossy JPEG 2000 within 2 levels of OpenJPEG. JPEG within 3 levels of
  libjpeg-turbo (IDCT and chroma-upsampling rounding).
* Encodes that every other decoder reads: lossless round trips exactly
  through OpenJPEG, OpenJPH and imagecodecs; JPEG quality and size match
  libjpeg-turbo's at the same `level` (within 0.05 dB and 2% in bytes,
  measured at level 90).
* `out=` takes a numpy array, a CuPy array (the pixels then stay on the
  GPU) or page-locked memory from `opencodecs.backends.pinned_empty`,
  which the GPU copies into faster. Encodes take numpy or CuPy arrays.

What it refuses (raises, so the caller can use `backend=None`):
`reduce=` and region options, JPEG `scale=`/`colorspace=` and the other
decode options, 12-bit and lossless (SOF3) JPEG, CMYK JPEG, HTJ2K float
samples (NLT) or mixed precision, two-sample JP2, JPEG 2000 `ratio=`
(nvJPEG2000 0.9 has no rate target), an HTJ2K quantization step
(`level` below 1), and lossless JPEG encode.

### Measured on an RTX 4090

RTX 4090 (CUDA 13, nvImageCodec 0.9.0, CuPy 14.2) against a 64-core
AMD Zen 2 workstation CPU, end to end through `decode()` / `encode()`:
bytes in, numpy array out, or the reverse. Medians of 11 interleaved
runs after warm-up, in milliseconds. The CPU column is the opencodecs
0.7.0 wheel with its default threads.

| Case (4096 x 4096) | CPU | GPU | GPU, pinned `out=` | Speedup |
|---|---|---|---|---|
| HTJ2K lossless decode, uint16 | 93.1 | 10.3 | 5.8 | 9.0x (16x pinned) |
| JPEG 2000 lossless decode, uint16 | 78.3 | 27.9 | 23.7 | 2.8x (3.3x pinned) |
| JPEG decode, RGB q90 | 51.4 | 23.5 | 16.7 | 2.2x (3.1x pinned) |
| JPEG 2000 lossless encode, uint16 | 96.4 | 54.6 | | 1.8x |
| HTJ2K lossless encode, uint16 | 102.5 | 26.5 | | 3.9x |
| JPEG encode, RGB q90 | 49.0 | 10.2 | | 4.8x |

Against the CPU build in the same process (an older OpenJPH, and a
different OpenJPEG thread default) the ratios were higher: 12x/22x,
5.3x/6.2x, 2.3x/3.3x, 1.9x, 7.2x and 4.0x. An earlier engine-level
measurement on the same machine found 8.5x for HTJ2K decode (14.7x
pinned) and 5.3 to 6.4x for JPEG 2000 decode. Treat the ratios as a
range; the CPU side varies with build and thread count.

Where it does not help:

* One image per process, because of the 1 to 1.6 s startup.
* Many small tiles. 64 JPEG tiles of 512 x 512 took 15 ms as one GPU
  batch against 3.7 ms on 32 CPU threads; this backend decodes one
  stream per call and has no batch API on purpose.
* Small images generally: the fixed per-call cost (header parse, launch,
  copy back) is a few milliseconds.

Inputs: a smooth RGB image with mild noise (JPEG at quality 90) and a
microscopy-like uint16 image (dark background, bright structures,
shot-like noise), both 4096 x 4096.

## NVIDIA video engines: `backend="nvvideocodec"`

NVIDIA GPUs carry fixed-function video engines next to the CUDA cores:
NVDEC decodes HEVC, the codec inside HEIF, and NVENC encodes HEVC and
(from the Ada generation on) AV1, the codec inside AVIF.
`backend="nvvideocodec"` uses them for still images:

* `heif` decode: the primary image on NVDEC, one coded picture or every
  tile of a grid, 8 to 12 bits, in the chroma formats the GPU decodes;
* `heif` encode: lossy HEVC 4:2:0 on NVENC, 8 or 10 bits;
* `avif` encode: lossy AV1 4:2:0 on NVENC, 8 or 10 bits.

```python
img = oc.read("photo.heic", backend="nvvideocodec")
blob = oc.get_codec("avif").encode(img, level=60, backend="nvvideocodec")
```

### How it reaches the hardware

Through NVIDIA's PyNvVideoCodec (`pip install PyNvVideoCodec`, in the
`gpu` extra), the Video Codec SDK's own Python package. It talks to the
driver's NVDEC/NVENC libraries directly and ships wheels for Python 3.10
to 3.14 on Linux x86_64 and aarch64 and Windows x64 and arm64, so nothing
is built against a system FFmpeg. The other two routes considered:

* libavcodec through ctypes, as the engine-level benchmark did. It needs
  an FFmpeg built with NVDEC/NVENC, which pip does not provide (Linux
  distributions do, Windows does not), and that FFmpeg's swscale returned
  wrong pixels for subsampled deeper-than-8-bit to RGB48.
* A small C extension against the SDK headers. It would have to be built
  into every wheel for a feature few users enable, and would still need
  the RGB conversion kernels.

PyNvVideoCodec is not slower: decoding the same 4096 x 3072 8-bit 4:2:0
picture took 4.3 ms on the GPU through it (RTX 4090), where the libavcodec
route took 4.2 ms of hardware time on an RTX 5060 Ti. The rest of the
pipeline is opencodecs': the container (read and written in Python), the
YCbCr/RGB conversions (CUDA kernels compiled by CuPy with NVRTC, which is
why the `gpu` extra includes `cuda-toolkit[cudart,nvrtc]`, about 140 MB),
and the copies.

### Decode

The coded pictures are taken out of the container, decoded on NVDEC,
converted to RGB on the GPU and copied to the host through a reused
page-locked buffer (a fresh pageable buffer per call copied at about a
tenth of the speed). Grid tiles go to one decoder session in a row and
are placed into one canvas, cropped to the grid's size.

What comes back is libheif's result, bit for bit: same dtype (uint8, or
uint16 holding 10 or 12-bit values), same shape, same pixels. The
conversion reproduces libheif 1.21's own choice and arithmetic: the
container's `colr` nclx, else the HEVC stream's VUI, for the matrix and
range; libheif's fixed-point path for 8-bit 4:2:0 at full range and its
floating-point path otherwise (with fused multiply-add off, as on the
CPU); nearest-neighbor chroma. Checked on 8 and 10-bit, 4:2:0 and 4:4:4,
full and limited range, BT.601, BT.709 and BT.2020 matrices, the identity
matrix of lossless files, and 512 x 512 grids: the maximum difference
from `backend=None` was 0 on every one.

Files go to libheif instead, with the same result as `backend=None`,
when the GPU would return something else or cannot decode them: an
`index=` or `photometric=`, alpha, monochrome, a rotation, mirror or crop
property (odd-sized images written by libheif carry a crop), a chroma
format, depth or picture size the GPU does not decode (asked of the
driver at run time; NVDEC's smallest HEVC picture is 144 x 144), or the
YCgCo and other matrices libheif treats as special cases.
`opencodecs.backends._nvvideocodec.route(data)` says which path a file
takes and why, and the decision is logged at debug level.

`out=` takes a numpy array, a CuPy array (the pixels then stay on the
GPU and never cross the bus) or page-locked memory from
`opencodecs.backends.pinned_empty`.

### Encode

`level` (0 to 99) is required, since both codecs' default is lossless,
which these encoders do not do. For AVIF it maps to the AV1 quantizer
libavif gives aom for the same level; for HEIF to a constant HEVC QP,
`51 - level * 51 / 100`. Input is (H, W, 3) RGB: uint8 for 8 bits, or
uint16 with values up to 1023 for 10 bits (numpy or CuPy). The RGB to
YCbCr conversion (BT.601, full range, 2 x 2 chroma averaging) runs on
the GPU. `speed` (AVIF) picks the NVENC preset; `iccprofile` is written
as a second `colr` box. Tile, codec, color and 4:4:4 options raise.

The container is written in Python around the encoder's output, with
`ispe`, `pixi`, `colr` (nclx, plus the ICC profile if given) and the
`av1C` or `hvcC` record. The coded stream is corrected where NVENC's
headers would mislead a reader: the AV1 sequence header and the HEVC
SPS get the color description the pixels were converted with (NVENC
writes none, so FFmpeg took the HEVC files as limited range: 31.7 dB
before, 42.4 dB after), and the HEVC SPS loses its HRD buffering
parameters, whose values above 8.9 megapixels libde265 (the HEVC decoder
in most libheif builds, opencodecs' wheels included) refuses as out of
range. An odd-sized HEIF is padded to even and cropped back with a
`clap` box, as libheif writes them; an odd-sized AVIF raises, since
PyNvVideoCodec takes odd-sized 4:2:0 input only with its chroma
truncated, and libavif-based readers ignore `clap`.

Foreign readers checked (`tests/test_foreign_readers.py`): imagecodecs
(libavif) and FFmpeg (PyAV) for AVIF; pillow-heif (libheif 1.23 with
libde265), FFmpeg and imagecodecs' libheif where it has one for HEIF;
both at 8 and 10 bits, and odd-sized HEIF.

### Measured on an RTX 4090

RTX 4090 (Ada, driver 580, CUDA 13, PyNvVideoCodec 2.2.3, CuPy 14.2)
against a 64-core AMD Zen 2 workstation CPU, end to end through
`decode()` / `encode()`. Medians of 11 interleaved runs after warm-up, in
milliseconds. HEIF inputs are 4096 x 3072 RGB (smooth gradients and
blobs with mild noise), written by libheif 1.23.4 and x265 4.2 at
quality 50 unless noted; the CPU column is this tree's build (libheif
1.21 with libde265).

| HEIF decode, 4096 x 3072 | CPU | GPU | GPU, pinned `out=` | GPU, CuPy `out=` | Speedup |
|---|---|---|---|---|---|
| 8-bit 4:2:0, 98 KB | 327.3 | 10.9 | 6.5 | 4.6 | 30x |
| 8-bit 4:2:0, quality 85, 6.6 MB | 1172.3 | 41.6 | 35.2 | 32.7 | 28x |
| 8-bit 4:4:4, 78 KB | 500.7 | 15.2 | 11.6 | 9.1 | 33x |
| 8-bit 4:4:4, opencodecs level 90, 10.6 MB | 1934.9 | 63.5 | 54.2 | 51.9 | 30x |
| 10-bit 4:2:0 | 366.2 | 14.6 | 9.0 | 4.6 | 25x |
| 10-bit 4:4:4 | 625.8 | 19.2 | 14.2 | 9.7 | 33x |
| 48 tiles of 512 x 512, 8-bit 4:2:0 | 117.1 | 19.4 | 16.2 | 13.5 | 6.0x |
| 48 tiles of 512 x 512, 8-bit 4:4:4 | 135.0 | 23.5 | 21.2 | 18.3 | 5.7x |
| 48 tiles of 512 x 512, 10-bit 4:2:0 | 170.1 | 22.8 | 18.0 | 13.2 | 7.5x |

libde265 did not divide these untiled pictures between threads on this
machine (`numthreads` from 1 to 64 measured the same, as did pillow-heif),
so the CPU column is close to single-threaded, which inflates the ratios.
On the RTX 5060 Ti machine, where libheif's threaded decode took the same
8-bit 4:2:0 file in 66 ms, the engine-level benchmark found the GPU 5 to
8 times faster. Of the 10.9 ms, the hardware decode and conversion take
4.4, the copy to page-locked memory 1.7, and the copy into a new numpy
array (split over 8 threads) about 2.

| Encode, 4096 x 3072, 8-bit | CPU | NVENC | Size, CPU vs NVENC | PSNR, CPU vs NVENC |
|---|---|---|---|---|
| AVIF level 30 (aom) | 5969 | 13.6 | 8.7 KB vs 11.2 KB | 42.19 vs 41.53 dB |
| AVIF level 50 | 5723 | 13.9 | 15.1 KB vs 18.8 KB | 43.31 vs 42.75 dB |
| AVIF level 60 | 5673 | 13.6 | 21.3 KB vs 26.3 KB | 43.60 vs 43.19 dB |
| AVIF level 80 | 6257 | 13.9 | 39.9 KB vs 57.0 KB | 43.83 vs 43.55 dB |
| AVIF level 90 | 9985 | 14.0 | 85 KB vs 378 KB | 43.94 vs 43.74 dB |
| HEIF level 30 (x265, 4:4:4) | 943 | 37.0 | 32 KB vs 17 KB | 42.67 vs 41.41 dB |
| HEIF level 50 | 1090 | 37.4 | 78 KB vs 33 KB | 43.53 vs 42.99 dB |
| HEIF level 70 | 2948 | 39.6 | 2.96 MB vs 0.62 MB | 44.63 vs 43.55 dB |
| HEIF level 90 | 5144 | 46.4 | 10.6 MB vs 4.3 MB | 50.01 vs 45.02 dB |

AV1 on NVENC is 400 to 700 times faster than aom (libavif 1.4.1, aom
3.9.1, its default speed) but not as efficient: at the same level its
files are 1.2 to 1.4 times larger and 0.2 to 0.7 dB worse, and at equal
PSNR about 1.8 times larger near 43.2 dB and 2.7 times larger near
43.6 dB (NVENC's `uhq` tuning, which might narrow this, is refused for
AV1 on Ada). HEVC on NVENC is 25 to 110 times faster than x265 through
the CPU path (which codes 4:4:4); against x265 at 4:2:0 its files are
about the same size at 43 dB and 6 to 8 times larger from 43.5 dB up.
10-bit: AVIF level 60 in 14 ms (24.5 KB, 43.96 dB at a 1023 peak), HEIF
level 60 in 45 ms (82 KB, 44.1 dB). On this noisy test content PSNR
saturates near 44 dB, which exaggerates the size gap at high levels.

### Startup, and when it helps

In a fresh process (CuPy's compiled kernels already in its on-disk
cache; the very first run in a new environment also compiles them), the
first HEIF decode returned 0.8 s after the call, against 0.33 s for the
CPU path, and later decodes took 10 ms. The first AVIF or HEIF encode
returned after 1.55 s, most of it creating the NVENC session (about
0.6 s), against 5.7 s for one aom encode. Sessions are kept for later
calls (one decoder per chroma format and depth, reconfigured for each
size; the last encoder) and closed explicitly at interpreter exit or by
`opencodecs.backends._nvvideocodec.close()`: an NVENC session left to
the interpreter's own teardown once hung a process for 17 minutes.

So: one AVIF encode already pays; HEIF decode and encode pay from the
second image on. Grids of small tiles gain less (5 to 7x here) than one
large coded picture, but still gain on this machine, unlike on Apple's
engine. Images below NVDEC's 144 x 144 go to libheif, and NVENC's
smallest pictures are 192 x 128 (AV1) and 129 x 33 (HEVC). Decoding many
images from several threads is serialized on one GPU session per process
here; the engine-level benchmark found a pool of CPU threads faster than
NVDEC for batches of small images.

### Formats by GPU generation

What a GPU decodes is asked of the driver at run time, and a file it
cannot take goes to libheif; encoding a format the GPU lacks raises.

| | RTX 4090 (Ada), measured here | RTX 5060 Ti (Blackwell), engine-level benchmark only |
|---|---|---|
| HEVC decode 4:2:0, 4:4:4, 8 to 12 bits | yes (8 and 10 measured, 12 per the driver) | yes |
| HEVC decode 4:2:2 | no (goes to libheif) | yes |
| HEVC decode monochrome | no (goes to libheif) | no |
| HEVC encode 4:2:0, 8 and 10 bits | yes | yes (8-bit measured) |
| AV1 encode 4:2:0, 8 and 10 bits | yes | yes (8-bit measured) |

This backend itself has been run on Ada only; the Blackwell column is
the driver's capability report from the earlier standalone benchmark.
AV1 encoding needs Ada or newer (NVIDIA's support matrix); on older GPUs
`avif` encode with this backend raises `BackendUnavailable`.

## Apple ImageIO: `backend="imageio"`

### Decode

ImageIO hands HEVC decoding to the media engine. On one large coded
picture it is far faster than libde265; on a grid of small tiles, which
is how iPhones and ImageIO itself write HEIC, it is not, because libheif
decodes the tiles on several threads. So `backend="imageio"` sends a file
to ImageIO only when it helps and the result keeps the CPU path's layout:

* the primary image (no `index=`, no `photometric=`),
* one coded picture, or a grid whose tiles are at least 1024 pixels on
  each side,
* 8-bit color (not monochrome), no alpha plane,
* no rotation, mirror or crop property.

Every other file is decoded by libheif exactly as with `backend=None`;
`opencodecs.backends._imageio.route(data)` returns which path a file
takes and why, and the decision is logged at debug level on the
`"opencodecs"` logger. Off macOS the backend raises `BackendUnavailable`
rather than falling back.

The output is `(H, W, 3)` uint8, as from libheif. The pixels are the
decoder's own (`CGDataProviderCopyData`; drawing into a bitmap context
changed some). ImageIO returns four bytes a pixel, and Accelerate's
vImage drops the fourth into the output array, which takes about 3 ms
for 4096 x 3072 against 55 to 75 ms for the same copy in numpy. `out=`
is written in place.

Agreement with libheif: exact for lossless files; within 1 level for
4:4:4. For 4:2:0 the two decoders upsample chroma differently: up to 4
to 6 levels on photographic content (mean 0.2 to 0.4), and up to 50 at
isolated pixels of noisy, saturated synthetic color.

### Encode

`encode(..., backend="imageio")` uses the hardware HEVC encoder. It is
lossy only, 8-bit 4:2:0, written in 512 x 512 tiles; it takes `(H, W, 3)`
uint8 and needs an explicit `level` (the codec's default is lossless, so
a bare call raises). `level` is passed as ImageIO's quality, `level/100`,
which is not on libheif's scale: ImageIO 80 and libheif 50 give about the
same PSNR. Files are larger at equal quality, so use it when encode time
matters more than size. imagecodecs and pillow-heif read its files
(`tests/test_foreign_readers.py`).

### Measured on an M1 Ultra

M1 Ultra (20 cores), macOS 26, 4096 x 3072 RGB, end to end through
`decode()` / `encode()`, medians in milliseconds.

| Case | libheif | ImageIO | Speedup |
|---|---|---|---|
| Decode, one coded picture, 4:4:4, 77 KB (libheif level 50) | 281.8 | 27.1 | 10.4x |
| Decode, one coded picture, 4:4:4, 3.1 MB (level 70) | 953.9 | 49.3 | 19.4x |
| Decode, 48 tiles of 512 x 512 (ImageIO's own file) | 82.4 | 82.7 (libheif) | 1.0x |
| Encode, equal PSNR 43.5 dB: libheif level 50 vs ImageIO level 80 | 715.3 | 132.8 | 5.4x, file 3.2x larger |
| Encode, equal PSNR 44.7 dB: libheif level 70 vs ImageIO level 95 | 2060.3 | 133.7 | 15.4x, file 1.4x larger |

The tiled file goes to libheif under `backend="imageio"`, so it costs
what it costs without it; with `numthreads=20` libheif decodes it in
45 ms, where ImageIO took 83 to 93 ms in the engine-level measurement.
`numthreads=` is the lever for tiled files, not this backend.

Startup is small: loading the frameworks through ctypes takes under
1 ms (importing pyobjc's Quartz bindings took 140 ms, one reason this
backend does not use them), and the first decode in a process took
147 ms, about 120 ms more than later ones, for ImageIO's own first use.

Inputs: a smooth 4096 x 3072 RGB image (gradients, sinusoids, blobs and
mild noise), written by libheif at levels 50 and 70 and by ImageIO at 80.
