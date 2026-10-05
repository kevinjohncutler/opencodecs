# Opt-in hardware backends

Two codec backends hand the work to dedicated hardware. Neither is ever
chosen by default: a call opts in with `backend=`, the same keyword
`deflate` uses for ISA-L, and nothing changes for code that does not.

| `backend=` | Codecs | Runs on | Needs |
|---|---|---|---|
| `"nvimgcodec"` | `jpeg`, `jpeg2k`, `htj2k`: encode and decode | NVIDIA GPU | CuPy and nvImageCodec (`pip install 'opencodecs[gpu]'`) |
| `"imageio"` | `heif`: decode, and a lossy fast encode | Apple media engine | macOS; no package (ImageIO through ctypes) |

```python
import opencodecs as oc

htj2k = oc.get_codec("htj2k")
img = htj2k.decode(blob, backend="nvimgcodec")          # same array as the CPU path
blob = htj2k.encode(img, backend="nvimgcodec")          # lossless by default

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
another 0.58 s (nvJPEG's own setup). A script that decodes one image is
much slower with the GPU. Each backend is created once per process,
lazily, and reused, so the cost is paid once.

Importing `opencodecs` imports none of CuPy, nvImageCodec or any Apple
framework (`tests/test_backend_selection.py` checks this in a fresh
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
