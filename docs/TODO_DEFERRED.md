# Deferred codec / format items

Each entry below is a real follow-up from the opencodecs roadmap that
needs more than an overnight slot to land cleanly. They're documented
here so the next person picks up the work with full context.

Items shipped from the same roadmap pass are not relisted; see the
git log for `2026-05-11` for the night's shipped commits.

---

## BC1-7 / DDS texture decoder

* **Status**: not implemented.
* **Use case**: game-asset DDS files, S3TC / BPTC compressed textures
  in 3D engines, GPU-side compressed mipmaps.
* **Sources**: `/Volumes/HiprDrive/imagecodecs/3rdparty/bcdec/bcdec.h`
  is the single-header Sergii Kudlai implementation we'd vendor.
  MIT licensed, no external deps.
* **Sketch**:
  1. Copy `3rdparty/bcdec/{bcdec.h, bcdec_dds.h}` into
     `opencodecs/3rdparty/bcdec/`.
  2. Write `src/opencodecs/codecs/_bcdec.pyx` — thin Cython binding
     to `bcdec_bc1/bc2/bc3/bc4/bc5/bc6h/bc7` (decode only; encoding
     is rarely needed for image-codec use cases).
  3. Wire into `_registry.py` next to `qoi`.
  4. Tests: synthesize a few-block DDS file via numpy, decode + assert
     pixel-equal at known reference patterns.
* **Effort**: ~3-4 hours.

## SPERR (Scientific Error-bounded Lossy Compression)

* **Status**: not implemented.
* **Use case**: scientific lossy compression with absolute or PSNR-
  bounded error guarantees — climate/CFD/imaging data where rate-
  distortion control matters.
* **Source**: https://github.com/NCAR/SPERR (BSD-3, CMake build).
* **Sketch**:
  1. Build SPERR via CMake into `~/.cache/opencodecs/sperr/`
     (same pattern as Tier 1 SZ3 / pcodec).
  2. Write `src/opencodecs/codecs/_sperr.pyx` calling SPERR's
     C++ `SPERR3D_OMP_C/D_API` (it's C++ — need a `.cpp` shim
     exposing a C entry point, similar to `b2nd_helpers.c`).
  3. Tests: round-trip a few PSNR settings, verify error bound holds.
* **Effort**: ~4-6 hours including build automation.

## EER (cryo-EM electron event)

* **Status**: not implemented.
* **Use case**: Falcon 4 / Selectris X cryo-EM detector raw output.
* **Source**: Thermo Fisher published the EER format spec; ImageJ /
  RELION have reference readers. Wikipedia summarizes the layout:
  TIFF container with EER-specific sub-tags carrying event lists +
  per-event x/y coordinates + dose information.
* **Sketch**:
  1. Parse the TIFF wrapper via existing `_tiff_codec.py`.
  2. Add an EER-specific decoder for the event-list sub-tags
     (custom RLE-of-coordinates layout).
  3. Output: dose-corrected float32 frames at user-specified
     temporal binning.
* **Effort**: ~6-8 hours. Hard to test without a real EER fixture
  from a Falcon 4 acquisition.

## CharLS / JPEG-LS

* **Status**: not implemented.
* **Use case**: DICOM medical imaging (the lossless predictive codec
  for radiology).
* **Source**: https://github.com/team-charls/charls (BSD-3, C++).
  brew: `brew install charls`.
* **Sketch**: similar to other image codec bindings; CharLS's C++ API
  is `charls::jpegls_encoder` / `decoder`. C wrapper available via
  `<charls/charls_jpegls_encoder.h>` (provides C entry points).
* **Effort**: ~3-4 hours.

## MozJPEG

* **Status**: not implemented.
* **Use case**: drop-in JPEG encoder that produces ~10-15% smaller
  files than libjpeg-turbo at the same quality.
* **Source**: https://github.com/mozilla/mozjpeg — API-compatible with
  libjpeg, can be a build-time swap.
* **Sketch**: the cleanest path is a `compression=mozjpeg` codec name
  that we map to a separate `_mozjpeg.pyx` linking libmozjpeg
  (which has the same `jpeg_*` symbols as libjpeg, so the binding
  is nearly identical to `_jpeg.pyx`).
* **Effort**: ~2-3 hours.

## Brunsli

* **Status**: not implemented.
* **Use case**: lossless JPEG → smaller JPEG transcoder (~22% size
  reduction, fully reversible). Useful for cloud storage of large
  JPEG corpora.
* **Source**: https://github.com/google/brunsli (MIT, C++).
* **Sketch**: `brunsli::EncodeBrn` / `DecodeBrn` are the entry points.
  Binding similar to MozJPEG. Note Brunsli requires libbrotli at
  runtime — we already have it from `_brotli.pyx`.
* **Effort**: ~3-4 hours.

## DICOMweb reader

* **Status**: not implemented.
* **Use case**: stream pixel data from a DICOMweb-compliant server
  (orthanc, dcm4chee, Google Healthcare API) via WADO-RS HTTP-range
  requests.
* **Sketch**: implement a `DicomwebClient` that issues WADO-RS
  `GET /studies/{study}/series/{series}/instances/{instance}/frames/N`
  requests, parses the multipart/related response, and dispatches the
  pixel payload through `core.segment_compression` (DICOM frames can
  be JPEG-LS, JPEG-2K, RLE, deflate — all of which we already decode).
* **Effort**: ~6-8 hours, mostly DICOM transfer-syntax detection
  glue. The actual codec dispatch is reusable.

## HDF5 cloud (h5coro)

* **Status**: not implemented.
* **Use case**: NASA / SnowEx / Argo-float HDF5 archives served from
  S3 — read them without downloading the whole file via per-chunk
  HTTP Range requests.
* **Source**: https://github.com/SlideRuleEarth/h5coro (BSD-3).
  Pure C++; not currently bound for Python.
* **Sketch**: option (a) thin Cython wrapper around h5coro's C++
  classes; (b) implement a minimal HDF5 reader natively (HDF5 is a
  complex spec — option a is much smaller).
* **Effort**: ~8-12 hours for a full pyramidal-HDF5 reader; ~4-6
  hours for the subset of HDF5 we actually use.

## zlib-ng / ISA-L deflate swap

* **Status**: deferred; current `_deflate.pyx` uses system libz.
* **Use case**: ~1.5-2× speedup on deflate / gzip / PNG-encode
  byte-stream paths; matches what imagecodecs gets on
  `conda-forge` (which uses zlib-ng-compat).
* **Path A — zlib-ng-compat**: build setup.py probes for
  `-lz-ng-compat` first; if available, links it (drops in as
  `z` because the compat package replaces symbols). Mac brew
  ships native `libz-ng` (NOT compat). Linux distros and conda
  have `zlib-ng-compat` packages.
* **Path B — Intel ISA-L** (`igzip`): a separate library exposing
  `isal_deflate` / `isal_inflate`. Even faster than zlib-ng on
  Intel hardware; needs its own binding.
* **Sketch (path A)**:
  1. Add probe in `setup.py` for `pkg-config zlib-ng-compat`.
  2. When found, link `-lz-ng-compat` instead of `-lz`.
  3. No code changes needed — symbol-compatible.
  4. h2h bench should show a measurable speedup.
* **Effort**: ~2-3 hours for path A; ~4-6 hours for path B.
