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

* **Status**: bitstream decoder shipped — see
  `src/opencodecs/codecs/_eer.pyx` and `3rdparty/imcd_eer/eer.{c,h}`.
* **What's done**: native event-list decoder, vendored from imagecodecs'
  BSD-3 imcd.c (no runtime dep). Verified against the EER spec test
  vector and cross-validated against `imagecodecs.eer_decode` over a
  parameter sweep on random bitstreams. Supports binary (uint8) and
  uint16 accumulator output, and the super-resolution sub-pixel mode.
* **What's left**:
  1. File-level wrapper: a `tifffile`-style EER reader that opens the
     TIFF container, extracts the per-strip skipbits/horzbits/vertbits
     tags (65007/65008/65009), and feeds each strip through
     `_eer.decode`. Could live in `src/opencodecs/io/eer.py` and
     reuse `_tiff_codec.py` for the TIFF walk.
  2. Dose correction / temporal binning across frames.
  3. Real-acquisition test fixture (would need a Falcon 4 sample;
     synthetic ones suffice for the bitstream decoder itself).

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

* **Status**: WADO-RS frame retrieval shipped — see
  `src/opencodecs/_dicomweb.py` and `tests/test_dicomweb.py`.
* **What's done**:
  * `DicomwebClient.get_frame(study, series, instance, frame)` issues
    the WADO-RS request and decodes the returned multipart/related
    body into an ndarray.
  * `DicomwebClient.list_instances()` for QIDO-RS series enumeration.
  * Self-contained multipart/related parser (RFC 2046).
  * Transfer-syntax dispatch covering JPEG baseline/extended,
    JPEG-LS (CharLS), JPEG-2000 (OpenJPEG), HTJ2K (OpenJPH),
    raw Explicit/Implicit VR LE, and DICOM Annex G RLE Lossless.
  * 13 unit tests pass — multipart parsing, RLE round-trip, codec
    dispatch through CharLS/OpenJPH for the JPEG-LS / HTJ2K syntaxes.
* **What's left**:
  * Live integration test against an `orthanc` Docker container — the
    current tests synthesize responses; a real CI smoke test against
    a public DICOMweb server (e.g. IDC's WADO-RS endpoint) would
    increase confidence.
  * STOW-RS (upload) — out of scope for the read-side codec layer.
  * OAuth dance for Google Healthcare / AWS HealthImaging — caller
    supplies bearer tokens via `headers=...` for now.

## HDF5 cloud (h5coro)

* **Status**: shipped — see `src/opencodecs/_hdf5_http.py`.
* **What's done**: ``open_remote_hdf5(url)`` returns an ``h5py.File``
  whose backing storage is our ``HTTPDataSource`` — a Python file-
  like that issues HTTP Range requests with HTTP/1.1 keep-alive and
  an LRU cache. h5py's existing chunked-dataset machinery pulls only
  the chunks covered by the slice the user reads, so a 100GB HDF5
  archive in S3 can be sliced into a tiny ndarray with kilobytes of
  transfer. Same pattern as kerchunk + xarray but without those
  runtime deps — stdlib only.
* **What's left**:
  * Concurrent multi-chunk prefetch (the h5py driver is single-
    threaded; a smarter dispatcher could parallelize chunk fetches
    for large slices).
  * Live IDC / NASA SnowEx smoke test.
* **Why this beats binding h5coro**: h5py 3.x already accepts file-
  like objects, so the wrapper is < 100 lines, no native code, and
  uses the entire upstream h5py decode path (filters, fill values,
  reference resolution) instead of reimplementing it.

## libspng filter_sum SIMD vectorization

* **Status**: deferred. Real-world PNG-encode workloads now beat
  imagecodecs (see numbers below) thanks to the libdeflate IDAT
  fast path. The filter-bound gap shows up only on synthetic
  filter-dominated workloads.
* **What's measured** (2026-05-16 with libdeflate path active):

  | benchmark                | encode oc/ic | decode oc/ic |
  |--------------------------|-------------:|-------------:|
  | 4MP RGB u8 random        |    0.893×    |    0.593×    |
  | 4MP RGB u16 random       |    0.903×    |    0.882×    |
  | 1080p RGB u8 random      |    0.874×    |    0.603×    |
  | Kodak01 RGB u8 (photo)   |    0.496×    |    0.868×    |
  | 512×512 u16 gradient     |    1.732× ⬇  |    0.421×    |

  The 512×512 u16 gradient case is the only encode regression —
  it's filter-bound (input compresses 99.95%, so all the
  wall-clock is in filter scoring, not deflate).

* **Root cause**: libspng's `filter_sum` is a per-byte switch
  statement on each candidate filter. imagecodecs uses libpng,
  which ships SIMD-vectorized filter scoring (NEON + SSE).
* **Sketch**:
  1. Add `filter_sum_neon` (arm64) and `filter_sum_sse` (x86) in
     a sibling file `3rdparty/libspng/filter_sum_simd.c` that
     ships alongside the vendored libspng.
  2. `#ifdef SPNG_USE_SIMD_FILTER_SUM` block in
     `get_best_filter` (spng.c line ~1711) dispatches to the
     SIMD variants per architecture.
  3. Bench until the gradient case hits parity.
* **Effort**: ~1-2 hr focused SIMD work. Off the bench-tracked
  workload, so prioritise only if a user surfaces filter-bound
  PNG-encode wall-clock.

## libdeflate-in-libspng (PNG encode)

* **Status**: SHIPPED. The vendored `3rdparty/libspng/spng.c` is
  patched with an `SPNG_USE_LIBDEFLATE` accumulator path that
  collects the filtered scanline stream and calls
  `libdeflate_zlib_compress` once at IDAT-finalize time, replacing
  the per-scanline `deflate()` calls. setup.py probes for
  libdeflate at build time and defines `SPNG_USE_LIBDEFLATE=1`
  while linking `-ldeflate` when found.
* **Result**: PNG encode is faster than imagecodecs on every
  real-world workload (see filter_sum SIMD table above). decode
  stays on zlib-ng-compat for tEXt/zTXt/iTXt + IDAT.

## libdeflate raw-deflate backend (general compressor)

* **Status**: SHIPPED. `_deflate.pyx` has a compile-time backend
  selector (`-DOPENCODECS_HAVE_LIBDEFLATE=1`) and ships in builds
  where setup.py finds the library. `_deflate.backend()` reports
  `"libdeflate"` or `"zlib"` at runtime.
* **Result**: raw deflate encode 1.92–2.20× faster than zlib/
  imagecodecs; decode 7.11× faster.

## zlib-ng / ISA-L deflate swap

* **Status**: zlib-ng-compat path A SHIPPED (commits c15d3b6 + 027e267).
  ISA-L path B still deferred.
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

## CCITT Fax3 / Fax4 encode

* **Status**: deferred. CCITT decoder is shipped (vendored in
  `3rdparty/ccitt/`); encoder is not.
* **Use case**: 1-bit fax / scanned-document images. Effectively
  legacy — modern scientific TIFF doesn't use CCITT.
* **Sketch**: implement Group 3 1D, Group 3 2D, and Group 4 encoders
  per ITU-T T.4/T.6. ~1500 lines of bit-stream encoding logic, plus
  per-mode tables. tifffile delegates CCITT encode to imagecodecs;
  we could do the same and only ship decode natively.
* **Why deferred**: the audience for *writing* new CCITT TIFFs from
  scientific imaging code is essentially zero; the audience for
  *reading* legacy CCITT TIFFs is small but real and is already
  served. Estimated 8-12 hours work for a feature that benefits
  nobody on our roadmap.

## JPEG-XR encode/decode

* **Status**: deferred (no native codec; jxrlib is available via
  Homebrew but not currently bound).
* **Use case**: niche — Windows Imaging Component / DICOM JPEG-XR
  transfer syntax (1.2.840.10008.1.2.4.105/106). Most modern
  imaging stacks have abandoned it in favor of JPEG XL.
* **Sketch**: Cython binding around libjxrencode/libjxrdecode (BSD-2,
  Homebrew `jxrlib`). Estimated 4-6 hours.
* **Why deferred**: virtually no scientific-imaging workflow we know
  about emits JPEG-XR, and the DICOMweb client we ship would only
  hit it on very old radiology archives. If a user actually needs it
  we'll prioritize it; until then it sits below CCITT in the queue.
