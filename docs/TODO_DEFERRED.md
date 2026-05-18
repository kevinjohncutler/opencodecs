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

## Codecs with imagecodecs-parity perf gaps

Pareto-default audit (see ``docs/codec_api_conventions.md`` and
``bench/bench_codecs.py``) covered all 38 codecs where ``imagecodecs``
ships an equivalent encoder. Two of the four originally-flagged
real perf gaps were fixed; two remain documented:

* **rcomp**: SHIPPED Cython binding to cfitsio's ``ricecomp.c``
  (vendored into ``3rdparty/cfitsio/``). Was a pure-Python
  bit-stream encoder running ~1000× slower than imagecodecs; now
  at parity (~0.018 ms each on a 4 K-element int16 array). Output
  is byte-stream-compatible with the cfitsio implementation —
  ``opencodecs.rcomp`` blobs decode through any other cfitsio
  consumer. Caveat: old blobs from the pure-Python era used a
  different payload format and will NOT decode anymore. rcomp is
  an in-process compressor, not a long-term storage format, so
  the break is acceptable.

* **bmp**: SHIPPED Cython encode + decode fast paths
  (``codecs/_bmp.pyx``). Encode beats imagecodecs ~8.7× on a Kodak
  photo (0.03 ms vs 0.26 ms) by writing directly into a
  ``PyBytes_FromStringAndSize`` buffer with a tight RGB→BGR loop;
  decode is at parity (~0.031 ms each) after switching the inner
  loop to linear pointer increments (clang autovectorises to NEON
  ``vld3.u8``/``vst3.u8``). Pure-Python fallback retained for
  builds that don't compile the Cython extension.

* **aec**: still ~3.5× slower at MATCHED settings. The gap is
  inside ``_aec.pyx`` (libaec's ``aec_buffer_encode`` does init +
  encode + end in one call; imagecodecs uses a streaming setup
  that's measurably faster). Fix would be re-plumbing the pyx to
  call ``aec_encode_init`` / ``aec_encode`` / ``aec_encode_end``
  directly. ~1-2 hr. Deferred — sub-millisecond absolute times
  and our default settings produce ~3% smaller output than ic's,
  so the size win partially offsets.

* **blosc2**: 2.1× slower than imagecodecs at zstd-level-1 (the
  matched setting) but produces ~9% smaller output. Root cause:
  opencodecs links Homebrew's c-blosc2 3.x; imagecodecs bundles
  c-blosc2 2.23.0. c-blosc2 3.x applies different default filter
  chains that produce smaller output but at higher CPU cost. No
  code change can fix this without changing the linked
  c-blosc2 — out of scope for one codec since the brew package
  is shared.

* **zfp**: CLOSED. Was 17% slower with brew's stock libzfp bottle.
  After building libzfp from source with ``-O3 -march=native +LTO``
  via the existing ``bench/build_codec_libs.sh --only=zfp`` recipe,
  oc beats imagecodecs by ~10% on x86_64 threadripper
  (encode 0.65 ms vs 0.72 ms, ratio 0.909).

  Shipped in commits ``517d346``, ``95c238a``, ``18294ea``:

  * memoryview-cast write path in ``_zfp.pyx`` (matches
    ``_zstd.pyx``) — saved ~5–10 µs / encode.
  * ``setup.py`` probes ``~/.cache/opencodecs/{zfp,libs}`` ahead
    of system / brew, so a per-user tuned build is preferred
    automatically.
  * ``-Wl,-rpath`` for every cache lib dir baked into the
    extension's DT_RUNPATH via ``extra_link_args`` (the distutils
    ``runtime_library_dirs`` keyword was silently dropping
    duplicates). Verified with ``readelf -d`` — runpath now lists
    the cache prefix; import works without ``LD_LIBRARY_PATH``.

  **Recovery recipe** (for any new user who installs from
  Homebrew's libzfp bottle and notices the gap)::

      MARCH=native USE_LTO=1 bash bench/build_codec_libs.sh --only=zfp
      <python> setup.py build_ext --inplace

  Needs bash ≥ 4.x for the build script's associative arrays;
  macOS ships 3.2, so on Mac run via ``/opt/homebrew/bin/bash``
  after ``brew install bash``.

  **Open follow-up**: the rpath-via-extra_link_args fix applies
  to ``_zfp`` only. Other codecs that link to ``~/.cache/opencodecs``
  cached builds (lerc / sperr / sz3 / pcodec / brunsli / brotli /
  zstd / giflib) should get the same treatment — currently they
  link OK at build time but rely on the per-codec-cache symlinks
  the build script creates rather than DT_RUNPATH. Low priority
  since those builds usually live under prefixes the dynamic
  loader already searches.

Marginal cases (within +10-15% of ic on small absolute times,
all pass on size/quality) — adapter overhead dominates and is not
worth a Cython rewrite:
delta, xor, byteshuffle (already ★), numpy, snappy, qoi, jpegls,
webp-lossless, zfp, sperr.

## libspng filter_sum SIMD vectorization

* **Status**: SHIPPED via filter-switch split (commit landing
  in 2026-05-17 with this doc edit). The vendored
  `3rdparty/libspng/spng.c` now has five specialised
  `filter_sum_<filter>` functions instead of a single
  switch-inside-loop. Modern compilers (clang on arm64, gcc/clang
  on x86_64) autovectorise each branch — no hand-written SIMD
  intrinsics needed.

* **Measured before-and-after** (M1 Ultra, 2026-05-17):

  | benchmark              | before    | after     | speedup |
  |------------------------|----------:|----------:|--------:|
  | 4MP RGB u8 random      |  0.893×   |  0.506×   |  1.76×  |
  | 4MP RGB u16 random     |  0.903×   |  0.517×   |  1.75×  |
  | 1080p RGB u8 random    |  0.874×   |  0.498×   |  1.76×  |
  | Kodak01 RGB u8         |  0.496×   |  0.318×   |  1.56×  |
  | 512×512 u16 gradient   |  1.732× ⬇ |  0.549×   |  3.15×  |
  | 2048×2048 u16 gradient |    n/a    |  0.588×   |    —    |

  All ratios are oc/ic (encode time); smaller = faster.
  The previously-regressing filter-bound gradient case now beats
  imagecodecs by ~2×, and every real-world workload picked up an
  extra 1.5–1.8× because the inner-loop switch was hurting
  autovectorisation everywhere, not just on filter-bound input.

* **What this leaves open**: hand-written NEON/SSE kernels for the
  PAETH filter could shave another 10–20% on RGBA8 photographic
  data (paeth is the hardest case to autovectorise). Not
  prioritised — we're already well ahead of imagecodecs.

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
