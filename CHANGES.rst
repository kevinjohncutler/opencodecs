Changelog
=========

opencodecs is a fork-then-divergence of Christoph Gohlke's
``imagecodecs`` aimed at Pareto-better defaults, native streaming
readers for cloud-backed scientific imaging, and codec coverage
that fits a modern (post-2024) imaging pipeline.

Versions follow the same ``YYYY.M.D`` cadence as upstream when we
publish; the entries below cluster work by date rather than by
release because most of it has shipped continuously to ``main``.

0.1.6 (2026-06-01)
------------------

**libultrahdr now bundled (fixes missing ``_uhdr`` on PyPI wheels)**

* ``bench/build_codec_libs.sh::build_libultrahdr`` builds libuhdr
  v1.4.0 (Apache-2.0) into ``$OPENCODECS_LIBS_PREFIX/{include,lib}``
  alongside the other source-built codec libs. The Linux + Windows
  CI ``--only=...`` lists pick it up; the macOS ``before-all`` brew
  install now includes ``libultrahdr`` as well.
* setup.py's ``_uhdr`` Extension is now built by a probe
  (``_maybe_build_uhdr_ext``) that finds the cached prefix, sets
  ``library_dirs``, and bakes the rpath into the resulting ``.so``
  so delocate / auditwheel / delvewheel can bundle ``libuhdr``'s
  dylib/.so/.dll into the wheel.
* ``_uhdr`` is now in ``MUST_SHIP_ALL_PLATFORMS`` in
  ``ci/check_wheel_contents.py``; a wheel without the Ultra-HDR
  extension fails the wheel-coverage step.
* The stale ``_ultrahdr.pyx`` source file (orphaned in commit
  ``c9347f5`` when the direct libuhdr binding shipped as ``_uhdr``)
  is removed.

**New API: ``opencodecs.uhdr.decode_native``**

Fused-Cython fast-path Ultra-HDR decoder. Uses libuhdr's parser to
pull out the compressed SDR base + gain-map JPEGs + metadata (no
pixel decode), then decodes both JPEGs in parallel via
``imagecodecs.jpeg_decode`` (libjpeg-turbo SIMD, GIL released) and
applies the gain map in a Cython kernel that uses the same sRGB
EOTF LUT + IEEE-754 polynomial ``exp2`` the encoder uses. ~1.4×
faster than libuhdr's reference decode on a 2k² float HDR (M-series
Mac: ~52 ms vs ~72 ms). Output matches libuhdr's decode to within
JPEG-q95 + 8-bit gain-quantisation noise. ``display_boost`` kwarg
exposes the ISO 21496-1 headroom scaler — default is full HDR
(``hdr_capacity_max``), pass ``1.0`` to match libuhdr's default
SDR-equivalent decode.

Backed by two new Cython helpers exposed for advanced callers:

* ``opencodecs.codecs._uhdr.extract_layers(data)`` — parse the
  container, return ``{base_jpeg, gainmap_jpeg, gainmap_metadata,
  width, height, gainmap_{width,height}}`` without decoding pixels.
* ``opencodecs.codecs._uhdr.apply_gainmap_fp32(sdr_u8, gain_u8,
  metadata, display_boost=...)`` — the per-pixel gain-application
  kernel.

**New API: ``opencodecs.uhdr.encode_to``**

Streaming variant of :func:`encode_native` — writes Ultra-HDR bytes
directly to a file-like (anything with ``write(bytes)``: open file,
``io.BytesIO``, HTTP upload streamer). Returns the byte count.
Forward-compatible alias: the libuhdr api-4 path we currently use
for container assembly doesn't expose a streaming writer, so the
function is ``fp.write(encode_native(...))`` today; the API exists
so callers can adopt it now and pick up any future libuhdr
streaming write-out without changing their code.

**CI / build robustness**

* ``_uhdr`` extension now links ``-lmvec -lm`` on Linux. GCC at
  ``-O3 -ffast-math`` auto-vectorises the sRGB LUT-init ``pow()``
  calls into ``_ZGV*`` symbols from glibc's libmvec. AlmaLinux 8's
  manylinux toolchain auto-links libmvec so the issue was masked
  in CI, but Ubuntu's GCC doesn't — caught when verifying on a
  the linux x86_64 host Ubuntu 25.04 box.
* The libaec source URL moved from ``gitlab.dkrz.de`` (now
  auth-gated; anonymous requests redirect to ``/users/sign_in``)
  to DKRZ's GitHub mirror at
  ``github.com/Deutsches-Klimarechenzentrum/libaec``.
* ``fetch_tar`` now has a 4-attempt shell retry loop with empty-
  extract detection — covers transient mirror outages that
  curl's own ``--retry`` doesn't see (e.g. 200 OK with a
  truncated body).

0.1.5 (2026-05-31)
------------------

**New codecs**

* ``opencodecs.uhdr`` — Ultra-HDR / ISO 21496-1 (gainmap JPEG) via a
  direct Cython binding to Google's libultrahdr. ``encode(hdr, ...)``
  wraps libuhdr's full pipeline; ``encode_native(hdr, sdr=None, ...)``
  is a fused-Cython fast path that computes SDR base + gain map in
  ``nogil`` kernels (IEEE-754 polynomial log2/pow, cross-platform —
  no Apple Accelerate intrinsics), JPEG-encodes both layers in
  parallel via a 3-worker ThreadPoolExecutor, then hands the
  pre-encoded JPEGs to libuhdr's api-4 for container assembly.
  Measured 31 ms median for a 2000² float HDR on M-series Mac vs
  ~173 ms for the libuhdr reference path — 5.5× faster. The
  ``sdr=`` argument lets callers (notably tilescan pipelines) supply
  their own SDR base instead of accepting the peak-normalised default.

* ``PLIO_1`` — IRAF run-length mask coding closes the last FITS
  tile-compression gap. Vendors cfitsio's ``pliocomp.c`` (Doug Tody /
  NRAO, public-domain) and adds the tiny
  ``opencodecs.codecs._plio.decode_raw`` shim. Round-trip tested
  against astropy. The ``COMPRESSED_DATA`` BINTABLE column walker
  now tracks per-column element byte width so the ``1PI`` (int16
  opcodes) layout PLIO uses works alongside the ``1PB`` byte VLA the
  other compressors use.

**New API**

* ``EerReader.sum(start, stop, *, weights=None, dtype=...)`` —
  per-frame dose curve. ``weights[k]`` multiplies the k-th frame in
  the requested range; output promotes to float64 when weights are
  given so fractional contributions don't truncate. Use case:
  beam-induced-motion correction and temporal-binning schemes that
  emphasise different exposures across the acquisition.

**Bug fixes**

* HEIF encode against libheif ≥ 1.18 — newer libheif calls
  ``strlen()`` on the user write-callback's ``heif_error.message``
  even on success and rejects a ``NULL`` pointer with *"heif_writer
  callback returned a null error text"*. The callback now hands back
  a static empty string on success and a descriptive string on OOM.
  Six tests (``test_heif_*``, ``test_phase5_icc.py[heif]``) go back
  to green across every CI matrix entry.

**CI / build**

* ``bench/build_codec_libs.sh`` — the ``is_built`` cache-marker now
  records the install dir alongside the version and re-verifies the
  install dir exists before short-circuiting a recipe. Without this,
  cibuildwheel's manylinux cache (which only covers ``/cibw-jxl-prefix``)
  preserved the marker for recipes that install to
  ``~/.cache/opencodecs/<lib>/`` outside the cached prefix —
  ``lerc``, ``mozjpeg``, ``brotli``, ``zstd``, ``giflib`` — so the
  recipe thought it was done while the actual library files were
  gone. ``_lerc`` + ``_mozjpeg`` had been silently dropped from
  Linux wheels for this reason; the fix restores them.

* ``test_omezarr.py`` skips at module level on zarr-python < 3 —
  fixtures use the v3 API surface, which doesn't exist on the v2
  branch pip resolves on Python 3.10.

* ``test_eer_imagecodecs_cross_validate`` is now symmetric on
  decoder error: if either implementation raises, both must raise
  for the combo to count. Tolerates imagecodecs ≥ 2026.5's tighter
  output-buffer sizing.

0.1.4 (2026-05-25)
------------------

**libvips-inspired streaming improvements**

* ``HTTPDataSource(access='sequential', sequential_chunk_bytes=4*1024*1024)``
  — opt-in libvips-style sequential read mode. Replaces the LRU +
  adaptive read-ahead with a single rolling buffer that slides forward
  as the caller reads. Memory stays bounded to one chunk regardless of
  file size. Target workload: tile-by-tile raster scan over a huge
  COG / OME-TIFF served over HTTP. Backward seeks still work but
  invalidate the buffer; ``stats['sequential_backward_seeks']`` tracks
  the count so users can spot a workload that's actually random and
  would benefit from ``access='random'``.

* ``TiffPyramidReader.read_region`` now parallelises tile decode
  across a thread pool (default ``min(cpu_count(), 8)``). The
  compressed-tile decoders (JPEG, JPEG-2000, deflate, zstd, LZW, WebP,
  LERC) all release the GIL in their C path, so threads scale across
  cores on CPython. Kicks in only when the region covers 4+ tiles —
  fewer than that and thread-pool spin-up cost dominates. Pass
  ``num_decode_workers=1`` to force the serial path for benchmarks /
  regression-diff.

* ``TiffWriter.write_pyramid_auto`` gained ``stream_levels=True``
  (default for COG layout). Computes and writes one level at a time
  via a new ``iter_pyramid_levels`` generator, dropping each finished
  level before computing the next. Peak memory drops from
  ``~1.33 × level0`` (entire geometric series materialised) to
  ``~2 × current_level``. Useful when level 0 already dominates RAM
  (whole-slide pathology, cryo-EM tomograms). Output file is
  byte-identical to the materialize-all path. SubIFD layout
  (``subifds=True``) still uses the materialize path because that
  layout writes sub-resolution IFD offsets into the main IFD's tag
  330 — sub-level offsets have to be known up front.

  Validated via tests:
  ``test_http_sequential_serves_from_rolling_buffer``,
  ``test_http_sequential_backward_seek_counted``,
  ``test_pyramid_reader_parallel_decode_matches_serial``,
  ``test_pyramid_auto_stream_matches_materialize``,
  ``test_iter_pyramid_levels_matches_make_pyramid_levels``.

**MozJPEG ships on every wheel**

* Added MozJPEG (Mozilla's libjpeg-turbo fork) to the cibuildwheel
  codec-lib build set on Linux, macOS, and Windows. MozJPEG produces
  JPEG files ~10-15% smaller than baseline libjpeg-turbo at the same
  quality setting (progressive encoding + trellis quantization +
  better quantization tables). The ``_mozjpeg`` extension has shipped
  on macOS wheels for a while via brew; v0.1.4 brings it to
  Linux + Windows so the codec is available everywhere.
* New ``build_mozjpeg`` recipe in ``bench/build_codec_libs.sh``
  installs into a keg-style ``mozjpeg/`` subdir under the prefix
  (rather than the shared ``$PREFIX/{include,lib}``) to avoid the
  libturbojpeg / libjpeg name collision with the libjpeg-turbo 3.x
  install. setup.py's mozjpeg probe was extended to recognise the
  new candidate paths on Linux (``~/.cache/opencodecs/mozjpeg``),
  Windows (``$CONDA_PREFIX/Library/mozjpeg``), and the Windows
  fallback (no ``nm``: trust the keg-style directory name).
* Locally validated on the Windows VM (MSVC 14.44 — same as CI) and
  the Linux x86_64 host (Linux x86_64): ``bash bench/build_codec_libs.sh
  --only=mozjpeg`` followed by ``setup.py build_ext --inplace``
  produces ``_mozjpeg.{pyd,so}`` cleanly. Per the new lesson in
  CLAUDE.md, this iteration was validated end-to-end through the
  same script CI runs, with the same env vars, before any push.

**Post-publish wheel coverage check (CI hardening)**

* New ``ci/check_wheel_contents.py`` runs in the wheel-build job
  immediately after ``cibuildwheel`` produces each wheel. It unzips
  the wheel, walks ``opencodecs/codecs/``, and asserts every codec
  in the ``MUST_SHIP_ALL_PLATFORMS`` set has a ``.pyd``/``.so``
  present. Missing codec → matrix job fails → publish step gated.
* Catches the v0.1.2-style "changelog overclaim" silent-drop class
  of bugs at build time instead of after publish. v0.1.2's Windows
  wheels shipped without ``_sperr``/``_brunsli`` despite the
  changelog claim; this check would have failed that build.


0.1.3 (2026-05-22)
------------------

**SPERR + brunsli land on every wheel**

* Added ``SPERR`` and ``brunsli`` to the cibuildwheel codec-lib
  build recipe for Linux, macOS, and Windows. Previously the
  ``build_codec_libs.sh --only=...`` selection for cibuildwheel
  ran only SZ3 + pcodec, so the SPERR / brunsli ``cmake_build``
  paths (already in the script) never fired on CI — setup.py's
  header probe dropped ``_sperr`` and ``_brunsli`` from every
  wheel released through v0.1.2.
* brunsli vendors its own brotli submodule via CMake, so no
  additional system-brotli dependency at link time. Both libs
  install into the per-user opencodecs cache and get bundled
  into the wheel via auditwheel / delocate / delvewheel.
* v0.1.3 ships ``_sperr`` (NCAR wavelet error-bounded compressor)
  and ``_brunsli`` (Google lossless JPEG transcoder, ~22% smaller
  storage) on all four wheel cells (Linux x86_64 + aarch64,
  macOS arm64, Windows AMD64) × Python 3.10-3.13.

**Errata for v0.1.2**

* The v0.1.2 CHANGES claimed "Restored ``_sz3``, ``_pcodec``,
  ``_sperr``, ``_brunsli`` on Windows." In reality only ``_sz3``
  and ``_pcodec`` were restored — the SPERR / brunsli libs never
  got into the ``--only=`` build selection, so their extensions
  silently dropped via the missing-header probe. v0.1.3 closes
  the gap.


0.1.2 (2026-05-22)
------------------

**Windows wheels get _sz3 and _pcodec back**

* Restored ``_sz3`` and ``_pcodec`` on Windows. Root cause was
  conda's bash putting ``gcc.exe`` ahead of ``cl.exe`` on PATH;
  CMake then produced gnu-format ``libSZ3c.dll.a`` import
  libraries that cibuildwheel's MSVC link.exe couldn't consume.
* Workflow now uses ``ilammy/msvc-dev-cmd`` to source vcvars64.bat
  before the SZ3+pcodec source-build step; CMake's auto-detect picks
  cl.exe and produces MSVC-format ``SZ3c.lib`` / ``cpcodec.lib``.
* Cargo's MSVC linker pinned via
  ``CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER`` so rustc doesn't
  PATH-resolve to GNU coreutils' ``link.exe`` at
  ``C:\Program Files\Git\usr\bin\link.exe``.
* Validated end-to-end on a Windows 11 VM (clean SZ3 install with
  Ninja + cl.exe + vcvars-sourced env).

**README rewrite for the released project**

* ``pip install opencodecs`` + PyPI badge at the top.
* New "Why opencodecs" table mapping common scientific-imaging needs
  to concrete shipping capabilities.
* New "Streaming-reader examples" section with 3 copy-paste recipes:
  HTTP region-fetch from a remote Aperio TIFF, TIFF → OME-Zarr v3
  sharded conversion, and the native progressive JXL thumbnail path.
* Status / Install sections updated for the 0.1.x cadence.


0.1.1 (2026-05-21)
------------------

Supersedes 0.1.0 (yanked). Same codec coverage as 0.1.0 plus the
work since, with source-comment metadata scrubbed.

**CMS — sRGB ↔ Display-P3 fast converter**

* Add ``opencodecs._cms_codec.srgb_to_display_p3_uint8(arr)``
  convenience for the gallery / Jupyter / web-display case: convert
  sRGB-encoded uint8 RGB(A) → Display-P3-encoded uint8 in ~28 ms
  for a 2Kx2K image on macOS arm64 (vs ~110 ms for an equivalent
  numpy LUT + matmul pipeline).
* Add ``_builtin_profile_icc(name)`` returning ICC bytes for the
  built-in profiles ``"srgb"`` and ``"display-p3"``. Display-P3 is
  synthesized via ``cmsBuildParametricToneCurve`` +
  ``cmsCreateRGBProfile`` rather than depending on lcms2 ≥2.16's
  ``cmsCreate_DisplayP3`` — works on older liblcms2 too.
* Lcms2 ``COPY_ALPHA`` flag doesn't combine with manually-built
  RGB-only profiles, so the RGBA path transforms RGB in a
  contiguous temporary and stitches the alpha channel back.
* 9 new tests under tests/test_phase7_cms.py cover canonical
  primary-color transforms, gray invariance, alpha preservation,
  and input validation.

**OME-Zarr v3 sharded write**

* Add ``shards=`` kwarg to ``opencodecs.write_zarr_array``,
  ``write_omezarr_pyramid``, and ``write_omezarr_pyramid_auto``.
  Enables the Zarr v3 ``sharding_indexed`` codec: each shard file
  on disk packs ``prod(shards / chunks)`` inner sub-chunks plus
  a trailing ``uint64`` ``(offset, nbytes)`` index. The reader
  (which already supported sharded *reads* with HTTP-range
  fetches) now has a writer counterpart, closing the OME-Zarr
  v3 write story.
* Validation: ``shards=`` requires ``zarr_format=3`` and each
  shard axis must be a multiple of the corresponding chunk axis
  (so each shard holds a whole number of inner chunks).
* Pyramid writers auto-adapt ``shards`` per level — when a
  downsampled level is smaller than the requested shard shape,
  the shard clamps to the largest multiple of ``chunks`` that
  fits, and falls back to per-chunk layout when a level is too
  small to hold even one chunk.
* Verified pixel-equal round-trip via the reference zarr-python
  reader (including the edge-of-array case where the array
  dimensions aren't a multiple of the shard shape — those slots
  use the standard Zarr empty-chunk sentinel of ``2**64 - 1``).
* Index uses bytes-only encoding (no CRC32C yet — would add a
  ``crc32c`` runtime dep and the reader already handles its
  absence gracefully).

**JXL ``subsample`` kwarg for downsample positioning**

* Add ``subsample={'top-left', 'center'}`` kwarg to
  ``opencodecs.jxl.read`` / ``open`` / ``iter_frames`` and the
  underlying ``JxlReader`` / ``decode``.
* ``'top-left'`` (default) keeps the historical
  ``arr[::N, ::N]`` semantic — back-compat with imagecodecs and
  every existing caller.
* ``'center'`` takes ``arr[N//2::N, N//2::N]`` so each output
  pixel represents the geometric centroid of its source NxN
  block. The right choice for visual thumbnails: when the
  downsampled raster gets drawn in an NxN region of an output
  canvas (SVG ``<image>``, GL texture upload, …), centroid
  semantics keep the thumb positionally self-consistent with
  the full-res source. Top-left semantics shift features by
  ~(N-1)/2 source-pixels.
* Output shape is ``ceil(src/N)`` per axis in both modes.
  When source dimensions aren't divisible by N, the centered
  slice would otherwise come up one row/col short; we replicate
  the bottom/right edge to preserve the shape contract.
* No perf change — both modes are pure index/copy on the
  already-decoded buffer. The decode path is unchanged.

**Tier 3 streaming-reader plumbing**

* Add ``HTTPDataSource`` covering-cache lookup: when ``read_at(off, n)``
  misses the exact ``(off, n)`` LRU key, scan for any cached blob that
  fully covers the requested range and slice it out. Compounds for
  free — any time a reader fetches a big region, later small reads
  inside it don't round-trip.
* Add adaptive read-ahead trigger. When 3 consecutive small cache
  misses fall within ``adaptive_locality`` (64 KB) of one another —
  the signature of a FITS HDU walk, h5py B-tree traversal, or TIFF
  IFD chain — the next miss gets bumped to ``adaptive_window``
  (64 KB) so subsequent adjacent reads serve from cache. Scattered
  reads keep the streak at 1 and never trigger.
* ``HTTPDataSource`` now returns ``b""`` (not a raised
  ``HTTPException``) for ``Range`` requests past end-of-file. Matches
  file-like ``read()`` behavior and lets the FITS HDU walker probe
  the next-HDU offset without a try/except dance.
* Add ``read_hdf5_slice(dataset, sel)`` convenience wrapper that
  bundles ``prefetch_hdf5_chunks`` + the actual read into one call.
  Falls through cleanly when the dataset isn't backed by
  ``open_remote_hdf5`` (local file path).

**FITS reader**

* Native FITS reader (``opencodecs._fits.FitsStream``). Multi-HDU
  parsing, BITPIX 8/16/32/64/-32/-64, BSCALE/BZERO with the FITS
  unsigned-int convention (BZERO=2**(N-1) → uintN). HTTP-range
  friendly: one Range request per HDU header at open time, image
  data fetched lazily on ``asarray()``.
* Compressed-image decode for BINTABLE+ZIMAGE HDUs. Supported
  ZCMPTYPE: RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1, NOCOMPRESS.
  Per-tile ZSCALE / ZZERO quantization for floats. Fall-back path:
  when the primary COMPRESSED_DATA descriptor is empty, decode from
  GZIP_COMPRESSED_DATA (lossless gzipped original bytes) instead.
  HCOMPRESS_1 uses cfitsio's ``fits_hdecompress`` vendored under
  ``3rdparty/cfitsio/`` (BSD-style license).
* astropy.io.fits is the cross-validation oracle: every supported
  ZCMPTYPE × dtype combination decodes pixel-equal.

**Ultra HDR / Radiance HDR**

* Add ``rgbe`` codec (Radiance ``.hdr``) — Cython binding to the
  vendored Bruce Walter / Greg Ward C library. RLE-compressed by
  default; cross-validated against ``imagecodecs.rgbe_encode/decode``.
* Add ``ultrahdr`` codec (ISO 21496 gainmap JPEG) — Cython binding
  to Google's libultrahdr 1.4.0. Default encode: ``(H, W, 4) float16``
  linear BT.2100 RGBA at quality 95; decode returns the same. Tested
  against imagecodecs's ultrahdr binding for pixel-equal interop.

**Intel ISA-L deflate (opt-in)**

* Add ``backend="isal"`` option to ``DeflateCodec.encode/decode``.
  ISA-L's igzip is ~4× faster than libdeflate at encode but
  produces ~19% bigger output and is slightly slower at decode —
  not strictly Pareto-better, so opt-in via the backend kwarg rather
  than the new default.

**Compressor coverage**

* Add ``gzip`` and ``none`` codecs (stdlib-based wrappers); add
  ``zlibng`` as an alias of ``deflate``. Lets ic-compatible callers
  (tifffile, zarr filter chains) name backends explicitly.

**Pareto-default closures vs imagecodecs**

* ``aec`` encode: 3.51× → 1.74× on the 200 KB uint16 bench workload.
  Streaming ``aec_encode_init`` / ``aec_encode(AEC_FLUSH)`` /
  ``aec_encode_end`` API plus correct worst-case output cap
  (``srcsize * 67/64 + 257``).
* ``zfp`` encode: 1.19× → 0.98× on Mac (parity). Switched to
  absolute-dylib link to bypass macOS sysconfig's prepended
  ``/opt/homebrew/lib`` ordering; cache builds via
  ``bench/build_codec_libs.sh --only=zfp`` now win.
* ``blosc2``: 2.12× → 1.06× by matching imagecodecs's
  ``typesize=8`` default for bytes input (rather than 1). Output is
  bit-identical to ic on default settings. Also added the cached
  c-blosc2 2.23 build to the build recipe.

**Build infrastructure**

* ``bench/build_codec_libs.sh``: add ``-DCMAKE_POLICY_VERSION_MINIMUM=3.5``
  to CMAKE_COMMON so cmake 4.x can configure projects with pre-3.5
  ``cmake_minimum_required`` (x265, libheif).
* Add ``setup.py`` ``_user_cache_rpath_args()`` helper that auto-bakes
  the per-user cache lib dirs into every extension's DT_RUNPATH /
  LC_RPATH. Closes a class of "import works in tests but breaks at
  runtime" bugs.

**Live archive smoke tests**

* Add ``tests/test_live_archives.py`` (marked ``slow``) — open and
  read a 700 KB NASA GSFC FITS file and an EMBL-EBI IDR OME-Zarr
  dataset over HTTPS. Skips cleanly when the network is unreachable.
  Catches the class of bugs synthetic tests can't (Content-Type
  quirks, server recompression, real CDN HTTPS retry behavior).
