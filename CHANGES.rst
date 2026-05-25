Changelog
=========

opencodecs is a fork-then-divergence of Christoph Gohlke's
``imagecodecs`` aimed at Pareto-better defaults, native streaming
readers for cloud-backed scientific imaging, and codec coverage
that fits a modern (post-2024) imaging pipeline.

Versions follow the same ``YYYY.M.D`` cadence as upstream when we
publish; the entries below cluster work by date rather than by
release because most of it has shipped continuously to ``main``.

0.1.5 (unreleased)
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


0.1.4 (2026-05-25)
------------------

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
