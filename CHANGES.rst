Changelog
=========

opencodecs is a fork-then-divergence of Christoph Gohlke's
``imagecodecs`` aimed at Pareto-better defaults, native streaming
readers for cloud-backed scientific imaging, and codec coverage
that fits a modern (post-2024) imaging pipeline.

Versions follow the same ``YYYY.M.D`` cadence as upstream when we
publish; the entries below cluster work by date rather than by
release because most of it has shipped continuously to ``main``.

Unreleased
----------

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
