# Deferred codec / format items

What's actually still open, as of 2026-05-19. Items that shipped are
not relisted — `git log` is the source of truth; this file only tracks
work that didn't fit in the session it was scoped for.

Each entry says **what's done so far**, **what's left**, and roughly
**how much work** the gap is. Effort estimates are calibrated to the
"50–100× faster than the original estimate" cadence we've been hitting;
take them as upper bounds.

---

## Compressed FITS — `PLIO_1` — done

* **Status**: every standard FITS tile-compression algorithm now
  ships. RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1, NOCOMPRESS,
  GZIP_COMPRESSED_DATA / UNCOMPRESSED_DATA fallbacks, and PLIO_1.
  PLIO_1 vendors cfitsio's ``3rdparty/cfitsio/pliocomp.c`` (Doug
  Tody, NRAO; public-domain) and exposes a tiny
  ``opencodecs.codecs._plio.decode_raw`` shim. The PLIO column uses
  ``1PI`` (int16 opcodes) rather than ``1PB``, so
  ``_fits_compressed.py`` now tracks the COMPRESSED_DATA column's
  element-byte-width to compute the heap-payload length correctly.
  Round-trip tested against astropy in
  ``tests/test_fits.py::test_compressed_fits_plio_1_roundtrip``.

## EER file-level reader follow-ups

* **Status**: ``_eer_reader.py`` + the bitstream decoder ship.
  ``sum(start, stop, *, weights=None, dtype=...)`` now supports a
  per-frame dose curve via the ``weights`` argument (commit landing
  with this doc edit). Frame iteration, flat-sum, weighted-sum, and
  range-validation are all tested.
* **Done (2026-09)**: the real-acquisition fixture arrived with the
  corpus work and needed no collaborator after all. EMPIAR-10568 is
  CC0, so `corpus.py fetch eer_empiar10568` pulls a genuine 230 MB
  Falcon 4 BigTIFF: 721 IFDs written by Thermo Fisher's software,
  with the private tags and IFD chaining a real acquisition emits.
  `test_eer_reader_on_real_falcon4_acquisition` and
  `test_eer_reader_iteration_matches_indexed_access_on_real_data`
  drive `EerReader` against it. The other reader tests still build a
  minimal TIFF so they run on a fresh clone, but they were only ever
  testing our own idea of the container.

## DICOMweb live integration test — done

* **Status**: shipped in commit ``4d435ae``. The 13 synthetic-response
  unit tests are joined by ``test_dicomweb_live_orthanc_demo_end_to_end``,
  which runs QIDO → WADO frame fetch → decode against
  ``demo.orthanc-server.com`` and skips cleanly when the endpoint is
  unreachable.

## HDF5 cloud — concurrent prefetch + live test — done

* **Status**: both halves shipped.
  * Concurrent multi-chunk prefetch lives in
    ``prefetch_hdf5_chunks`` (``src/opencodecs/_hdf5_http.py``): walks
    ``dataset.iter_chunks(sel)``, resolves each chunk's
    ``(byte_offset, size)`` via ``get_chunk_info_by_coord``, then
    issues a single ``source.read_many`` batch. Subsequent ``dataset[sel]``
    serves from the LRU cache without a fresh round-trip.
    ``test_prefetch_collapses_chunk_fetches`` and
    ``test_prefetch_correct_values`` cover the dispatcher.
  * Live smoke test ``test_remote_hdf5_live_github_endpoint`` runs
    against the HDFGroup-hosted reference file on github and skips
    when offline.

## HTTPDataSource prefetch tuning (cross-cutting) — done

* **Status**: shipped end-to-end.
  * Covering-cache lookup + opt-in fixed-window read-ahead (commit
    ``f7a0889``).
  * Adaptive trigger (commit ``4d435ae``): the streak counter in
    ``_observe_miss`` watches for N adjacent small misses inside
    ``adaptive_locality`` of each other and auto-promotes the next
    miss to an ``adaptive_window`` fetch. On by default with
    ``adaptive_window=65536``, ``adaptive_streak_threshold=3``;
    set ``adaptive_window=0`` to disable.
  * Eight regression tests across ``tests/test_http_prefetch.py``
    verify both the explicit and adaptive paths fire / skip the
    network correctly; ``tests/test_http_byte_savings.py`` keeps the
    sparse-workload bytes-budget assertions green.

## libspng PAETH filter NEON / SSE intrinsics

* **Status**: decode-side PAETH SIMD is already in the vendored
  libspng — `3rdparty/libspng/spng.c` ships both SSE2
  (``defilter_paeth{3,4}`` near line 6688) and NEON
  (``defilter_paeth{3,4}`` near line 7078) defilters. PNG decode is
  not the gap.
* **What's left**: encode-side PAETH filter selection still uses the
  generic scalar path in libspng — no SSE2 / NEON kernel for the
  forward-filter loop. A small ``.c`` shim could shave ~10-20% on
  RGBA8 encode of natural-image data.
* **Measured 2026-09-03, and the estimate above was low.** Encoding
  six Kodak RGB images with the default all-filters choice takes
  155.2 ms; forcing `filter_choice="none"`, which skips
  `get_best_filter` entirely, takes 85.5 ms. So the filter path is up
  to 45% of encode, not 10-20%.
* **Read that as a ceiling, not a target.** Dropping filters also
  makes the output 1.11x bigger, so zlib does less work in the fast
  case, and the gap conflates the five `filter_sum` passes with both
  `filter_scanline` and that reduced deflate work. The true SIMD win
  is some fraction of 45%.
* **Why still deferred**: PNG encode is already ~1.5-3× ahead of
  imagecodecs on natural-image data, and every kernel added to
  `3rdparty/libspng/spng.c` widens a divergence from upstream that
  `ci/check_vendor_drift.py` now has to carry. Worth doing, but as a
  deliberate patch with its own benchmark, not as a drive-by.

## blosc2 perf — Mac + Linux at parity — done

* **Status**: Mac (`a237f29`) and Linux x86_64 both at parity with
  imagecodecs. Both ``_blosc2`` and ``_b2nd`` artifacts are present
  in ``src/opencodecs/codecs/`` for the two arches; the SMB-wedged
  rebuild state cleared after the remount as expected. No code
  change was needed.

## Excluded by user filter (won't ship)

These were considered + rejected because they don't fit the
"streaming + scientific imaging" thesis. Listed for posterity so
nobody re-proposes them:

* N5 (supplanted by zarr v3)
* JPEG-XR (Microsoft-deprecated)
* JPEG-XS (broadcast-only)
* JPEG-SOF3 (legacy medical; revisit only if DICOM compat needs it)
* LZHAM, LZFSE, LZO, LZF (all supplanted by zstd)
* Jetraw (closed SDK; revisit if licensing changes)
* Meshoptimizer (mesh data, not image data)
* PGLZ (PostgreSQL-internal; not a public format)
* Zopfli (much slower than zstd at comparable ratios)
* CCITT Fax3 / Fax4 encode (decoder ships; new-write audience is zero)
* APNG (libspng doesn't support it; libpng+APNG-patch is a fork
  with maintenance overhead, not worth it for the audience size)

---

## Format gaps vs imagecodecs

After the May 2026 work, every coverage gap that fit the thesis is
closed. The remaining items in `ic.{...}_encode/decode` are all in
the user-excluded list above. The Pareto-default sheet (size + speed
vs ic) is fully green; see `bench/bench_codecs.py` for the full
matrix and `git log` for the per-codec commits.
