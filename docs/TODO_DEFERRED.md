# Deferred codec / format items

What's actually still open, as of 2026-05-19. Items that shipped are
not relisted — `git log` is the source of truth; this file only tracks
work that didn't fit in the session it was scoped for.

Each entry says **what's done so far**, **what's left**, and roughly
**how much work** the gap is. Effort estimates are calibrated to the
"50–100× faster than the original estimate" cadence we've been hitting;
take them as upper bounds.

---

## Compressed FITS — `PLIO_1`

* **Status**: RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1, NOCOMPRESS, and
  the GZIP_COMPRESSED_DATA / UNCOMPRESSED_DATA fallback columns all
  ship. HCOMPRESS_1 uses vendored cfitsio source
  (``3rdparty/cfitsio/fits_hdecompress.c``) with a minimal
  ``fitsio2.h`` stub.
* **What's left**: ``PLIO_1`` — IRAF mask-coding (run-length over a
  bit-packed mask). Tiny (<300 lines), but only seen on segmentation
  masks in IRAF pipelines. ~1-2 hr. Raises a clear
  ``NotImplementedError`` until then so users get a precise signal.

## EER file-level reader follow-ups

* **Status**: `_eer_reader.py` (the file-level wrapper) and the
  bitstream decoder both ship. Frame iteration + sum across ranges
  work.
* **What's left**:
  1. Dose-corrected temporal-binning helpers (apply a per-frame dose
     curve when summing — currently the API only does flat sums).
  2. Real-acquisition test fixture (a Falcon-4 sample file from an
     actual scope; the synthetic bitstream test catches encoding
     bugs but not file-container quirks the wild EER writers emit).
* **Effort**: 1-2 hr for the dose helper; the fixture is a "find a
  willing collaborator + redistribute the data under an open
  license" task more than a coding one.

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
* **Why deferred**: PNG encode is already ~1.5-3× ahead of
  imagecodecs on natural-image data, so this is diminishing-returns
  territory.

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
