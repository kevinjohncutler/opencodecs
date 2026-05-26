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

## DICOMweb live integration test

* **Status**: client ships, 13 unit tests pass against synthetic
  multipart/related responses.
* **What's left**: a smoke test against a real DICOMweb server (the
  IDC public WADO-RS endpoint is the obvious target). The current
  tests prove the parser is right; the live test would catch
  network-stack regressions (HTTPS retries, chunked transfer,
  Content-Type quirks) that synthetic tests miss.
* **Effort**: 2-3 hr to write a careful test that skips cleanly when
  the network is unavailable.

## HDF5 cloud — concurrent prefetch + live test

* **Status**: single-threaded HTTP-range reader ships via
  `open_remote_hdf5(url)`; h5py's normal chunked-dataset path handles
  everything else.
* **What's left**:
  1. **Concurrent multi-chunk prefetch**. The h5py driver issues
     range requests one chunk at a time. A coalescing prefetch
     dispatcher (similar to the one in `TiffPyramidReader`) would
     hide network latency when a slice touches many chunks. ~4-6 hr.
  2. **Live IDC / NASA SnowEx smoke test**, same pattern as the
     DICOMweb item above. ~1-2 hr.

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

* **Status**: switch-in-loop split into per-filter functions is shipped
  — clang/gcc autovectorise everything except PAETH. Real-world PNG
  encode already beats imagecodecs ~1.5-3× on natural-image data.
* **What's left**: hand-written NEON / SSE kernel for PAETH would
  shave another 10-20% on RGBA8 photographic data. Cython doesn't
  emit SIMD intrinsics; would need a small `.c` shim.
* **Why deferred**: we're well ahead of imagecodecs already; this is
  in "diminishing returns" territory.

## blosc2 perf — Mac is at parity; Linux build verified, src/.so SMB-wedged

* **Status**: Mac (`a237f29`) at parity with imagecodecs after
  matching `typesize=8` default + cache-build of c-blosc2 2.23.0.
  Linux build verified at the `build/lib.linux.../` level (NEEDED =
  libblosc2.so.7, pointing at our cache). The `src/.so` copied
  artifact is stuck in an SMB-inconsistent state on the NAS — neither
  `rm`, `mv`, nor in-place overwrite work from either side.
* **What's left**: clean rebuild after SMB remount / reboot — should
  resolve automatically. No code change.

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
