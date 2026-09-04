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

## Writer ABC — done

* **Status**: `core/codec.py` declared three ABCs and exported all
  three, but nothing implemented `Writer`. Six streaming writers, none
  of them a `Writer`, and only three with a `write_frame` at all: TIFF
  called it `write_page`, CZI `write`, the CZI pyramid writer
  `write_level`. NDTiff had the name with the axes dict in the array's
  position, so it satisfied the name and not the contract.
* All six now implement it, the format-specific names stay, and
  `tests/test_writer_contract.py` drives each of them from a helper
  that knows nothing about the format.
* **Also done**: `Codec.writer()` and `oc.writer()` now mirror
  `Codec.open()` / `oc.open()`, including the default -- where `open()`
  decodes eagerly and wraps the result in a one-frame Reader,
  `writer()` buffers frames and encodes on close. So a caller drives
  any encoder through one interface and only pays the buffering for
  formats that cannot stream. TIFF, CZI, GIF and JXL return real
  streaming writers.
* Two adapters were needed to make the contract call *correct* rather
  than merely accepted. GIF needs the canvas size before the first
  frame, which a generic caller has not got, so construction defers to
  the first `write_frame` and takes it from that frame's shape. JXL
  needs the final frame of an animation flagged when it is submitted --
  `close()` is not a substitute -- so its writer holds one frame back.
  That one is worth remembering: without it the stream is *exactly the
  same length* and simply fails to decode, so no size or smoke check
  catches it.

## libspng encode filters — measured, and the win was elsewhere — done

* **The deferred item was encode-side PAETH SIMD**, on the strength of a
  measurement that encoding six Kodak images with all filters took
  155 ms against 85 ms with filtering off, so "the filter path is up to
  45% of encode".
* **That attribution was wrong.** Comparing against no-filtering also
  changes what deflate sees, so it conflates three costs. Varying only
  the number of `filter_sum` passes separates them, using a single
  filter choice as the control (it returns from `get_best_filter`
  immediately but still filters every row):

  | content | sums @ default | sums @ "all" |
  |---|---|---|
  | kodak RGB8 | ~0% | 5.2% |
  | gradient RGB8 | 9.1% | 20.8% |
  | noise RGB8 | ~0% | 2.0% |
  | gradient u16 | 1.9% | 5.6% |

  So a perfect SIMD kernel in `filter_sum` would have bought roughly
  nothing at the default and ~5% on real content at `"all"`. The
  `filter_sum_*` specialization already there had taken the win.
* **`filter_scanline` was the real target**, and nobody had looked at
  it: it kept both a `switch(filter)` and an `if(i >= bytes_per_pixel)`
  inside the per-byte loop, so it could not vectorize. It also runs on
  every row of every encode, on the filter that won — where the sums are
  computed for candidates and discarded.
* Specializing it the same way (one function per filter, boundary split
  out as a prelude) gives **13.4% faster encode on arm64 and 5.7% on
  x86_64** at the default filter choice, with **byte-identical output**
  across 104 (case, filter) combinations. `filter_choice="off"` is
  unchanged, which is the control: it skips filtering entirely.
* **What is left**: genuine SIMD intrinsics for the paeth kernels, worth
  perhaps a few more percent, at the cost of platform-specific code in a
  vendored file. Not obviously worth it now that the branchy loop is
  gone.

## blosc2 perf — Mac + Linux at parity — done

* **Status**: Mac (`a237f29`) and Linux x86_64 both at parity with
  imagecodecs. Both ``_blosc2`` and ``_b2nd`` artifacts are present
  in ``src/opencodecs/codecs/`` for the two arches; the SMB-wedged
  rebuild state cleared after the remount as expected. No code
  change was needed.

## Shipped since this list was written

* **N5** was on the "won't ship" list below as "supplanted by zarr v3".
  It shipped anyway in the scientific-reader work: `_n5.py`, a corpus
  entry (Janelia's jrc_hela-2), and parity tests against tensorstore.
  The reasoning that excluded it was about what people should write,
  not about what is already on disk, and the FIB-SEM datasets that
  matter are N5.

## Excluded by user filter (won't ship)

These were considered + rejected because they don't fit the
"streaming + scientific imaging" thesis. Listed for posterity so
nobody re-proposes them:

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
