# HTJ2K (High-Throughput JPEG 2000) — deferred

## Status

**Not implemented.** A native Cython binding is a multi-hour project that
didn't fit in the same overnight run as the rest of the roadmap.

## What it would take

HTJ2K (Part 15 of the JPEG 2000 family, ITU-T T.814 / ISO/IEC 15444-15)
is not implemented in either of the obvious upstream libraries:

* **openjpeg 2.5.4** — header has no `OPJ_PROFILE_HTJ2K` or
  `OPJ_EXTENSION_HTJ2K`. The official OpenJPEG project hasn't merged
  HT support as of 2.5.x.

* **imagecodecs 2026.3.6** — exposes `imagecodecs.htj2k_encode` /
  `htj2k_decode` but the implementation is a `STUB`. Calling either
  raises rather than encoding.

* **Pillow** — no HTJ2K support.

The only viable open-source HT encoder/decoder is **OpenJPH**
(https://github.com/aous72/OpenJPH), MIT-licensed, C++ API. brew has
it as `openjph` (currently 0.27.2).

## Implementation sketch

1. Add OpenJPH to the build dependencies (homebrew on Mac, conda
   recipe on Linux, vendored under `3rdparty/openjph/` for Windows
   wheels).
2. Write a thin C shim around `ojph::codestream` (which is C++ only)
   exposing two C entry points:
   ```c
   int ojph_oc_encode(const uint8_t *pixels, int W, int H, int C,
                      int bit_depth, int reversible,
                      uint8_t **out, size_t *out_size);
   int ojph_oc_decode(const uint8_t *data, size_t size,
                      int *W, int *H, int *C, int *bit_depth,
                      uint8_t **pixels);
   ```
3. Cython binding in `src/opencodecs/codecs/_htj2k.pyx` calling those C
   entry points (same shape as `_jpeg2k.pyx`).
4. Wire `htj2k` into `opencodecs/codecs/_registry.py` next to
   `jpeg2000`.
5. Add to `bench/run_benchmarks.py` head-to-head section. Note that
   imagecodecs comparison won't work until upstream imagecodecs
   actually implements HTJ2K — for now the bench would compare
   against the openjpeg jpeg2000 codec (showing HTJ2K's ~5-10× speedup
   on natural images at the same quality).

## Estimated effort

* C shim: 1-2 hours
* Cython binding + registry wire-up: 2-3 hours
* Tests + h2h bench row + bit-depth coverage: 1-2 hours

Total: ~4-6 hours of focused work.
