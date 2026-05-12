# HTJ2K (High-Throughput JPEG 2000) — shipped

## Status

**Implemented** via OpenJPH (libopenjph, MIT). Lossless + irreversible
modes; uint8 / int8 / uint16 / int16 input across 1 / 3 / 4 components.

- Cython binding: `src/opencodecs/codecs/_openjph.pyx`
- C++ shim around `ojph::codestream` + `mem_in/outfile`:
  `src/opencodecs/codecs/openjph_shim.{h,cpp}`
- Tests: `tests/test_openjph.py` (13 pass + 1 cross-decode skip when
  imagecodecs has no HTJ2K backend built locally)
- Build: `_maybe_build_openjph_ext()` in setup.py — built only when
  libopenjph is on the system (homebrew at `/opt/homebrew/opt/openjph`).

## Why a C++ shim, not a direct Cython binding

OpenJPH's encoder/decoder is a stateful C++ class with value-returning
accessors (`param_siz access_siz()`) and a union-typed `line_buf::i32`
field. Wrapping the full class from Cython is doable but messy; the
shim is far smaller and keeps the .pyx focused on numpy<->codec
plumbing.

## Cross-decode

When the local imagecodecs install includes the HTJ2K backend
(builds where `imagecodecs.htj2k_decode` actually loads), the
`test_openjph_imagecodecs_cross_decode` test confirms output is
byte-compatible with the reference.

## Notes for callers

- `encode(arr)` defaults to reversible (lossless).
- `encode(arr, level=<delta>)` selects irreversible lossy with `delta`
  as the quantization base step. Smaller -> closer to lossless, larger
  files. Typical range ~ 0.001 .. 0.1 for natural images.
- `decode_info(bytes)` reads the SIZ marker without sample decoding.
- Output is a raw HTJ2K codestream (.j2c-style); no JP2 box wrapping.
