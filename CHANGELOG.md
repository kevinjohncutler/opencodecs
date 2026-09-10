# Changelog

## 0.2.0

126 commits since 0.1.13. The theme is closing the gap between what the
capability manifest claimed and what the code did, in both directions,
plus the parallel-decode work that came out of measuring rather than
reading.

### Breaking

- **`avif.decode()` on an image sequence returns every frame**, shaped
  `(frames, H, W, C)`, where it previously returned the first image and
  said nothing about the rest. A still is unchanged, `out=` on a still is
  unchanged, and `out=` on a sequence is now refused rather than quietly
  filled with frame 0. This matches `gif`, which has always returned the
  stack for a time sequence, and matches what imagecodecs returns for the
  same file. If you relied on getting one frame, use `open()` and index it.

  `heif` deliberately does NOT change: a HEIF is a SET with a
  standard-designated primary image, not a time sequence, and its members
  need not share a shape.

### Added

- **Animated WebP** (`webp`), **AVIF image sequences**, and **every
  top-level image of a HEIF** (bursts, Live Photo stills, depth beside
  color). Previously each returned one image, or in WebP's case failed
  with an unexplained "WebP decode failed".
- **Whole-slide pyramids for two formats.** DICOM VL Whole Slide
  Microscopy series (`open_pyramid(dir, format="dicom")`), where a slide
  is a series of instances rather than a file; and VSI/ETS
  (`open_pyramid("slide.vsi")`), a genuine tiled pyramid where a region
  decodes only the tiles it covers.
- **zfp fixed-rate block access**: `decode_block(data, n)` reaches one
  4x4x4 block without touching the rest, and `block_grid()` reports the
  geometry.
- **BCn band decode**: `decode_rows()` decodes a horizontal band of a
  texture alone.
- **JPEG 2000 region and tile decode**: `decode_region()` and
  `decode_tile()`.
- **Every codec accepts an `http(s)` URL** wherever it accepts a path.
  Formats that can seek fetch only what they need; whole-codestream
  formats fetch once, which is honest when every byte is needed.
- Offset reads for **DM** (`.dm3`/`.dm4`), **NRRD** slices, **NIfTI**
  planes, and a native **`.npy`** reader.
- **HTJ2K** (`htj2k`) now builds OpenJPH from source and ships, including
  Linux aarch64.

### Performance

- **Five codecs were holding the GIL** through their decode and so
  serialized every threaded caller while passing every correctness test:
  `jpeg2k`, `mozjpeg`, `charls`, `zfp`, `openjph`. All measured 0.94x to
  1.12x on eight threads before, ~7x after. Found by measuring each
  codec, not by reading the bindings: openjph's `.pxd` already declared
  the shim `nogil`, which means "safe to call without the GIL", not
  "called without it".
- **Parallel decode** where the pieces were already independent: FITS
  compressed-image tiles (HCOMPRESS 603 -> 87 ms), HDF5 chunks (134 -> 13
  ms), DICOM frames, EER frame accumulation (1124 -> 214 ms), BCn bands
  (BC7 4096x4096, 70 -> 6 ms), zfp blocks (110 -> 30 ms end to end).
- **gzip decode allocates once** from the ISIZE in the trailer: 148 MB
  peak to 67 MB for a 67 MB result, and 13.0 ms to 6.8 ms.
- TIFF: a byte swap that swapped nothing (half the cost of an
  uncompressed read), and worker counts sized by output bytes.
- Earlier in the cycle: RGBE decode 1.7x, a replacement EER decoder 1.7x,
  a replacement TIFF LZW encoder 1.4x.

### Fixed

- **cfitsio `fits_hdecompress` is not thread-safe.** It keeps the bit
  reader's position in three file-scope variables, so overlapping
  HCOMPRESS_1 tiles corrupted each other and returned status 414, a
  format error for a concurrency bug. The vendored copy marks them
  thread-local; single-threaded behavior is byte-identical.
- Rice decompressor bounds guards for the short and byte variants, which
  upstream cfitsio still lacks.
- Capability flags that were wrong in both directions: `chunked` claimed
  by EER while indexing walked the file (1200 ms to reach frame 720 of
  721, now 1.70 ms), and `chunked`/`range_reads`/`streaming_decode`
  denied by DICOM, FITS, MRC, OIB, VSI and OIR which had them all along.
- `streaming_decode` presupposes a frame axis; fifteen byte-compressor and
  single-image codecs carried it as a gap they could never close, and the
  checker now enforces the invariant.

### Tooling

- `capabilities.toml` has no open gaps: 25 capabilities done, 35 recorded
  as not applicable with reasons, verified against the code on every CI run.
- Corpus manifest records a license source and note per dataset. 26 of 40
  carry terms; the other 14 each say what was checked and why it did not
  resolve, rather than a bare "unverified".
- `corpus.py coverage` reports what is on disk, not what the manifest
  declares.
