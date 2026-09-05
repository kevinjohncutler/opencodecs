# Codec performance audit

Every codec we build, measured on real corpus data. Run it with
`bench/sweep.py --compare`.

## Result

**No confirmed performance losses.** Across 29 measurable codecs we are
ahead in eight lanes, at parity in the rest, and not behind anywhere
that survived repeat measurement.

| Confirmed faster | |
|---|---|
| bmp encode | ~8x |
| jpeg2k encode / decode | ~5x / ~4x |
| png encode | ~3.2-3.7x |
| deflate decode | ~2.0x |
| rgbe decode | ~1.5x |
| brotli encode | ~1.4x |
| avif encode | ~1.3-1.4x |
| jpeg encode, png decode | ~1.1-1.2x |

Parity: qoi, zstd, lz4, snappy, lzma, bz2, bitshuffle, blosc2, deflate
encode, brotli decode, bmp decode, pcodec, zfp, rgbe encode.

Not measured: `isal` is x86-only and does not build on arm64. `mozjpeg`,
`jxl`, `heif`, `brunsli`, `openjph`, `sperr`, `aec`, `eer`, `ndtiff` and
`uhdr` have no comparable imagecodecs entry point at the same settings,
so they carry absolute numbers only.

## Two results that look like losses and are not

`sz3` and `lerc` appear 5x and 2x slower until you look at what each
side produced:

| codec | our output | their output |
|---|---|---|
| sz3 | 1,885,737 B (2.12x) | 3,709,732 B (1.08x) |
| lerc | 3,315,340 B (1.21x) | 4,000,079 B (1.00x, no compression) |

We compress roughly twice as hard at the same call, and the speed
difference is the price of that rather than a defect. `bench/sweep.py`
now withholds the ratio when the two outputs differ by more than 10% and
prints `n/c`, because a speed number spanning different work is worse
than no number.

## How to measure this without fooling yourself

The first version of this sweep reported ten losing lanes. **All ten were
wrong**: nine were noise, the tenth was the settings mismatch above.
What it took to get an answer worth acting on:

- **Warm both sides.** A cold `zstd` encode measures 5.6 ms against
  3.2 ms warm, larger than any real difference on this page.
- **Interleave A/B/A/B.** Timing all of A then all of B turns machine
  drift into a fake result.
- **Minimum, not mean.** The mean measures the machine's other work.
- **Repeat the sweep and discard anything that moves.** This is the step
  that does the work and the easiest to skip. `webp` encode read 0.84x,
  then 1.06x, then 0.75/0.95/1.04 over three more runs. Unstable means
  unmeasured, not slow.
- **Check both sides did the same job** before believing a ratio.
- **Do not run next to a build.** The two flaky tests in this repo, a
  PNG timing assertion and the HTTP range tests, are the same failure.

## Where our wins come from

- **png**: our own SIMD defilter replacing libspng's scalar loop.
- **jpeg2k**: threading defaults and one avoided copy.
- **rgbe**: compile-time exponent tables replacing per-pixel `ldexp` on
  decode and `frexp` on encode.
- **deflate decode**: libdeflate instead of zlib's streaming loop.

The recurring shape is an expensive per-element operation hoisted into a
table or a wider primitive. A scan of every Cython source for libm calls
inside per-pixel loops found one remaining, a `log2f` in the UHDR
gain-map encoder, whose neighboring `powf` an existing `gamma == 1.0`
fast path already skips. There is no second rgbe waiting to be found.

## An API trap worth knowing

`webp` encode defaults to **lossless** and silently ignores `level=`
unless `lossless=False` is also passed. `encode(img, level=10)` returns
a 332 KB lossless file, not a small lossy one. imagecodecs behaves the
same way, so this is compatibility rather than a bug, but it surprises
people and it is why webp encode measures ~190 ms here rather than ~5 ms.


## Container-format readers vs their reference libraries (2026-09-03)

Every format added in the scientific-reader work has a canonical Python
implementation, so these are real comparisons rather than
self-comparisons. Warm, minimum of seven, on the corpus fixtures.
Reproduce with `bench/bench_readers.py`.

| Format | Reference | Ours (ms) | Reference (ms) | Speedup |
|---|---|---|---|---|
| dm | rosettasciio | 0.75 | 5.19 | 6.95x |
| dicom | pydicom | 0.29 | 0.70 | 2.45x |
| mrc | mrcfile | 0.07 | 0.17 | 2.39x |
| nrrd | pynrrd | 0.04 | 0.07 | 1.63x |
| emd | rosettasciio | 2.57 | 3.87 | 1.50x |
| nifti | nibabel | 1.20 | 1.58 | 1.32x |

Read with the caveat the script itself states: these are small files, so
a good part of the gap is per-call overhead in libraries that build rich
domain objects where we return an ndarray. That overhead is real for a
caller opening thousands of files and is not the same claim as being
faster per megabyte on one large one.

### Streaming, and where it is missing

MRC, DICOM and NRRD read through the `read_at` contract, so all three
open over HTTP without downloading the file. N5 and Imaris reach their
data through a store and through HDF5 chunking respectively, which gets
the same effect by a different route.

* **MRC** costs 64 KB to open a 4 MB volume, and one plane read stays
  well under half the file.
* **DICOM** and **NRRD** did not stream and now do. Both put native
  pixel data at a fixed offset after a header, which is exactly the
  shape range requests are for. NRRD parses its text header from one 64 KiB
  read and then fetches only the volume; DICOM parses its dataset from
  the same-sized prefix and reads a single frame by offset, so frame 40
  of a 48-frame series costs one frame rather than forty-one. The
  encapsulated path walks only the 8-byte fragment headers to locate a
  frame, rather than reading every fragment before it.

All three now reach storage through one helper,
`core._io_helpers.open_read_at`, which turns a path into a seek, an
http(s) URL into range requests and bytes into slices. MRC had grown its
own copy of that; there is one now. `tests/test_reader_streaming.py`
asserts the byte savings for each, because "the URL opened" is not the
claim worth testing.

The two that still read everything do so for reasons that belong to the
formats. Nearly every NIfTI in circulation is gzipped and a gzip member
has no usable random access, so reaching the last slice means inflating
everything before it; for a bare `.nii` the whole-file read is one
syscall anyway. DM has no header describing the image at all, so the tag
tree has to be walked from the front before anything can be located.

### Correctness, which the speed table above assumed (2026-09-03)

An external review of the reader work found four defects, all of them
real, and checking it turned up two more. Recording them because the
pattern behind them is more useful than the individual fixes.

| Format | Defect | How it hid |
|---|---|---|
| NRRD | Axes reversed twice, so every non-cubic volume came back scrambled | Fixture and reader shared the assumption; the corpus volume is a 30x30x30 cube |
| DM | dm4 header offsets off by eight | Both corpus fixtures are dm3 |
| DICOM | Explicit VR big-endian read little-endian throughout | No big-endian fixture |
| DICOM | Deflated datasets inflated from a 64 KiB prefix | No deflated fixture |
| DICOM | Implicit VR read Columns 48 as the ASCII "0" it spells | No implicit-VR fixture |
| DICOM | Undefined-length items walked as if they were items | No such sequence in the corpus |

Every one is a gap in what the fixtures covered rather than a mistake in
reasoning about the spec, and two of them are worse than that:

**A test that writes its own fixture tests nothing about the format.**
The NRRD fixture builder wrote Fortran-ordered bytes and the reader read
them back Fortran-ordered, so they agreed with each other and disagreed
with pynrrd, ITK and Slicer. There *was* a pynrrd comparison test, and it
passed, because `reshape(sizes).T` equals `reshape(sizes, order="F")`
exactly when `sizes` is its own reverse -- which a 30x30x30 phantom is.
The fix is `tests/test_nrrd.py::test_axis_order_on_a_volume_that_is_not_a_cube`:
pynrrd writes a 2x3x5 volume, we read it. Three different extents, and
the reference library holds the pen.

**A corpus of real files is not coverage of the format.** Two real .dm3
files exercise the tag walk thoroughly and say nothing about dm4, whose
header this reader had wrong the whole time. Synthetic files are the
complement, not the poor substitute: they can be minimal, and they can
be wrong on purpose. `tests/test_dm_synthetic.py` builds both versions;
`tests/test_dicom_encodings.py` builds all four transfer syntaxes and
cross-checks each against pydicom.

### One reader interface, actually implemented

The same review noted that `Codec.open` is documented to return a
`Reader` and did not. MRC, NIfTI, NRRD, DM and EMD each returned their
own format class with no `read()` and no `iter_frames()`, and DICOM was
not registered as a codec at all, so `oc.read("study.dcm")` failed while
every other format in the package answered to it.

`core.codec.ArrayReader` now supplies the interface for the formats
built around `asarray()`, and they all inherit it. A format that can
fetch one plane without the others says so by defining `_frame`, so
`iter_frames` on an MRC or a DICOM series streams a plane at a time
rather than materializing the volume and slicing it. For DM and EMD,
which hold separate images that need not share a shape, a frame is a
whole image and `shape_at(i)` / `dtype_at(i)` reach the others.

`tests/test_reader_contract.py` runs the same contract over all of them,
which is the part that keeps it true.

### The audit that followed, and the gap it found (2026-09-03)

Having found the six defects, the obvious question was how many more of
the same kind were sitting in the readers nobody had cross-checked. So
every remaining container reader was compared against an independent
implementation, on volumes asymmetric in every axis.

| Format | Reference | Result |
|---|---|---|
| N5 | tensorstore | Correct. Differs from tensorstore by exactly a transpose, which is a convention choice, not a layout error |
| OME-Zarr | zarr v2 and v3 | Correct, including ragged chunk grids and straddling sub-regions |
| Imaris | raw h5py | Correct; the padded stored array is trimmed to the declared extent |
| NDTiff | the ndtiff package | Correct, all 128 frames |
| DICOM corpus | pydicom | Correct, all six files, including RLE and JPEG 2000 |
| FITS | astropy | Already cross-checked; astropy writes the fixtures |
| EER | imagecodecs | Already cross-checked on real Falcon4 data |

No further defects. The N5 result is worth stating precisely because it
looked like one at first: N5 dimension 0 varies fastest, and we present
the reverse so the fastest axis is last, which is what C order means and
what pynrrd, FITS and z5py do. tensorstore keeps the file's own order.
Both are right about the bytes, and `test_reference_parity.py` pins the
transpose relation rather than either convention in isolation.

**The gap was not in a reader. It was that none of this ran in CI.**
Every cross-check in the suite is an `importorskip`, and `.[test]`
installed none of the reference implementations: not pydicom, pynrrd,
nibabel, mrcfile, astropy or rosettasciio. Twelve formats' worth of
independent verification reported success while testing nothing, which
is precisely how a green suite carried six reader bugs.

There is now a `reference` extra, CI installs it one package at a time
so a missing wheel costs one comparison rather than the run, and one job
sets `OPENCODECS_REQUIRE_REFERENCE=1`, which turns a missing core
reference reader from a silent skip into a failure. The suite also
prints what is and is not being cross-checked on every run.

### Is the filter_scanline trick applicable anywhere else? (2026-09-04)

The libspng win came from a specific shape: a per-element loop whose body
branches on something the loop cannot change. Worth asking how much more
of that there is, so the rest of the codebase got scanned for it.

Two searches, over every `.c`, `.h` and `.pyx` under `3rdparty/` and
`src/`: loops containing a `switch`, and loops containing an `if` whose
condition names nothing the body assigns. 13 and 35 candidates
respectively. **None of them is worth changing**, and the reasons are
worth writing down because they are the criteria, not excuses.

The transformation only pays when all three hold:

1. **The loop is per-element**, not per-row or per-frame. A branch
   amortized over a whole row costs nothing. `_gif.pyx`'s
   `if bg_r == 0 and ...` looked like a hit and is a per-frame choice
   between two vectorized numpy fills.
2. **The branch is genuinely loop-invariant.** Most of the `switch` hits
   are in cfitsio's H-compress and PLIO, switching on `a[k]` or
   `b[s00]` -- the data being decoded. There is nothing to hoist.
3. **The body is otherwise vectorizable.** This is the one that
   disqualifies the most plausible-looking candidates:
   * `_uhdr.pyx`'s gain-map kernel has an invariant `if use_gamma` in a
     per-pixel loop, but the body calls `log2f` per pixel. That
     dominates, and no branch removal makes the loop vectorize.
   * `qoi.h`'s encode loop has an invariant `if (channels == 4)` over
     pixels, and clang emits no vectorization remark for that loop at
     all: QOI is a run-length-and-index codec where each pixel depends
     on the previous one and a running hash table. Serial by
     construction.
   * `_bcdec.pyx` dispatches on `fmt_id` per 4x4 block, so the branch is
     already amortized ~16:1 over a real function call.

Verified with the compiler rather than by argument
(`clang -Rpass=loop-vectorize -Rpass-missed=loop-vectorize`):

| loop | before | after |
|---|---|---|
| libspng `filter_scanline` | `loop not vectorized` | 7 loops, width 16 |
| libspng `filter_sum_*` | (already done earlier) | 8 loops, width 16 |
| qoi encode | no remark at all | unchanged, serial |
| `_bmp.pyx` swizzle | already vectorized | unchanged |

So `filter_scanline` was the outlier: the only place in the tree that
combined a per-byte loop, cheap arithmetic, and an invariant branch.

### Asking the compiler instead of guessing the shape (2026-09-04)

The scan above looked for the shape that had just been fixed, which only
finds failures whose cause is already known. The better question is the
outcome one: **which loops does the vectorizer refuse, and why?**
`-Rpass=loop-vectorize -Rpass-analysis=loop-vectorize` answers it
directly, and the reasons differ, so the fixes would differ too.

Over every vendored `.c` we compile plus every Cython-generated `.c`:
641 loops vectorized, 4125 refused. The raw count is meaningless --
Cython emits ~130 refused loops per file for refcounting and argument
marshalling, and a thin wrapper whose real work is one call into
libdeflate does not care. Mapping each remark back through Cython's
`/* "....pyx":NNN */` comments and keeping only the `nogil` kernels
leaves 17 loops that were written to be fast:

| kernel | vectorized | why not |
|---|---|---|
| `_bytetools` byteshuffle | 6 | — |
| `_uhdr._upscale_gainmap_kernel` | 2 | — |
| `_uhdr._rgb_to_rgba_pack_kernel` | 1 | — |
| `_uhdr._sdr_from_hdr_kernel` | 0 | cannot identify array bounds |
| `_uhdr._srgb_eotf_lut_np` | 0 | call instruction (libm) |
| `_jxl._decode_one_frame*` | 0 | unsupported switch (decoder driver, not per-pixel) |

The one worth chasing was `_sdr_from_hdr_kernel`, whose docstring
claimed it "compiles to tight NEON/SSE2/AVX2 with gather". It does not:
the LUT index is data-dependent, so the load is a gather, and arm64 has
no gather instruction.

**And the kernel is still right.** Measured against a branch-free sRGB
OETF written as arithmetic, which does vectorize at width 8, on a
4.19 Mpx RGB frame:

    scalar LUT gather      8.7 ms
    vectorized polynomial 65.7 ms      0.13x

Vectorized `exp2f`/`log2f` is far more expensive than a scalar table
lookup, so the table wins by 7.5x while losing the vectorizer entirely.
The docstring was corrected; the code was not.

Two things worth keeping from this. Not vectorizing is not a defect --
`_gain_map_kernel` next door deliberately uses a polynomial `_fast_log2`
*so that* it can vectorize, and that is the same trade decided the other
way for a cheaper function. And a performance claim in a comment is
worth checking against the compiler: this one had been wrong for as long
as it had been written, and it points the next reader at the wrong work.
