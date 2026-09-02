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
