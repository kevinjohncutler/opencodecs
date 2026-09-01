# Codec performance audit

A sweep of every codec we and imagecodecs both implement, to find where
we lose. Kodak photo for image codecs, its raw bytes for the byte
compressors, on Apple Silicon.

## Result

| Faster | Parity | Slower |
|---|---|---|
| bmp encode 7.8x, jpeg2k encode 4.2-4.8x, jpeg2k decode 4.1-4.6x, png encode 3.2x, deflate decode 2.0x, brotli encode 1.4x, png decode 1.2x, rgbe decode 1.5x | qoi, jpeg, zstd, snappy, bitshuffle, blosc2, deflate encode, brotli decode, webp decode, bmp decode | webp encode 0.81-0.89x |

One real gap: **webp encode**, reproducibly ~15% behind. Output size
matches to within 20 bytes of 332 KB, so it is not a quality or method
difference; it is overhead somewhere in how we drive libwebp. Worth a
profile, not worth a guess.

## How to measure this without fooling yourself

The first version of this sweep reported ten losing lanes. Nine were
noise. What it took to get a trustworthy answer:

- **Warm both sides before timing.** A cold `zstd` encode measured
  5.6 ms and a warm one 3.2 ms, which is larger than any real
  difference on this list.
- **Interleave A/B/A/B** rather than timing all of A then all of B. The
  machine drifts, and a block layout turns drift into a fake result.
- **Take the minimum of many**, not the mean. The mean measures the
  machine's other work.
- **Run the whole sweep twice and compare.** Anything that moves between
  runs is not a finding. This is what demoted nine of the ten.
- **Do not run it next to a build.** Two earlier flakes in this repo, a
  PNG timing test and the HTTP range tests, are the same failure.

The `ab()` helper in the sweep script does the first three. The fourth
is the one that matters most and is easiest to skip.

## Where our wins come from

Recorded so the pattern is reusable rather than incidental:

- **png**: our own SIMD defilter replacing libspng's scalar loop.
- **jpeg2k**: threading defaults, and skipping a redundant copy.
- **rgbe**: a compile-time exponent table replacing a per-pixel `ldexp`,
  which is where the 1.5x on real files comes from.
- **deflate decode**: libdeflate instead of zlib's streaming loop.

The recurring shape is an expensive per-element operation that can be
hoisted into a table or a wider primitive. A scan for libm calls inside
per-pixel loops across the Cython sources found only one remaining, a
`log2f` in the UHDR gain-map encoder, and the `gamma == 1.0` fast path
already skips its neighboring `powf`.
