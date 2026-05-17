# opencodecs benchmarks

Performance setpoints + regression harness for the codec wrappers we
ship. Locks in the perf wins we've measured against `imagecodecs` so
future commits can't silently regress them.

## Quick start

```bash
# measure + print a table of oc/ic ratios
python bench/bench_codecs.py

# enforce the captured ratios stay within +30% of baseline
python bench/bench_codecs.py --check

# re-record after an intentional perf change
python bench/bench_codecs.py --update-baseline

# focus on a subset
python bench/bench_codecs.py --only png,deflate,zstd
```

The PNG-only fast bench (also has --check) is `bench/bench_png.py`.
Use it for tight feedback loops on libspng / libdeflate edits.

## How baselines work

Setpoints live in `bench/perf_baseline.<machine>.json` — one file per
architecture because compilers autovectorise to different widths
(NEON 128b on arm64, SSE2 128b / AVX2 256b on x86_64). The committed
baselines were captured 2026-05-17 on:

| machine    | architecture | system                       |
|------------|--------------|------------------------------|
| M1 Ultra   | arm64        | macOS 15.5, clang from Xcode |
| threadripper | x86_64     | Ubuntu, gcc 13               |

`--check` compares the current run to the baseline keyed by
`platform.machine()` and fails if any encode-or-decode ratio drifts
more than +30% above its recorded value. 30% is intentionally loose:

* normal build-to-build / CI-runner jitter is <5%
* the kinds of regressions we want to catch (whole fast-path goes
  dark, wrong library gets linked, an autovectorisation pattern
  breaks) move ratios by 2-3×

So a regression that trips `--check` is real, not noise.

## What's measured

Each codec gets a single workload chosen to be *meaningful for that
codec* — random bytes are the worst case for any compressor, so we
use natural-image data for image codecs and smooth float fields for
scientific compressors. The Kodak24 photo corpus (downloaded by
`tests/download_test_corpus.sh`) is the source where available.

Workload keys (`codec/data`):

* `png/kodak_photo`, `qoi/kodak_photo`, `jpegls/kodak_photo`, ...
* `deflate/kodak_bytes`, `zstd/kodak_bytes`, `lz4/kodak_bytes`, ...
* `zfp/smooth_f32_3d`
* `blosc2/kodak_bytes`

Workloads that fail to set up on a given platform (codec backend not
built, upstream encoder unavailable) are reported under
`# skipped N workload(s)` and dropped from the run.

## Updating baselines

After an intentional perf change:

```bash
# bench locally on each machine you care about
python bench/bench_codecs.py --update-baseline       # arm64
ssh threadripper "...python bench/bench_codecs.py --update-baseline"  # x86_64

# pull the threadripper baseline back
scp threadripper:.../bench/perf_baseline.x86_64.json bench/

# commit both .json files together
git add bench/perf_baseline.{arm64,x86_64}.json
git commit -m "perf: record new setpoints after <change>"
```

## Known quirks

* **LERC is omitted.** Both `opencodecs` and `imagecodecs` vendor
  their own LERC build, and whichever resolves dylib symbols second
  ends up calling into the first's allocator → SIGABRT in
  `ic.lerc_encode` on certain inputs. Run LERC benches via a separate
  Python process if needed.

* **Per-codec defaults differ between libraries.** A few entries look
  dramatic in either direction:
  * `avif/kodak_photo` is ~90× faster than `imagecodecs` because we
    default to a fast preset; ic defaults to high-quality.
  * `brotli/kodak_bytes` is ~200× slower because we default to level
    11 (max quality / slow); ic defaults to level 1.
  * `jpeg2k/kodak_photo` on x86_64 is 1.8× slower for the same
    underlying reason — different default precincts / progression.

  These are not bugs — they're documented-defaults differences.
  `--check` baselines them so any *further* drift is caught.
