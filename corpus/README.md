# Reference corpus

Real files, from their original homes, for testing codecs against data a
microscope or camera actually produced rather than data we synthesized.

**Nothing here is redistributed.** `manifest.toml` records where each
file lives upstream; `corpus.py` fetches from origin into the gitignored
`.test_data/`. That keeps the repository small and sidesteps the
question of whether we are allowed to rehost a vendor's sample file,
while still letting anyone reproduce the exact corpus.

```
python corpus/corpus.py list        what is in the manifest, and what is on disk
python corpus/corpus.py coverage    which codecs have a native-format fixture
python corpus/corpus.py fetch       download everything (or: fetch <id> ...)
python corpus/corpus.py verify      re-hash what is on disk against checksums.json
python corpus/corpus.py freeze      record checksums for newly fetched files
```

## Why a manifest and not just the shell script

`tests/download_test_corpus.sh` still does the downloading for the
common cases and will keep working. The manifest adds three things it
cannot:

- **Checksums.** Upstream files change. Without a recorded digest that
  is a silent difference in test results, not an error.
- **Licenses.** Written down per dataset, so if we ever do want to
  mirror something the answer is already recorded. `unverified` means
  nobody has confirmed the terms yet; it is a TODO, not a claim.
- **Codec mapping.** `coverage` answers "which codec has no file in its
  own format, written by somebody else" directly. That was previously an
  ad-hoc grep, and the grep was wrong twice over: it counted any codec
  mentioned in a corpus test, and it conflated two different things.

The two will drift if left alone, so `tests/test_corpus_manifest.py`
asserts every URL the script fetches is in the manifest. The intended
end state is the script becoming a thin wrapper over `corpus.py`.

## Adding a dataset

Append a `[[dataset]]` block with an `id`, a `name`, the `license`, the
`codecs` it exercises, and one `[[dataset.file]]` per file. Then
`corpus.py fetch <id>` and `corpus.py freeze` to record its checksum.

Prefer files that stress a decoder rather than merely exercise it: the
EER entry is there because a real Falcon 4 frame terminates by landing
exactly on its last cell and then carries a footer, and a synthetic
bitstream does not do that. Two of our decoder bugs were only visible
against real data.

## Tiers

Not everything belongs in CI. Roughly:

- **synthetic**: generated in-process, no download. Runs everywhere,
  every push. Most codec tests are here and should stay.
- **light**: small real files, a few MB. `download_test_corpus.sh
  --light`.
- **full**: the whole corpus, hundreds of MB, run locally or nightly.
- **opt-in**: large or specialized sets fetched on request.
  `--eer` for the 220 MB Falcon 4 micrograph, `--sdrbench` for the
  scientific arrays.

Tests that need corpus files skip cleanly when the file is absent, so a
fresh clone with no downloads still runs a full green suite.

## Native fixtures versus round-trips

These are not the same thing and the distinction matters.

`tests/test_corpus_codec_decode.py` already round-trips most codecs
against real Kodak photographs: encode with ours, decode with ours and
with imagecodecs, assert bit-equality for lossless and bounded PSNR for
lossy. That is real data and it catches encoder and decoder disagreeing.

What it cannot catch is somebody else's writer. Every format has habits
that only show up in files produced in the wild, and both of the decoder
bugs found recently were of exactly that kind: a real EER frame ends by
landing on its last cell and then carries a footer, and real Radiance
files use resolution-line orientations we rejected. No amount of
round-tripping our own output would have surfaced either.

So `coverage` reports the narrower thing: codecs with no native-format
fixture. Six today:

| codec | what it would need |
|---|---|
| `bmp` | a conformance set; bmpsuite generates its BMPs rather than shipping them, and is GPL-3.0 |
| `qoi` | covered by the opt-in benchmark suite, but that is 1.1 GB; a smaller set would be better |
| `rgbe` | a Radiance HDR image, e.g. a CC0 environment map |
| `uhdr` | an Ultra HDR JPEG with a real gain map |
| `openjph` | an HTJ2K conformance codestream |
| `bcdec` | a DDS texture using BC1 through BC7 |
| `bytetools` | arguably not applicable; it is a byte-shuffle helper rather than a format |

None of these are hard, they just need a source whose terms are clear.
Prefer conformance sets over pretty pictures: the point is to exercise
the awkward corners of a format, not to have a nice image.
