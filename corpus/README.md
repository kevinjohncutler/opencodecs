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
python corpus/corpus.py coverage    which codecs have real data behind them
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
- **Codec mapping.** `coverage` answers "which codec has no real data
  behind it" directly. That was previously an ad-hoc grep, and the
  ad-hoc grep was wrong: it counted any codec mentioned in a corpus
  test, which over-reported.

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
- **opt-in**: large single files, like the 220 MB EER micrograph.
  `download_test_corpus.sh --eer`.

Tests that need corpus files skip cleanly when the file is absent, so a
fresh clone with no downloads still runs a full green suite.
