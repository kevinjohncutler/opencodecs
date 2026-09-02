# HTJ2K conformance: where we stand

Measured 2026-09-01 against the JPEG committee's own conformance
bitstreams ([gitlab.com/wg1/htj2k-codestreams](https://gitlab.com/wg1/htj2k-codestreams)),
all 42 HT codestreams from `htj2k_bsets_profile0`, `htj2k_bsets_profile1`
and `codestreams_hifi`. Fetch them with:

    python corpus/corpus.py fetch htj2k_wg1_conformance

`tests/test_htj2k_conformance.py` pins the score so it cannot move
without someone noticing.

## The score

| Outcome | Count | Why |
|---|---|---|
| Decodes | 7 | single quality layer, no unread markers |
| Refused: multiple quality layers | 28 | OpenJPH implements one layer only |
| Refused: unread marker segment | 1 | QCD inside a tile; see below |
| Refused: other | 5 | POC/RGN markers, missing EPH, tile header, one non-HT file correctly rejected |

The headline is that **28 of the 35 refusals are a single missing
feature**. Multiple quality layers is not a corner case: it is JPEG
2000's quality-scalability mechanism, the reason the format is chosen
for archives and streaming, and files carrying 5, 8, 19 or 30 layers are
ordinary.

## It is upstream, not our wrapper

The refusal is raised in OpenJPH's own codestream parser the moment it
reads a COD marker whose layer count is not 1, before any block
decoding happens:

    // ojph_codestream_local.cpp
    if (num_qlayers != 1)
      OJPH_ERROR(0x00030053, "The current implementation supports "
        "1 quality layer only.  This codestream has %d quality layers", ...);

We link 0.27.2. That code is unchanged in 0.31.0 (July 2026), the newest
release, and it is still a hard error rather than a warning, so a
version bump does not help.

Nor does the other decoder we already ship. Running the same 42 files
through our `jpeg2k` codec (OpenJPEG, which gained HTJ2K support in 2.5)
decodes 10, a strict superset of OpenJPH's 7 plus the non-HT file, and
fails the same layered files with a bare `opj_decode failed`. So the
union of both decoders in this package is 10 of 42.

## What a conformant decoder scores

[OpenHTJ2K](https://github.com/osamu620/OpenHTJ2K) (BSD-3-Clause, by
Osamu Watanabe) decodes **41 of 42**. The single failure is a
segmentation fault on `ds0_ht_13_b11.j2k`, a file OpenJPH also rejects,
so that one may be a genuinely pathological codestream; a crash is still
a crash and would need containing before shipping it.

Spot-checking its output against the committee's PGX reference images,
the components whose dimensions line up match exactly or within one
level. A full conformance verdict would need the per-test descriptions,
which specify the decode reduction level and the allowed tolerance; the
references are stored at reduced resolution, so a naive comparison
reports shape mismatches that are not errors.

## What we fixed on our side

Three problems in our OpenJPH wrapper, all found by running this corpus:

**Silent wrong pixels.** OpenJPH treats several unimplemented marker
segments as warnings and keeps decoding, so the caller receives an image
built from a codestream the library admits it did not fully read. One
conformance file decoded 954 levels away from the reference this way.
`decode()` now raises `OpenJphUnsupportedFeature` when the library
reports skipping something, with `ignore_unsupported=True` to override.
Returning data the decoder has disclaimed is worse than failing.

**Unusable error messages.** OpenJPH's default error handler prints the
reason to stdout and throws a generic exception, so every failure
surfaced as `ojph error (rc=2)`. The shim now installs a collector, and
the reason reaches the Python exception: you get "this codestream has 5
quality layers" instead of a number.

**Output corruption.** The default handlers write to *stdout*, not
stderr. A library that prints to a process's stdout corrupts whatever
that process was writing there, which is how this was noticed: it broke
a JSON parse in a test harness. Both handlers are now intercepted, and
a test asserts nothing leaks.

## Closing the gap

Two routes, neither started:

1. **Implement multiple quality layers in OpenJPH and upstream it.** The
   HT block decoder needs no changes; the work is in the packet reader,
   where each precinct emits one packet per layer and a codeblock's
   coding passes accumulate across them. Well specified, but it is real
   C++ in someone else's parser.
2. **Add OpenHTJ2K as a second HTJ2K backend** and route to it when the
   codestream declares more than one layer. Cheaper, and the license is
   compatible, but it is another dependency to build on every platform,
   and the segfault above says its input validation needs review before
   we hand it untrusted data.
