# Rice decompressor bounds checks, and where upstream still lacks them

Short version: our vendored Rice decompressors reject a truncated input
in all three pixel widths. Upstream cfitsio 4.7.0 rejects it in one of
the three. The other two read the first pixel out of a buffer that may
be shorter than the read.

## Why the checks are absent upstream at all

cfitsio's own `ricecomp.c` explains it:

> Note that beginning with CFITSIO v3.08, EOB checking was removed to
> improve speed, and so now the input compressed bytes buffers must have
> been allocated big enough so that they will never be overflowed. A
> simple rule of thumb that guarantees the buffer will be large enough is
> to make it 1% larger than the size of the input array of pixels that
> are being compressed.

That is a reasonable contract for a library reading files it wrote
itself. It is the wrong contract for a codec handed arbitrary bytes off
a network or out of a user's file, which is what we are.

## The state of upstream 4.7.0

Each decompressor reads the first pixel unencoded, directly from the
head of the input, before decoding anything:

| Function | Bytes read up front | Guard in 4.7.0 |
|---|---|---|
| `fits_rdecomp` (int) | 4 | yes, `clen < 4` at line 931 |
| `fits_rdecomp_short` | 2 | **none** |
| `fits_rdecomp_byte` | 1 | **none** |

The `clen < 4` guard landed upstream on 2025-03-03. The short and byte
variants were not given the equivalent, so passing a zero- or one-byte
buffer to either reads past its end.

## What we carry

All three guarded, returning `RCOMP_ERROR_EOB` rather than reading:

    if (clen < 4)  ...  /* fits_rdecomp     */
    if (clen < 2)  ...  /* fits_rdecomp_short */
    if (clen < 1)  ...  /* fits_rdecomp_byte  */

The 4-byte form is a backport of the upstream fix, picked up in "Security:
pick up two cfitsio bounds checks we were missing" after the vendored
copy was found to predate it. The 2-byte and 1-byte forms have no
upstream equivalent; they are ours.

`tests/test_rcomp_truncated_input.py` pins this, feeding each variant
fewer bytes than its first pixel needs.

Worth reporting to HEASARC. Two of these are a one-line fix each, in the
same shape as the change they already made to `fits_rdecomp`.

## Related

The wider bounds story for this vendored code is in `3rdparty/VENDOR.toml`
and `THIRD-PARTY.md`. The PLIO decoder had the same class of problem, and
worse: it crashed with SIGBUS on malformed input because our copy
predated upstream's `srclen` checks entirely. `ci/check_vendor_drift.py`
exists so that kind of lag is visible rather than discovered by a crash.
