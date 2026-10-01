# Rice decompressor bounds checks, and where upstream still lacks them

Short version: our vendored Rice decompressors reject a truncated input
in all three pixel widths, and check every byte they read against the
end of the input. Upstream cfitsio 4.7.0 rejects a too-short input in
one of the three; the other two read the first pixel out of a buffer
that may be shorter than the read. In all three, upstream checks for
the end of the input only once per block, so a damaged stream can read
past its buffer inside a block (see "Reads inside the decode loop").

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

## Reads inside the decode loop

The up-front guards cover the first pixel only. After it, upstream
checks for the end of the input once per coding block, after decoding
the block. Within a block nothing stops the reads, and one of them has
no bound at all: a Rice code starts with a run of zero bits, and the
decoder keeps reading bytes until it finds a one. A damaged or
truncated stream whose remaining bits are zero sends it on through
whatever memory follows the buffer, until it finds a nonzero byte or
faults. A stream crafted to start with a valid FS code and then only
zeros does this in all three widths.

Our copy checks each byte before reading it (`RICE_NEED_BYTE` in
`ricecomp.c`) and returns `RCOMP_ERROR_EOS` at the end of the input.
A valid stream never reads past its last byte, so what it decodes to
is unchanged; the extra comparison costs 3 to 8 percent of decode
time in our measurements.
`tests/test_wrapper_streams.py::test_rcomp_decoder_stops_at_the_end_of_its_input`
puts such streams, and every truncation of real ones, against a page
with no access rights, so a read past the end crashes the test's child
process instead of passing unnoticed.

## Related

The wider bounds story for this vendored code is in `3rdparty/VENDOR.toml`
and `THIRD-PARTY.md`. The PLIO decoder had the same class of problem, and
worse: it crashed with SIGBUS on malformed input because our copy
predated upstream's `srclen` checks entirely. `ci/check_vendor_drift.py`
exists so that kind of lag is visible rather than discovered by a crash.
