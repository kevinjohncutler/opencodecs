Changelog
=========

opencodecs is a fork-then-divergence of Christoph Gohlke's
``imagecodecs`` aimed at Pareto-better defaults, native streaming
readers for cloud-backed scientific imaging, and codec coverage
that fits a modern (post-2024) imaging pipeline.

Versions follow the same ``YYYY.M.D`` cadence as upstream when we
publish; the entries below cluster work by date rather than by
release because most of it has shipped continuously to ``main``.

0.5.1 (2026-10-02)
------------------

``czi_recompress`` takes an optional ``should_continue``: a zero-argument
callable polled once per sub-block, before that sub-block is read. When it
returns false the rewrite raises the new ``CziCancelled`` and the partial
output is removed. Existing callers are unaffected; the default is ``None``.

It is a plain callable rather than a ``threading.Event`` so a caller can pass
a deadline, a signal flag or a cancellation token just as easily, and nothing
in the writer needs to know which. ``CziCancelled`` subclasses
``CziWriterError``, so code already catching that keeps working while code
that cares can tell "the user stopped it" from "the file is broken" without
matching on message text.

Why it exists: a caller replacing a subprocess (ZEISS's ``czicompress``) with
this function loses the ability to kill a process mid-file, and a Cancel
button that only takes effect between files is a visible regression on a
slide scan. Polling per sub-block takes effect within roughly ``workers``
sub-blocks, which on a 481-sub-block scan is sub-second.

An interrupted rewrite now removes ``dst`` on any exception, not only
cancellation -- a failed verification or a ``KeyboardInterrupt`` too. A CZI
missing sub-blocks still has a valid header and directory, so readers accept
it and the loss is silent; truncated-but-plausible is worse than absent.

CZI reading
~~~~~~~~~~~

- ``CziReader(path, index=...)`` reads pixels from a payload index (offsets,
  sizes, compressions, plane shape, pixel type) without reading the file
  header, the directory or any sub-block header, and ``payload_index()``
  returns one from an open reader. A caller that stores the index with its
  own metadata reads planes with no metadata round trips over a network
  share.
- ``read(indices=[...])`` reads chosen sub-blocks in parallel into one
  stack. ``attachments()`` and ``read_attachment(name)`` return embedded
  files (Thumbnail, TimeStamps, Label ...) exactly as stored.
- Files on disk are read with positional reads instead of a memory map: a
  cold plane off a network share was paged in a cluster at a time, and one
  exact-range read (split into concurrent pieces of about 1 MiB when only a
  few planes are read) fetches it in a fraction of the time. The file
  header is fetched when first needed, the directory comes in one request,
  and a sub-block's header and payload in one. The directory and metadata
  are read on first use, so a corrupt directory now raises ``CziError`` on
  first use rather than in the constructor.
- Uncompressed stacks are read on several threads. They were split into 8
  MiB tasks, the size zstd reads use, so nine 2 MiB planes made three
  tasks, which the worker policy ran on one thread; they now go one plane
  per 2 MiB task, and two tasks are enough to share. Warm, that stack reads
  1.63x faster than a reader copying from an mmap on 128-core Linux, where
  it had been 0.59x as fast (1.51x and 0.88x on a 20-core Mac). Slide
  regions that need two or three tiles decode them in parallel for the same
  reason.
- JPEG XR (compression 4, most slide scans) decodes through the copy of
  jxrlib maintained inside ZEISS's libCZI (BSD-2-Clause, Microsoft), which
  ZEISS reworked for speed. No compiler flags brought the upstream 2019.10.9
  release within 10% of aicspylibczi on x86-64 Linux; with libCZI's copy a
  slide region read inside one tile takes 0.99x aicspylibczi's time there
  and is 1.08x faster on a 20-core Mac (upstream: 0.90x and 0.93x). The
  wheels build it static, from a pinned libCZI commit, with ``NDEBUG``
  (upstream's Makefile left 68 ``assert()`` calls in the decode loop) and
  link-time optimization. A jxrlib built by ``bench/build_codec_libs.sh``
  is now linked by path, so a system jxrlib earlier on the library path
  can no longer replace it. Upstream jxrlib (Windows wheels, distribution
  packages) still works.
- The hi/lo byte unshuffle of zstd sub-blocks uses a byte loop for 2-byte
  pixels under GCC and Clang, which GCC vectorizes better: 1.46x faster for
  a 1000 x 1000 plane and 1.11x for 2000 x 2000 on x86-64 Linux. Clang ties;
  MSVC keeps the word form.
- Payload buffers are shared and reused, and the read pool holds at most
  the 32 threads one read uses (it was twice the CPU count).

TIFF reading
~~~~~~~~~~~~

- A multi-page TIFF decodes each page straight into its slice of one
  output, instead of decoding pages separately and stacking them, a second
  copy of the whole stack. A stack stored as plain pixels, end to end in
  the file (what writers produce for a contiguous series), is read as one
  span by parallel positioned reads: a 23-page 2000 x 2000 uint16 stack in
  10.8 ms where tifffile takes 33.8 ms (128-core Linux, warm); page by page
  it took 59.6 ms. ``TiffPage.asarray`` takes ``out=``.
- Decode workers are sized by whichever asks for more: 1 MiB of output each,
  or 256 KiB of compressed input each. Output alone held a deflate image
  with 6.8 MB of compressed data in 31 strips to 7 workers.

0.5.0 (2026-10-01)
------------------

Two things: speed, above all on Windows, where much of 0.4.0's work had
arrived only in part; and a pass over every codec opencodecs shares with
imagecodecs, which found and fixed formats and conventions that did not
agree. The rule applied throughout: a published specification wins;
a library's own format (pcodec, SZ3, SPERR, libaec, Rice, LZ4, Snappy)
is written bare, with no header of opencodecs' own, and files earlier
versions wrote still read; where no specification decides, opencodecs
matches imagecodecs; and no parameter imagecodecs defines is silently
ignored, and no data is silently lost: each is implemented or raises.

Several fixes change the bytes opencodecs writes (each says so): the
``floatpred`` codec, ``delta`` on floats, the five codecs that wrapped a
library's stream in a header of their own, and some defaults. Data the
old codecs wrote still decodes where the old form can be told apart;
where it cannot (float ``delta``), the bullet says how to read it.

Removed: ``opencodecs.tifffile_patch``, which swapped opencodecs codecs
into tifffile's dispatch. tifffile 2026.8.23 and later refuse codec
functions from any module other than imagecodecs and tifffile, so the
patch fails there with ``RuntimeError``. Read and write TIFF with
opencodecs directly (``opencodecs.read``, ``opencodecs.tiff_imwrite``,
``opencodecs.TiffWriter``); the README compares that reader's speed with
tifffile's.

Speed
~~~~~

The speed changes are the same code on every operating system and
compiler, in a form each compiler measured builds well, except two
below: bitshuffle's build guard, which now lets MSVC take the SSE2 path,
and how uncompressed strips are read among other reads. The Linux wheels
also link newer libraries (last bullet but one). Figures compare
the published 0.4.0 wheels with 0.5.0's, each version's CI build: fresh
processes, both versions alternating in shuffled order with 0.4.0 run a
second time as a control, identical output required, on a 20-core Apple
silicon Mac, a 64-core x86-64 Linux workstation (pinned to one core) and
a 4-core x86-64 Windows laptop.

- **Delta and XOR decode keep their running values in registers**, and
  distances 2, 3 and 4 (gray and alpha, RGB, RGBA) walk every chain in
  one pass. The old loops reread the value just stored, or walked one
  chain at a time. Distance 1: 4.0 to 5.2x on Windows, 1.05 to 1.08x
  elsewhere; distances 2 to 4: 4.3 to 5.1x on Windows, 6.1 to 6.7x on
  Linux, 2.2 to 2.5x on the Mac.
- **LZW decodes each code as a copy of output already written**,
  instead of walking the code's prefix chain onto a stack: 2.8 to 3.2x
  on all three, and serial reads of an LZW TIFF 1.8 to 2.0x. A code past
  the next free table entry, which no encoder can emit, is now an error;
  it used to decode stack memory left over from earlier codes.
- **TIFF predictors keep their running sums in registers** and load a
  whole pixel before adding it. MSVC compiled the old one-sample loop to
  add to memory and read the sum back. Serial reads with the horizontal
  predictor: uint8 1.70x and uint16 1.20x on Windows; floating point 1.4
  to 2.0x and gray with alpha 1.1 to 1.7x on all three.
- **Byte shuffling moves whole elements** in registers rather than byte
  loops only GCC and Clang vectorized, and only for 2-byte elements: 1.7
  to 2.0x on Windows for 2, 4 and 8-byte elements, and for 4 and 8-byte
  elements 1.6 to 1.9x on Linux and 5.6 to 9.6x on the Mac.
- **Bitshuffle takes its SSE2 path under MSVC**, which never defines
  ``__SSE2__``: 2.3x encode and 1.7x decode on Windows.
- **Bit-packed samples of up to 56 bits unpack through a 64-bit
  accumulator**: 1.7 to 4.6x for 4, 12 and 24-bit samples.
- **BC7 modes without secondary indices skip the general texel loop**
  and, with two or three subsets, interpolate all four channels at once
  in one 64-bit integer: 1.14 to 1.33x, and 1.29 to 1.55x for mode 6.
  The vendored bcdec carries this change, recorded in ``VENDOR.toml``.
- **Uncompressed TIFF strips copy from the file mapping on Windows while
  other reads are in flight**, as on macOS: contiguous positioned reads
  of one file from many threads contend in Windows' file cache. 0.4.0
  read eight concurrent 32 MB reads at 0.71x of 0.3.1; they are now level
  with it. On both, such a copy is now split into parts on several
  threads, as reads are: two concurrent readers on the Mac took 1.45
  against 1.87 ms per read.
- **The Linux wheels build their codec libraries from source**, as the
  macOS and Windows wheels get current ones from Homebrew and
  conda-forge. They took zstd, lz4, brotli, libdeflate, ISA-L, giflib,
  libwebp and openjpeg from AlmaLinux 8, and so shipped zstd 1.4.4, lz4
  1.8.3, libwebp 1.0, openjpeg 2.4 and a libdeflate built by GCC 8 that
  inflated 1.2x slower than the same source built by GCC 14. Serial
  reads of deflate TIFFs on Linux: 1.2 to 1.5x. With the old libraries,
  58 tests that pass on the other platforms failed against the Linux
  wheel (JPEG 2000 options, zstd parameters, lossless WebP); they pass.
- Level or within noise: RGB uint16 on the Mac and Windows, deflate
  strips on the Mac (1.21x on Windows, 1.24x on Linux with the libraries
  above), uncompressed strips, PackBits, BC1 and CRC-32C. Slower: LZW encode 0.95x on Windows, and BC3 decode 0.80x
  on the Mac (0.89x on Windows, where 0.4.0 against itself measured
  0.93x; 1.03x on Linux), because BC3 alpha now rounds as the
  specification defines (below), through bcdec's BC4 kernel.

Compatibility and correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix: the ``floatpred`` codec (``get_codec("floatpred")``) did not
  implement TIFF predictor 3, though it said it did: it put each float's
  least significant byte plane first and restarted the difference at every
  plane, so neither it nor imagecodecs could read the other's output. It
  now writes and reads predictor 3 as TIFF Technical Note 3 defines it,
  byte-identical to imagecodecs' ``floatpred`` and to this package's TIFF
  reader and writer, for any axis and distance. Data encoded with the old
  codec does not decode with the new one. TIFF files were never affected.
- Fix: TIFF strip reads on the fast path reopened the file by name.
  After an atomic save replaced the file, a read returned the new file's
  pixels under the old file's tags, silently. They now read through the
  descriptor the reader already holds; on Windows, which has no
  ``os.preadv``, each part still opens the file, checks that it is the
  same file as that descriptor, and otherwise copies from the reader's
  mapping.
- Fix: byte shuffle with a size whose product overflows (for example
  ``itemsize=4`` with ``2**62 + 1`` elements) passed the length check and
  crashed the interpreter; it now raises ``ValueError``. Delta and XOR
  decode with a distance near ``sys.maxsize`` wrote at a negative index,
  and an empty predictor axis raised on decode; both now work.
- Fix: the ``delta`` codec on float arrays subtracted float values, which
  rounds: it did not even round-trip its own output (1.0 after 1e8 came
  back as 0.0), and neither it nor imagecodecs could read the other's
  stream. Delta and XOR now work on each sample's bit pattern as an
  unsigned integer of the same width, modulo 2**bits, which is TIFF
  predictor 2 as libtiff applies it and what imagecodecs does. Integer
  output is unchanged; float delta output changes (a format change).
  A float stream the old codec wrote carries no header to tell it apart
  and decodes to wrong values with the default decode; pass
  ``legacy_float=True`` to ``delta`` decode to sum it as floats, which
  returns the values 0.4.0's decoder returned, bit for bit (in the
  requested byte order, where 0.4.0 returned native order for a
  big-endian dtype). ``xor`` on floats, which raised ``TypeError``, now
  works the same way.
  Output from both is byte-identical to imagecodecs for every dtype and
  byte order at distance 1, the only distance imagecodecs implements. A
  0-d array, which came back unchanged, now raises ``ValueError`` for
  its missing axis, as imagecodecs does. So does a bool array, as in
  imagecodecs: ``xor`` encoded and decoded one as bytes, and ``delta``
  decode summed one as logical or.
- Fix: ``delta`` and ``xor`` decode of big-endian integers without
  ``out=`` raised "Big-endian buffer not supported"; decode now returns
  the requested dtype, byte order included. Given an ndarray (the form
  imagecodecs returns) and no ``dtype``, ``delta``, ``xor`` and
  ``bitshuffle`` decoded it as flat bytes and returned wrong values
  without an error; they now take the dtype, shape and element size from
  the array, and ``bitshuffle`` and ``byteshuffle`` return an array of
  that dtype and shape for array input. ``delta`` and ``xor`` encode of
  bytes, which raised, now treats them as uint8.
- Fix: ``packints`` accepted imagecodecs' ``runlen=`` and ``bitorder=``
  and ignored them, returning a different stream. ``runlen`` now starts
  each run of samples on a byte boundary, as TIFF stores rows;
  ``bitorder="<"`` packs least significant bit first (GenICam
  ``Mono12p`` and siblings) and ``bitorder=">"`` packs pairs of 10 or 12
  bit samples into three bytes (GigE Vision ``Mono12Packed``), matching
  imagecodecs byte for byte. Calls that passed these keywords now write
  different bytes. A sample count that is not a whole number of runs
  raises in encode and decode rather than dropping the last run, as
  imagecodecs does. A sample too large for ``bitspersample`` (4096 at 12
  bits) was masked to its low bits, as imagecodecs also does, and so
  written as a different value; it now raises ``ValueError``. Whole
  byte widths (16, 32, 64 bits) stay big endian, as the most significant
  bit first stream the codec is defined as; imagecodecs copies them in
  memory order instead, and the docstring now says so. Decode to a float
  dtype, which returned the sample values as floats, or to an integer
  dtype narrower than ``bitspersample``, which kept their low bits, now
  raises ``ValueError`` as imagecodecs does; bool stays accepted for one
  bit samples.
- Fix: the filter codecs (``delta``, ``xor``, ``floatpred``,
  ``packints``, ``bitshuffle``, ``byteshuffle``, ``quantize``) accepted
  any keyword and ignored the ones they did not implement; they now raise
  ``TypeError``. ``byteshuffle`` is the whole-buffer HDF5 and Blosc
  shuffle, as before, and names imagecodecs' per-row ``axis``, ``dist``,
  ``delta`` and ``reorder`` keywords in that error rather than returning
  a stream imagecodecs would not read.
- Fix: ``quantize`` mode ``"nsd"`` multiplied by a scale of
  ``10**(nsd - 1 - floor(log10|x|))``, which is inexact once it is a
  negative power of ten, and for float32 input computed it in float32:
  1234.5678 (float32) at one digit came out 999.99994, 1175103902.8858647
  (float64) came out 999999999.9999999, and 4 to 7 percent of values
  missed the correctly rounded decimal. It now rounds each value to
  ``nsd`` significant digits (ties to even) and returns the value of the
  input type nearest that decimal, by dividing or multiplying by an exact
  power of ten and rounding the few values near a half, or outside the
  range of exact powers, from their decimal digits; a float32 or float16
  whose float64 result lies halfway between two values of its type is
  decided from the decimal rather than rounded a second time.
  Infinities, which became NaN, are now kept, as is a value whose rounding would overflow
  its type. These change the bytes ``"nsd"`` writes (a format change).
  Its docstring claimed both modes matched imagecodecs, and ``"nsd"`` has
  no imagecodecs counterpart. imagecodecs' modes ``"bitgroom"``,
  ``"granularbr"`` (``"gbr"``) and ``"scale"`` are now implemented, bit
  identical to imagecodecs for float32 and float64 at every ``nsd``
  netCDF-C accepts, except for the zeros, NaN and infinities imagecodecs
  alters, and ``mode`` and ``nsd`` may be passed by position as in
  ``imagecodecs.quantize_encode``. As netCDF-C does in all three of its
  modes, BitRound, BitGroom and Granular BitRound leave netCDF's fill
  value (9.9692099683868690e+36), ``+0.0``, ``-0.0`` and NaN unchanged;
  BitRound used to round the fill value and NaN, so a NaN whose payload
  was only in its low bits became an infinity, and these change the bytes
  written for data holding such values (a format change). Infinities are
  kept too: netCDF-C's BitGroom turns an odd-indexed infinity into NaN,
  and its Granular BitRound has undefined behavior for one. ``nsd`` is
  checked as netCDF-C's ``nc_def_var_quantize`` checks it: at least 1, at
  most the mantissa bits for BitRound, and at most 6 (float32), 15
  (float64) or 2 (float16) digits for BitGroom and Granular BitRound, so
  ``bitspersample=0``, which BitRound accepted, now raises; a fractional
  ``nsd``, which was truncated, raises too. ``quantize`` decode is still
  the identity, but an unknown ``mode``, or an ``nsd`` encode would
  reject, passed to it now raises instead of being ignored.

- Fix: the TIFF writer accepted ``predictor=2`` for float16, float32 and
  float64 and subtracted neighboring samples as floats. TIFF 6.0 defines
  Predictor 2 as horizontal differencing, and libtiff applies it to each
  sample's storage word as an unsigned integer of the same width, so
  libtiff and tifffile decoded those files to wrong values, and this
  package's reader refused them. The writer now differences the bit
  patterns as libtiff does, which is lossless. This changes the bytes
  written for float data with predictor 2; integer data is unchanged.
  Files written by earlier versions with float data and predictor 2 hold
  rounded float differences that no reader decodes to the original
  values, and have no tag that marks them, so they should be written
  again from their source. This package's reader used to refuse such
  files with ``NotImplementedError``. It now reads the LZW, Deflate and
  Zstandard ones as libtiff and tifffile do, to the same wrong values.
  For the uncompressed ones libtiff ignores the predictor and returns
  the stored differences, while this reader undoes the predictor on the
  storage word, as tifffile does for tiled files (it refuses the striped
  ones); neither result is the original data.
- Fix: the TIFF reader could not undo predictor 2 on 64-bit integer or
  floating-point samples and raised ``NotImplementedError``. libtiff and
  GDAL write predictor 2 for 8, 16, 32 and 64-bit samples of every
  sample format, tifffile for 64-bit integers, and this package's writer
  for all of them. The reader now undoes predictor 2 on the unsigned
  storage word for 8, 16, 32 and 64-bit samples of any sample format, on
  the fused native path as well as the general one.
- Fix: the TIFF writer accepted ``predictor=2`` with ``compression="none"``
  and wrote differenced samples under a Predictor tag. TIFF 6.0 defines
  Predictor alongside LZW, and libtiff ignores it on uncompressed data,
  so libtiff returned the raw differences while tifffile undid them.
  With ``verify=True`` the same call wrote the samples undifferenced
  under the Predictor tag, and every reader returned wrong values. For
  image codecs (JPEG, JPEG 2000, WebP, JPEG XL, LERC) the writer dropped
  ``predictor=2`` without saying so. Both now raise ``TiffWriterError``:
  a predictor is written only with LZW, Deflate or Zstandard. tifffile
  raises ``ValueError`` for the same calls except LERC, where it applies
  the predictor before encoding; this writer does not implement that.
  Uncompressed integer predictor 2 files written by earlier versions
  read back exactly.
- Fix: the TIFF writer took ``predictor=True`` as the integer 1 and wrote
  no predictor without saying so. It now picks 3 for float and 2 for
  integer samples, as tifffile does (tifffile refuses ``True`` for
  64-bit integers, where this writer uses 2), and ``predictor=False``
  means 1. This changes the bytes written for ``predictor=True``; with a
  compression that takes no predictor it now raises, as above.
  ``predictor=0`` and ``predictor=None`` raised and now mean 1, as in
  tifffile. ``planar_config`` now refuses a bool rather than taking
  ``True`` as 1.
- Fix: the TIFF writer wrote ``planar_config=2`` into the
  PlanarConfiguration tag but stored the samples interleaved, in one run
  of strips or tiles. libtiff refuses those files, tifffile raises or
  reads them to wrong values, and this package's reader raised or, for
  single-strip and single-tile LZW pages, returned wrong values. The
  writer now stores each sample's plane as its own strips or tiles, one
  plane after another, as TIFF 6.0 defines; this changes the bytes
  written for multi-sample images with ``planar_config=2``. Values other
  than 1 and 2, and ``planar_config=2`` with WebP, which cannot store a
  one-sample plane, now raise ``TiffWriterError``. The reader now reads a
  PlanarConfiguration=2 page that holds a single run of strips or tiles,
  which can only be that earlier layout, as interleaved samples, so
  those files read back as they were written (for float predictor 2
  files, see above). A full-page read of a separate-plane page with
  fewer strips or tiles than its planes need read the later planes from
  the wrong segments; it now raises ``ValueError``.
- Fix: the TIFF reader took the samples of a LERC segment in a big-endian
  file as native values and returned wrong values for 16, 32 and 64-bit
  samples. The writer stores each sample in the file's byte order before
  LERC encodes it, and tifffile swaps the decoded samples back; the
  reader now does too, so big-endian LERC files with integer samples
  that earlier versions wrote read back exactly. Byte-swapped floats can
  be NaN patterns, which LERC does not store exactly, so in a file whose
  byte order differs from the machine's (big-endian, on the usual
  little-endian machines) the writer could store other values than it
  was given without saying so. It now raises ``TiffWriterError`` when a
  float segment it would hand to LERC for such a file holds such a
  pattern.
- Fix: the TIFF reader returned complex samples (SampleFormat 6) as
  unsigned integers holding their bits. It now returns ``complex64`` and
  ``complex128``, swapping each component in a big-endian file as
  libtiff and tifffile do, and undoes predictor 2 on ``complex64`` as
  libtiff does, on the 64-bit sample word. Predictor 2 on ``complex128``
  raises ``NotImplementedError``; libtiff has no predictor 2 for 128-bit
  samples either.

- Fix: zstd decode sized its output from the first frame alone, so two
  concatenated frames, or a skippable frame followed by a frame, failed
  with "Destination buffer is too small". RFC 8878 defines zstd data as
  one or more frames, and the zstd CLI writes and reads concatenations;
  every frame now decodes and skippable frames are skipped. The LZ4
  codec decoded only the first of several concatenated frames and
  silently dropped the rest; it now decodes them all, as the LZ4 frame
  specification and the lz4 CLI do, and bytes after the last frame that
  are not a frame raise instead of being ignored.
- Fix: the N5 reader sent ``"lz4"`` blocks to the LZ4 frame decoder, but
  the N5 reference implementation writes lz4-java's ``LZ4Block`` stream,
  so no N5 lz4 dataset could be read. That format is now decoded, with
  its checksums verified; LZ4 frames are still accepted.
- Fix: the deflate codec ignored ``raw=``. ``encode(raw=True)`` returned
  a zlib stream and ``decode(raw=True)`` rejected bare DEFLATE. ``raw``
  now selects a bare DEFLATE stream (RFC 1951) on both sides, as in
  imagecodecs; the default stays zlib (RFC 1950).
- Fix: gzip output depended on the platform and Python version, because
  the stdlib let zlib write its own OS byte (19 on macOS or 3 on Linux
  under Python 3.11 and 3.12, 255 under 3.13). gzip now encodes through
  the same libdeflate engine as the deflate codec with a fixed header
  (MTIME 0, OS 255), and accepts levels up to 12 as imagecodecs does.
  Builds linked to libdeflate, as imagecodecs is, write bytes identical
  to imagecodecs; a build that falls back to zlib, because libdeflate
  was not found, writes the same header around zlib's DEFLATE data, and
  so does a build without the deflate extension at all, through the
  stdlib ``zlib`` module. The OME-Zarr writer's gzip chunks, which also
  carried the time they were written, use the same encoder. This changes
  the bytes written; decoding, including multi-member files, is
  unchanged.
- Fix: LZW and PackBits decode without ``expected_size`` capped the
  output at 8 and 2 times the input and failed on anything that
  compressed better, which broke ``decode_segment`` and
  ``verify_segment`` (used by the NDTiff reader) for those codecs. Both
  formats end themselves (TIFF 6.0 sections 9 and 13), so the output is
  now sized from the stream. An LZW decode that exactly filled the
  guessed buffer could also return a truncated result; it now grows and
  decodes again.
- Fix: ``.sz`` files were routed to the raw Snappy block codec, but
  ``.sz`` is the extension of Snappy's framing format, so real ``.sz``
  files failed to read. A new ``snappy_framed`` codec reads and writes
  the framing format, checking each chunk's CRC-32C, and owns ``.sz``;
  ``snappy`` stays the raw block, byte-identical to imagecodecs. This is
  a format change: ``write("x.sz", ...)`` now writes the framing format,
  where 0.4.0 wrote a raw Snappy block. ``.sz`` files 0.4.0 wrote still
  read, because ``snappy_framed`` decodes data without the stream
  identifier as a raw block (and raises if it is not a valid one).
- Fix: blosc2 encode took ``shuffle`` only as a bool and silently
  ignored ``splitmode``, ``blocksize`` and ``numthreads``. It now takes
  c-blosc2's filter codes 0 to 4 and their names (``"bitshuffle"`` can
  be written at last), split modes by code or name, a block size and a
  thread count, as imagecodecs does. The split default now matches
  imagecodecs, always split, where blosc2's own default skipped the
  split for some compressors and levels (zstd at level 9, for one), so
  equal settings write equal chunks. This is a format change at those
  settings; every blosc2 decoder reads both forms. The default
  ``typesize`` now also follows imagecodecs, in the codec and in the
  native encoder, ``opencodecs.codecs._blosc2.encode``: 8 for a flat run
  of unsigned bytes (``bytes``, ``bytearray``, a contiguous 1-D uint8
  array) and the buffer's own item size otherwise. A ``uint16`` array
  passed straight to the native encoder is shuffled as 2-byte items,
  where it used to be written with a type size of 8, and a 1-D uint8
  array given to the codec is written with a type size of 8, where it
  used to get 1. That is a format change too: the bytes written change,
  and the decoded data is the same.
- Fix: brotli encode ignored ``mode`` and ``lgwin``; both are now
  honored. Its default level is now 4, imagecodecs' default, instead of
  3, which was chosen on the mistaken belief that imagecodecs used level
  1 and was not smaller on every input. Default output changes and is
  byte-identical to imagecodecs.
- Fix: the deflate, gzip, zstd, LZ4, brotli, blosc2 and Snappy codecs
  ignored ``out=`` on encode, and gzip ignored it on decode: the call
  returned new bytes and left the caller's buffer untouched. ``out=``
  now follows imagecodecs, here and in the new ``snappy_framed`` codec:
  an int is a capacity, a writable buffer receives the result, and
  ``out=bytearray`` returns a bytearray; a result that does not fit
  raises. A buffer the result fills exactly is returned as is; a larger
  numpy array returns a uint8 array view of the written prefix, as
  imagecodecs does, where these codecs' decoders used to return a
  memoryview, and any other buffer returns a memoryview. The decoders
  other than gzip already wrote into a given buffer, but raised
  ``TypeError`` for ``out=bytearray`` and returned a memoryview of a
  buffer they filled exactly; both now follow imagecodecs. These codecs
  also accepted any keyword and dropped the ones they did not know, so a
  misspelled or unsupported option was silently ignored; an unknown
  keyword now raises ``TypeError``. LZ4 encode takes imagecodecs'
  ``blocksizeid``, ``contentchecksum`` and ``blockchecksum``; a block
  size code the LZ4 frame format reserves (0 to 3 in the header) raises
  instead of being written. Default output is unchanged. LZ4 encode now
  clamps ``level`` to -1 through 12, as imagecodecs does, where a level
  below -1 went to liblz4 as is and selected a faster, larger mode. This
  is a format change at those levels: the bytes written change, and the
  decoded data is the same.
- zstd keeps passing negative levels (libzstd's fast modes) through to
  libzstd; imagecodecs clamps them to its default, so ``level=-5``
  differs between the two. DICOM RLE decode returns interleaved
  ``(H, W, C)`` samples, Planar Configuration 0, where imagecodecs
  returns planar bytes. Both are now documented.

- Fix: PNG decode kept the color type only for some images. 1, 2 and
  4-bit grayscale came back as RGBA, an indexed image as RGBA whose alpha
  was always 255 because the tRNS chunk was never applied, and tRNS on
  gray or RGB images was ignored. Decode now follows the PNG specification
  and libpng, matching imagecodecs: gray stays ``(H, W)`` with sub-byte
  samples scaled to 8 bits, a palette gives RGB, and tRNS adds the alpha
  channel it defines (gray becomes ``(H, W, 2)``, RGB and indexed
  ``(H, W, 4)`` with the palette alpha). The row reader follows the same
  rule. Arrays for these images change shape; nothing written changes.
- Fix: ``PngCodec.encode`` dropped ``filter_choice`` and every other
  option except ``level`` and the ICC profile, and accepted unknown
  options silently. It now forwards ``filter_choice``, ``strategy`` and
  imagecodecs' ``filter=`` (its ``PNG.FILTER`` values), and raises
  ``TypeError`` on an unknown option. ``strategy`` did nothing even in
  the native encoder, because builds with libdeflate compress the image
  data through it and libdeflate has no strategy; an explicit strategy
  now compresses through zlib with that strategy, as in imagecodecs. A
  big-endian ``uint16`` array was refused; it is now stored by value, as
  PNG's most significant byte first rule requires (imagecodecs writes
  such an array byte-swapped). ``filter`` also takes imagecodecs'
  ``PNG.FILTER`` names, including ``"no"``, and ``strategy`` its
  ``PNG.STRATEGY`` names; a strategy outside 0 to 4 raises, as in
  imagecodecs, instead of reaching zlib. An invalid filter or strategy
  raises ``PngOptionError``, which is both the ``PngError`` the encoder
  raised before and the ``ValueError`` the row encoder raised.
  ``PngCodec.decode`` and ``QoiCodec.decode`` dropped unknown options;
  they now raise ``TypeError``, keeping ``out`` (the only option
  imagecodecs defines) and ``numthreads``, which the PNG and QOI
  encoders (and ``PngCodec.encode_rows``) also accept and which does
  nothing in either. The PNG, WebP and QOI codec encoders take
  imagecodecs' ``out=None``; any other ``out`` raises ``TypeError``
  instead of being dropped, since the encoded bytes are returned or
  written to ``dest``. Default PNG output is unchanged. This is a
  format change for PNG written through the codec with
  ``filter_choice`` or ``strategy``, since those settings now take
  effect.
- Fix: lossless WebP was not exact for RGBA. libwebp's default
  ``exact=0`` rewrote the RGB values under fully transparent pixels, so
  they did not round-trip; lossless encoding now sets ``exact=1``, as
  imagecodecs does. In lossless mode ``level`` was ignored; it is now
  libwebp's compression effort (0 fastest, 100 smallest), its meaning in
  libwebp and imagecodecs. Lossless bytes also depended on ``numthreads``
  and ``method``, because some argument combinations took libwebp's
  simple API (effort 70) and others the advanced one (effort 75); every
  encode now takes the advanced API. ``_webp.encode`` defaulted to lossy
  while ``WebpCodec`` and ``imagecodecs.webp_encode`` default to
  lossless; it now defaults to lossless, and with it TIFF
  ``compression="webp"``, which used to write lossy tiles unless told
  otherwise, the same default as tifffile. A negative ``level`` means
  lossless, ``level`` is clamped to 100 and keeps its fraction, and
  ``method`` is clamped to 0 to 6, with ``None`` meaning 4, as in
  imagecodecs; ``method=-1`` therefore now means 0, where it meant
  libwebp's default 4 before. ``lossless`` is read as imagecodecs reads
  it, ``int(lossless)``, so ``0``, ``1`` and NumPy bools work and a
  string such as ``"no"`` raises ``ValueError`` instead of meaning
  lossless. This is a format change for
  WebP: the bytes change for RGBA images with transparent pixels, every
  lossless encode at a level other than 75, lossless encodes that took
  the simple API, encodes with
  ``method=-1`` or a fractional ``level``, and WebP TIFF tiles written
  without an explicit setting. For an RGB or RGBA array, with the same
  libwebp release, the bytes now equal ``imagecodecs.webp_encode``'s for
  the same arguments.
- Fix: ``WebpCodec.decode`` dropped imagecodecs' ``hasalpha`` and
  ``index`` and any other option. ``hasalpha`` now forces RGBA or RGB as
  in imagecodecs, ``index`` selects one frame of an animation (negative
  values count from the end, out of range raises ``IndexError``), and an
  unknown option raises ``TypeError``. An animation always decoded to
  RGBA; with ``hasalpha=None`` it now keeps alpha only when a returned
  canvas has a pixel that is not fully opaque (any frame of the stack, or
  the frame ``index`` picks) and is RGB otherwise, as in imagecodecs, so
  the array of an opaque animation changes shape. ``open()`` still
  returns the RGBA canvas libwebp composes for every frame of an
  animation. New API: the native ``opencodecs.codecs._webp`` module's
  ``decode`` takes ``hasalpha``, it gains ``version()``, the linked
  libwebp in ``imagecodecs.webp_version()``'s format, and
  ``opencodecs._webp_codec.decode_webp`` is the decoder ``WebpCodec``
  uses.
- Fix: ``QoiCodec.encode`` dropped unknown options; it now raises
  ``TypeError``. QOI output is unchanged, and how its RGBA output differs
  from imagecodecs is now documented: header byte 13 only. The QOI
  specification defines that byte as informative; opencodecs writes 0,
  "sRGB with linear alpha", for RGB and RGBA alike, as the specification
  and its reference encoder do, and imagecodecs writes 1 for RGBA.
  ``srgb=False`` gives byte parity.

- Fix: JPEG-LS could not decode a multi-component image coded with
  interleave mode NONE (one scan per component, ITU-T T.87 Annex C.2.3),
  which DICOM archives and other encoders write; it failed with "output
  buffer too small". Such images now decode to (H, W, C) like the other
  modes, through ``oc.read`` and DICOMweb alike.
- Fix: JPEG-LS ignored ``level=``, imagecodecs' name for the NEAR bound,
  and wrote a lossless file. ``level`` now sets NEAR, the same stream
  imagecodecs writes apart from its SPIFF header, and disagreeing
  ``level`` and ``near_lossless`` raise. Output is still a bare
  codestream with sample interleave, both conforming choices.
- Fix: AVIF and HEIF coded uint16 data at 10 bits with no range check,
  so a plain ``encode`` clamped every value above 1023 and an explicit
  ``bit_depth`` clamped too. uint16 data is now coded at 10 or 12 bits,
  whichever holds its largest value, and data that does not fit the
  depth (AV1 and the HEVC encoders stop at 12 bits) raises instead.
  Files for data above 1023 change from a clamped 10-bit image to an
  exact 12-bit one.
- Fix: AVIF and HEIF stored gray input as three equal color planes, and
  AVIF decoded every file, real monochrome ones included, as RGB. Gray
  and gray with alpha are now coded monochrome (AV1 ``mono_chrome``,
  HEVC chroma format 0). A monochrome AVIF decodes to (H, W) or
  (H, W, 2), as imagecodecs.avif_decode returns it. A monochrome HEIF
  still decodes to RGB(A) by default, as imagecodecs.heif_decode returns
  it, and to (H, W) or (H, W, 2) with imagecodecs' keyword
  ``photometric='monochrome'``, which now works on decode (it raises for
  a color image, where imagecodecs returns the red channel). This
  changes the bytes written for gray input, and the shape read back from
  monochrome AVIF files. Gray AVIF is tagged with an
  unspecified matrix (2), as imagecodecs tags it, since a single plane
  has no chroma for a matrix to act on. Lossless gray AVIF then matches
  imagecodecs byte for byte when both link the same libavif and libaom,
  for images under 1024 px on the long axis; larger images are tiled
  4x4 by default here and untiled in imagecodecs, and
  ``tilelog2=(0, 0)`` writes them untiled.
- Fix: AVIF and HEIF ignored ``level=`` unless ``lossless=False`` was
  also passed, so imagecodecs-style ``encode(a, level=30)`` wrote a
  lossless file. ``lossless`` now defaults to None: with no level, or a
  level of 100 (AVIF) or above 100 (HEIF), the file is lossless, as in
  imagecodecs, and a lower level is lossy; an AVIF level of -1 or lower
  is lossy at libavif's own default quality, as imagecodecs maps it.
  Gray (1 or 2 sample) AVIF is the one difference: imagecodecs writes it
  lossless whatever the level, and opencodecs honors the level for it
  as for color.
  ``lossless=True`` with a lossy level raises. A bare
  ``oc.get_codec("avif")`` or ``oc.get_codec("heif")`` encode is
  unchanged (lossless). The extension functions
  ``opencodecs.codecs._avif.encode`` and ``_heif.encode`` used to be
  lossy by default (AVIF quality 60 at 4:2:0, HEIF quality 50); a bare
  call to them is now lossless too. This is a format change: a level
  alone now writes a lossy file where a lossless one was written, an
  AVIF level of -1 or lower writes libavif's default quality where it
  wrote quality 0, and a bare call to the extension functions writes
  lossless, larger files.
- Fix: lossy AVIF color was tagged with an unspecified matrix (2), which
  leaves readers outside libavif to guess. It is now tagged BT.601 (6),
  the matrix libavif converts with; pixels are unchanged, only that tag
  in the ``colr`` box differs. Lossy AVIF can still decode a level or
  two apart from imagecodecs on the same bytes: the imagecodecs build
  converts with libyuv, this one with libavif's own converter, which
  matches the rounded ITU-T H.273 result.
- Fix: lossy AVIF and HEIF color was subsampled 4:2:0 by default, where
  imagecodecs codes 4:4:4, which smears sharp color edges. Lossy color is
  now 4:4:4 by default for both; AVIF's ``yuv_format`` (or
  ``pixelformat``) still asks for subsampling. AVIF alpha is now coded
  lossless at every level, as imagecodecs codes it, rather than at the
  color quality. AVIF ``speed`` defaulted to 0, the slowest, through
  ``oc.get_codec("avif")`` and to 6 in the extension; both now leave
  libavif's default, as imagecodecs does, and a speed outside 0 to 10 is
  clamped to that range, as imagecodecs clamps it, instead of being
  ignored. This changes the bytes of lossy AVIF and HEIF files.
- Fix: the AVIF, HEIF, JPEG-LS and LERC codecs dropped keywords they did
  not know, on encode and decode. They now raise TypeError for them, and
  accept imagecodecs' names: ``bitspersample``, ``pixelformat``,
  ``tilelog2``, ``primaries``, ``transfer`` and ``matrix`` for AVIF
  encode, ``index`` for AVIF decode (one image of a sequence, IndexError
  past the end), ``bitspersample``, ``photometric`` and ``compression``
  (HEVC only) for HEIF encode, by name or as libheif's integer enum
  values as imagecodecs takes them, and ``photometric`` for HEIF
  decode. A HEIF ``photometric`` that disagrees with the array raises.
  The
  AVIF wrapper also passes ``codec``, tiling, ``yuv_format`` and
  ``codec_options`` through, which it used to drop.
- Fix: a HEIF with an 8-bit image and a deeper alpha plane (HEIF codes
  alpha as a separate image with its own bit depth) decoded to wrong
  alpha values with no error, through libheif's RGBA conversion. A HEIF
  whose alpha depth differs from the image's now raises HeifError, as a
  deeper image with a shallower alpha already did, since one array
  cannot hold both planes at their own depths. This needs libheif to
  list the alpha as an auxiliary image, which the libheif 1.21 the
  wheels bundle does. libheif 1.23 does not, and its own conversion
  rescales the alpha to the image's depth (up by bit replication, down
  by dropping low bits), so a build against it returns that instead of
  raising.
- Fix: LERC wrote Lerc2 codec version 6, which readers built on libLerc
  before 4.0 cannot open and which libtiff warns about in TIFF, where it
  fixes version 4. It now writes version 4 by default, byte-identical to
  imagecodecs, with ``version=`` (2 to 6) to choose; every version still
  decodes. This changes the bytes of every LERC blob and LERC TIFF tile
  written. LERC also accepts imagecodecs' ``level``, ``masks``,
  ``planar``, ``compression`` (zstd or deflate around the blob, unwrapped
  again on decode) and 1-D input, returns masks on request, and decodes
  pixels a mask marks invalid as 0 rather than leaving whatever the
  output buffer held. A deflate ``compressionargs`` level outside -1 to
  9 is clamped to that range, as imagecodecs clamps it, rather than
  failing inside zlib.
- Fix: the EER codec returned uint8 for a frame while ``oc.open`` on the
  same EER file returned bool. A frame holds at most one event per pixel
  and Falcon files declare BitsPerSample=1, so the codec now returns bool,
  as imagecodecs and tifffile do; ``out=`` of uint8 or uint16 still
  accumulates counts, and a bool ``out=`` now has events OR-ed in.

- Fix: the ``rcomp``, ``aec``, ``pcodec``, ``sz3`` and ``sperr`` codecs
  put a private header of ours in front of each library's stream, so
  nothing else could read what they wrote and they could not read the
  standard stream from anywhere else. Each now writes the stream its
  specification or library defines, byte-identical to imagecodecs for
  the same parameters and an array in native byte order, and takes what
  that stream does not record as decode arguments, as imagecodecs does.
  A big-endian array is coded by its values (aec by default, see
  below); imagecodecs codes its bytes as if they were native numbers
  (sz3 refuses it), so for such an array rcomp, pcodec and sperr write
  different bytes, and an imagecodecs stream of one holds byte-swapped
  numbers, which opencodecs decodes as the numbers they are, also into
  a big-endian ``dtype``. This changes the bytes all five write. Blobs
  written by earlier releases still decode: pcodec, sz3 and sperr
  recognize their old header by its magic. rcomp and aec, whose old
  headers had none, read a blob as an old one when every header field
  holds a value the old encoder could write and agrees with what the
  caller passes (coding parameters included, since 0.4.0's aec decode
  accepted and ignored them), and the rest decodes. Where the same
  bytes also decode as a standard stream, the reading whose values
  encode back to exactly those bytes is kept, the standard one first,
  and aec raises when neither does. Parameters take imagecodecs' names too (``nblock``,
  ``bitspersample``, ``blocksize``, ``flags``, ``pagesize``, ``abs``,
  ``rel``, ``level``, ``chunks``, ``numthreads``), and an option none of
  them defines now raises ``TypeError`` instead of being ignored.
- Fix: ``rcomp`` writes the bare cfitsio Rice stream, which is what FITS
  stores in a ``RICE_1`` tile (FITS 4.0, section 10.4.1). Decoding it
  takes ``shape`` and ``dtype`` and the block size (``blocksize`` or
  ``nblock``, 32 by default), and returns the requested signed or
  unsigned type, not flat unsigned words.
- Fix: the Rice decoders behind ``rcomp`` and FITS ``RICE_1`` tiles could
  read past the end of a damaged or truncated stream. cfitsio checks for
  the end of the input once per coding block, and inside a block a run
  of zero bits has no length limit, so the decoder kept reading until it
  met a nonzero byte or faulted. Every byte is now checked before it is
  read, and such a stream raises. Valid streams decode as before; the
  check costs 3 to 8 percent of Rice decode time in our measurements.
- Fix: ``aec`` writes the bare CCSDS 121.0-B-2 stream that libaec
  produces, the stream GRIB2 and imagecodecs use. Its defaults are now
  imagecodecs': block size 8 and reference sample interval 2 (they were
  32 and 128), so that streams decode with the same defaults on both
  sides; NetCDF and HDF5 szip data commonly use 32 and 128, which also
  compress better, so pass them to both calls when you want them.
  Decoding takes the coding parameters, and ``dtype`` and ``shape`` or
  ``out`` for the sample type and count. An output too small for the
  stream now raises instead of truncating, and a sample value that does
  not fit ``bits_per_sample`` raises instead of losing its high bits,
  for bytes input as for arrays. libaec takes a signed sample narrower
  than its item as a ``bits_per_sample``-bit two's complement number.
  Negative values in a signed array with ``bits_per_sample`` below the
  item width were passed to it sign-extended, and decoded to other
  values without an error (imagecodecs does the same); they are now
  masked to ``bits_per_sample`` bits first, so they decode back to the
  values given, and the bytes written for such data change. A value
  above the signed range of ``bits_per_sample`` bits raises rather
  than being read back as a negative number. A 0-d array was coded as
  a run of zero bytes, as many as its value; it is now coded as one
  sample, as imagecodecs codes it.
  When neither ``flags`` nor ``msb`` is given, the byte order bit
  (``AEC_DATA_MSB``) follows the array, so a big-endian array is coded
  by value where imagecodecs codes its bytes as little-endian samples,
  and decoding with ``dtype`` returns values. An explicit ``flags`` or
  ``msb`` is kept as given, as in imagecodecs: the array's bytes are
  coded in the order it gives, decoding returns the decoded bytes as
  they are with or without ``dtype``, and a sample that does not fit
  ``bits_per_sample`` read that way raises. int16 and int32 arrays
  set the signed flag and int8 arrays do not, as in imagecodecs, so int8
  streams are the same bytes and imagecodecs' int8 streams decode right
  with ``dtype='i1'``. A big-endian int16 or int32 array sets it too,
  where imagecodecs leaves it off (and cannot read the result back
  into that array); pass ``signed=False`` and the stream's ``flags`` to
  read such a stream. ``restricted`` and ``pad_rsi`` are new. libaec's
  encoder writes the ``AEC_PAD_RSI`` padding only when built with
  ``ENABLE_RSI_PADDING``; a libaec built without it accepts the flag
  and writes no padding, a stream no decoder reads back with the flag.
  Encoding with ``pad_rsi`` checks which the linked libaec does and
  raises ``ValueError`` when it does not pad; decoding a padded stream
  works with either build.
- Fix: ``pcodec`` writes pcodec's standalone format (magic ``pco!``),
  the format of the pcodec package, numcodecs and imagecodecs. It
  records the number type and a hint of the element count, which the
  format allows to be 0 for unknown, but no shape. ``decode`` takes the
  count from ``shape`` or ``out`` when given, as imagecodecs does, and
  otherwise from the hint, returning a flat array; a stream that holds
  more than its hint, such as any nonempty stream whose hint is 0,
  needs ``shape`` or ``out``.
- Fix: ``sz3`` writes SZ3's own stream, and its defaults are now
  imagecodecs': mode ``'abs'`` with an error bound of 0, where
  opencodecs used 1e-3. A call with no bound therefore writes the same
  bytes as ``imagecodecs.sz3_encode`` (more bytes than before, with no
  error allowed); pass ``abs_err`` (or ``abs``) to compress lossily.
  That stream does not reliably
  record its data type, so ``decode`` needs ``dtype`` (or ``out``); the
  shape defaults to the stored dimensions, which leave out those of size
  1. SZ3 reads a payload without bounds checks and throws exceptions its
  C API does not catch, so a stream it cannot read ends the process.
  ``decode`` checks the header, the configuration and the layout of the
  payload first, and raises for a truncated stream, a stream of another
  SZ3 data version, or a payload not laid out the way SZ3 writes its
  configuration. The layout also shows the value type wherever the
  stream holds values SZ3 could not predict, so a ``dtype`` that does
  not match raises ``ValueError`` instead of ending the process. A
  stream with no such values is laid out the same for float32 and
  float64, and a wrong ``dtype`` then decodes without an error. Damage
  inside the coded data that leaves the layout intact can still crash
  SZ3, as it does in imagecodecs. Asking for the ``psnr`` or ``norm``
  mode, for integer data or for more than four dimensions longer than 1
  also ended the process inside the SZ3 C API, which implements none of
  them; each now raises ``ValueError``.
- Fix: ``sperr`` writes SPERR's own format: a 2-D slice with SPERR's
  10-byte header (``header=False`` leaves it off) and a 3-D volume as
  SPERR returns it. ``decode`` takes the shape and precision from that
  header. SPERR's header has no magic, so ``oc.read`` recognizes a
  stream without ``format=`` by its fields: the version byte, flags
  SPERR sets, dimensions holding 1 to 2**40 values, the chunk lengths
  in the first 512 bytes, and as much of the first coded stream's fixed
  fields as those bytes hold (for a stream under 512 bytes, every check
  ``decode`` makes on the stream). In a volume of 117 chunks or more the chunk
  table can push some or all of those fields past the 512th byte; the
  chunk lengths then stand in for them. A 2-D stream written without its
  header, or a stream of more than 2**40 values, needs
  ``format='sperr'``.
  In ``psnr`` mode SPERR's quantization step overflows to infinity for
  float64 data of very large magnitude (values of about 1e154 and up in
  our tests), and the stream it writes decodes to NaN; ``encode`` now
  raises ``ValueError`` for such data instead of writing it, and
  ``decode`` still reads such a stream, from imagecodecs or 0.4.0, as
  NaN. SPERR's decoder checks none of its input, and a stream cut
  short inside its fixed fields or with a damaged flag, bit-plane or
  bit-count field ended the process (segmentation fault or abort), in
  0.4.0 as in imagecodecs. ``decode`` now checks the header, the chunk
  table and those fields of each coded stream first and raises
  ``SperrError`` instead; in a fuzz of 300 single-byte changes, the
  only failures left were the allocations described next. Damage the
  checks cannot tell from a valid stream is not covered: changed coded
  bits, or a header field changed to another plausible value, decode
  to wrong values without an error and may still crash SPERR, and a
  dimension made larger makes SPERR allocate memory for that size,
  which for a high bit means hundreds of gigabytes. A 3-D volume is now
  compressed as one chunk by default, as imagecodecs does, where it was
  split into chunks of 256 along each axis; ``chunks=(256, 256, 256)``
  gives that chunking (SPERR's sperr3d default), which lets the chunks
  of a large volume compress on separate threads.

- Fix: the ``jpeg`` codec's ``encode`` (``get_codec("jpeg")`` and
  ``write(..., format="jpeg")``) dropped every option except ``level`` and
  ``iccprofile``, without an error: ``subsampling="444"`` wrote 4:2:0 and
  ``lossless=True`` wrote a lossy baseline JPEG. It now takes the
  parameters of imagecodecs' ``jpeg8_encode`` (``colorspace``,
  ``outcolorspace``, ``subsampling``, also as a ``(2, 2)`` tuple,
  ``optimize``, ``smoothing``, ``lossless``, ``predictor``,
  ``bitspersample``, and ``validate``, which has no effect in either
  library), honoring each as libjpeg defines it. Where imagecodecs applies
  an option, the bytes are the ones imagecodecs writes, with three
  differences. imagecodecs ignores ``subsampling`` for an RGB, CMYK or
  YCCK JPEG (RGB and CMYK are never subsampled, YCCK always 4:2:0), and
  ``jpeg`` subsamples them as asked, which T.81 allows since every
  component has its own sampling factors; left unset, the sampling factors
  are imagecodecs'. And the packed input orders other than RGB (``"bgr"``,
  ``"rgbx"``, ``"bgrx"``, ``"xrgb"``, ``"xbgr"``) are libjpeg-turbo's
  JCS_EXT_* layouts in ``jpeg``: it reads the color samples in the order
  named and stores a three-component (or grayscale) JPEG, without the
  padding sample of a four-sample order. imagecodecs raises for the
  J_COLOR_SPACE integers of these orders, and reads their names as an
  unknown colorspace, storing every sample unconverted as its own
  component: a four-sample order keeps its fourth sample in a
  four-component frame, and a ``"bgr"`` array becomes three components
  that decode to other pixels. The orders with alpha (``"rgba"``,
  ``"bgra"``, ``"argb"``, ``"abgr"``) raise in ``jpeg`` and ``mozjpeg``,
  since a JPEG has no alpha channel and the alpha samples would be lost;
  imagecodecs raises for ``"rgba"`` and the integers, and keeps the fourth
  sample as a component for the other names. And YCbCr input
  (``colorspace="ycbcr"``) is stored unconverted, as in imagecodecs, but
  TurboJPEG converts only from RGB, so ``jpeg`` passes the samples through
  RGB storage and then labels the components as imagecodecs and libjpeg
  label YCbCr, with component ids 1, 2, 3 and a JFIF marker (T.871). The
  bytes still differ from imagecodecs': Cb and Cr share the luminance
  quantization and Huffman tables. IJG libjpeg 9 infers the colorspace
  from the component ids before any marker, and libjpeg-turbo reads the
  markers first; both read the two streams as YCbCr (IJG libjpeg decodes
  no lossless JPEG), and at quality 100 they decode to the same pixels. A
  value TurboJPEG cannot honor, such as ``smoothing``, YCCK input, or,
  with ``lossless=True``, a subsampling, a YCbCr, YCCK or grayscale JPEG
  of color input, or ``optimize=False`` (T.81 allows the first two;
  TurboJPEG's lossless mode writes neither and always computes its
  Huffman tables, and imagecodecs ignores these there), raises, and an option the codec does not know is a
  ``TypeError``. The native ``codecs._jpeg.encode`` and
  ``codecs._mozjpeg.encode`` no longer cast a list or other non-array
  input to uint8: as in imagecodecs, the input keeps the dtype NumPy gives
  it, so a list of Python ints raises; pass a uint8 (or, for ``jpeg``,
  uint16) array. The ``jpeg`` and ``mozjpeg`` decoders take imagecodecs'
  ``tables``, ``header``, ``colorspace``, ``outcolorspace``,
  ``fancyupsampling``, ``shape`` and ``bitspersample``; as in libjpeg,
  ``colorspace`` without ``outcolorspace`` returns the stored components
  unconverted, and ``fancyupsampling=False`` takes effect (imagecodecs
  2026.8.16 sets it before reading the header, which resets it). The
  packed output orders are libjpeg-turbo's layouts too, so
  ``outcolorspace="bgr"`` returns blue, green, red, as imagecodecs does
  for the J_COLOR_SPACE integer (8); imagecodecs reads the names other
  than ``"rgba"`` as unknown and returns RGB for ``"bgr"``. A TIFF
  photometric name that is no JPEG colorspace (``"CFA"``,
  ``"LINEAR_RAW"``, ``"CIELAB"`` and the like, which tifffile passes for
  DNG and other tiles) means the library default when decoding, as in
  imagecodecs; any other unknown colorspace raises, where imagecodecs
  falls back to the default. ``mozjpeg`` encode takes the parameters of
  ``mozjpeg_encode`` and raises for those MozJPEG's TurboJPEG API fixes
  (``optimize=False``, ``notrellis``, ``quanttable``, ``smoothing``, and
  ``progressive=False``, which used to write the same progressive JPEG as
  ``progressive=True``). ``jpeg`` raises for a stream with a component
  count TurboJPEG has no colorspace for (two, as in a two-sample lossless
  JPEG TIFF) and for a lossless one asked for a color conversion, which
  TurboJPEG's lossless mode does not make; imagecodecs reads both.
  Format change: calls that passed these options now get the stream
  they asked for; output with default options is unchanged.
- Fix: ``jpeg`` could not decode valid JPEG that imagecodecs writes and
  reads: 12-bit DCT (the T.81 extended process), lossless (SOF3) at 2 to
  16 bits, and four-component CMYK or YCCK (Adobe APP14). All decode now,
  to uint16 above 8 bits and to (H, W, 4) for four components, equal to
  imagecodecs' output, and JPEG-in-TIFF tiles of these kinds (a CMYK JPEG
  TIFF, for example) read through the same path. ``jpeg`` also writes
  them: uint16 as 12-bit lossy or 9 to 16-bit lossless, byte-identical
  to imagecodecs at the same settings, and (H, W, 4) as CMYK. For CMYK
  the default bytes differ from imagecodecs': ``jpeg`` writes the Adobe
  APP14 marker that declares four components CMYK (Adobe Technical Note
  5116) and component ids C, M, Y, K, which is what imagecodecs writes
  with ``colorspace="cmyk", outcolorspace="cmyk"`` (``colorspace="cmyk"``
  alone raises there); imagecodecs' default writes no marker and
  ids 0 to 3, which libjpeg also reads as CMYK, so the pixels are the
  same. Values wider than the precision raise. imagecodecs either raises
  or writes a 12-bit lossy stream that decodes to other values, and by
  default it stores lossless uint16 in a 12-bit frame even when samples
  exceed 12 bits, which T.81 does not allow (libjpeg-turbo happens to
  decode it back exactly). ``jpeg`` stores lossless uint16 data above
  4095 at 16 bits instead, so those bytes differ from imagecodecs'. A
  lossless stream asked to decode at a reduced scale raises a clear
  error. ``mozjpeg`` decodes CMYK and YCCK too, and hands 12-bit and
  lossless streams, which MozJPEG cannot decode, to ``jpeg``. Format
  change: ``jpeg`` writes uint16 and (H, W, 4) arrays, which it used to
  reject. The TIFF and NDTiff writers still reject them with
  ``compression="jpeg"``, raising ``JpegError`` as before, because the
  BitsPerSample and PhotometricInterpretation they write would not
  describe a 12-bit or CMYK JPEG.
- Fix: ``jpeg`` and ``mozjpeg`` rejected a trailing singleton channel,
  (H, W, 1). It is grayscale now, as in ``png`` and imagecodecs, and
  writes the same bytes as the (H, W) array.

- Fix: HTJ2K decode ignored the component transform (RCT/ICT) that a
  codestream signals in its COD marker, which imagecodecs, OpenJPH's
  ``ojph_compress``, Kakadu and DICOM encoders all use for RGB and RGBA.
  Such files decoded to wrong colors (uint8 off by up to 255) with no
  error. They now decode as ISO/IEC 15444-1 Annex G requires, matching
  imagecodecs and OpenJPEG, including the JPEG committee's conformance
  codestreams.
- Fix: HTJ2K encode now applies the component transform to 3- and
  4-component input by default, as imagecodecs and ``ojph_compress`` do;
  ``rgb=False`` turns it off. This is a format change: RGB and RGBA
  output differs from 0.4.0 and is byte-identical to imagecodecs'.
  Earlier files still decode.
- Fix: HTJ2K decode clamped samples of more than 16 bits into uint16 or
  int16, and decoded float32 codestreams (NLT type 3, ISO/IEC 15444-2)
  to meaningless int16, both silently. It now returns uint32, int32 and
  float32, and raises on codestreams whose components differ in
  precision, sign or sampling instead of misreading them. Encode accepts
  uint32, int32 and float32 and 1 to 16384 components, the range SIZ
  allows; more raises, where imagecodecs writes an invalid codestream.
- Fix: HTJ2K decode with ``planar=None`` (the default) returned every
  multi-component image as (H, W, C). It now follows imagecodecs:
  (H, W, C) when the codestream uses the component transform, as RGB
  and RGBA encodes do, and (C, H, W) otherwise, for example for 2 or 5
  components or ``rgb=False``. RGB files written by 0.4.0, which have
  no transform, therefore decode as (C, H, W) with the same samples;
  ``planar=False`` returns (H, W, C) for any codestream. The pyramid
  reader and DICOMweb frames still return (H, W, C).
- Fix: HTJ2K ``level`` now means what it means in imagecodecs: below 1 a
  quantization step held as a float32 (under 1e-5, lossless, which
  includes a level of exactly 1e-5), from 1 a quality factor up to 100.
  A level of 1 or more used to be taken as a quantization step, a level
  above 0 up to 1e-5 used to give a lossy file, and a level of 0 raised.
  This is a format change: output for a level of 1 or more, or above 0
  up to 1e-5, differs from 0.4.0. The other ``imagecodecs.htj2k_encode``
  keywords (``rgb``, ``planar``, ``tile``, ``resolutions``,
  ``reversible``, ``tlm``, ``tilepart``, ``block_size``, ``prog_order``,
  ``profile``) are implemented, with encoded output byte-identical to
  imagecodecs, and so are the ``htj2k_decode`` keywords ``planar``,
  ``skipres``, ``resilient`` and ``out``. Encode does not implement
  ``out=``; it raises ``TypeError``, as the codec does for any option it
  does not know, instead of dropping it. The one exception to byte
  identity is an explicit ``rgb=True``, which imagecodecs drops for
  planar and float32 input: planar input gets the component transform,
  and float32 input, or fewer than 3 components, raises ``ValueError``.
- Fix: JPEG 2000 decode returned signed components (Ssiz bit 7) as
  unsigned values offset by half their range, so a TIFF with signed
  JPEG 2000 tiles read back with the sign bit flipped. Signed components
  now decode to int8, int16 or int32, as ISO/IEC 15444-1 Annex A.5.1 and
  G.1 define them, and components above 16 bits decode to (u)int32
  instead of raising. Encode accepts int8 and int16.
- Fix: JPEG 2000 encode failed on any image under 32 pixels on a side,
  because it always asked OpenJPEG for 6 resolutions. It now uses
  imagecodecs' resolution count, and lossless output is byte-identical
  to imagecodecs' at every size. This is a format change for images
  under 256 pixels on a side. Two-component JP2 files now declare the
  gray color space, as imagecodecs writes them, instead of unspecified.
- Fix: JPEG 2000 ``level`` is now imagecodecs' PSNR target in dB
  (1 to 1000), and a level alone gives a lossy file, as in imagecodecs.
  It used to be ignored unless ``lossless=False`` was passed, and was
  then read as a compression ratio of 100/level; ``ratio=`` now asks for
  a ratio. ``lossless=True`` with a lossy level raises. This is a format
  change for lossy output. The lower-level encoder that the TIFF writer
  calls defaulted to a lossy 10:1 rate when given
  no level; it is now lossless by default, like the codec. The NDTiff
  writer with ``compression="jpeg2000"`` used to drop
  ``compression_level`` and write lossless frames; a level from 1 to
  1000 now gives lossy frames with that level as OpenJPEG's PSNR target,
  as in the TIFF writer and imagecodecs, which
  is a format change. Frames written with no level are still lossless.
  The other ``imagecodecs.jpeg2k_encode`` keywords (``codecformat``,
  ``colorspace``, ``planar``, ``bitspersample``, ``resolutions``,
  ``reversible``, ``mct``, ``verbose``) are implemented, and the codec
  raises on options it does not know. Encode does not implement
  ``out=``: the codec raises ``TypeError`` for it. Encode takes up to 4095 components, as imagecodecs does, so the
  TIFF writer now writes JPEG 2000 with more than 4 samples per pixel,
  which it used to refuse; it encodes each strip or tile as rows by
  width by samples, whatever its height.
  ``bitspersample`` takes 1 to 8 for 8-bit data and 9 to 16 for 16-bit
  data, where imagecodecs uses it; other values raise, where
  imagecodecs ignores them. ``verbose`` sends OpenJPEG's messages to
  the ``opencodecs`` logger, at the same thresholds as imagecodecs.
- Fix: GIF decoding ignored the Graphic Control Extension. A frame's
  transparent index was painted in its palette color instead of leaving
  the pixel underneath, disposal methods 2 (restore to background) and 3
  (restore to previous) were not applied, ``decode`` and ``open`` did
  not deinterlace interlaced frames, and the libgif path filled the area
  outside a single sub-rectangle frame with zeros instead of the
  background color. Every decode path now shares one reader whose
  compositor follows the GIF89a specification. ``decode`` also takes
  imagecodecs' ``index`` (one frame over the background, by position or
  keyword) and ``out`` (a C-contiguous array of the decoded shape and
  dtype, else ``ValueError``, as in imagecodecs), and ``asrgb=False``
  returns every frame's indices on a canvas-sized array, ``(H, W)`` or
  ``(N, H, W)``, where it used to refuse animations and return a
  frame-sized array. Four differences from imagecodecs remain. When the
  first frame uses its transparent index, imagecodecs returns a fourth
  channel that is 255 everywhere; opencodecs still returns RGB.
  imagecodecs skips the restore of a disposal 3 frame that follows a
  disposal 2 frame, which opencodecs applies as the specification says.
  A frame that extends past the logical screen is clipped to it, where
  imagecodecs enlarges the canvas. A frame of zero width or height draws
  nothing, where imagecodecs raises ``GifError``; with ``asrgb=False``
  such a frame used to crash the interpreter. A frame whose image data
  ends before width times height pixels now raises ``GifError``, as
  libgif and imagecodecs do, where ``GifCodec.decode`` and ``open``
  returned uninitialized memory for the missing pixels; codes past the
  last pixel are ignored, as libgif ignores them, where those two
  raised; and an LZW code naming a table entry that was never defined
  raises ``GifError``, where it could read outside the decoder's table
  and crash the interpreter. Decoded pixels change; written bytes do
  not.
- Fix: BMP read BI_BITFIELDS color masks from after the 52- and 56-byte
  headers, which is pixel data, so those files decoded with wrong colors
  (or raised ``OverflowError`` at 16 bits), and after a 40-byte header
  it took the first pixel for an alpha mask and returned a fourth
  channel of garbage. Masks are now read where each header defines them;
  a fourth DWORD after a 40-byte header counts as alpha only when it
  lies before the pixel data and is disjoint from the color masks.
  Channels of 1 to 3 bits topped out at 128, 192 or 224 (an opaque 1-bit
  alpha read as 128); every width now scales as
  ``round(v * 255 / (2**n - 1))``, the PNG specification's reference
  equation, which also moves
  some 5- and 6-bit levels by one (imagecodecs truncates, so it can be
  one lower). Uncompressed 1-, 2- and 4-bit paletted files and
  BI_ALPHABITFIELDS now decode, and the imagecodecs parameters ``asrgb``
  and ``out`` (decode) and ``ppm`` (encode), which were silently
  ignored, are implemented; ``out`` must be a C-contiguous array of the
  decoded shape and dtype, else ``ValueError``, as in imagecodecs.
  24-bit BI_BITFIELDS files, which Microsoft documents only for 16 and
  32 bits and imagecodecs refuses, still decode, now through their
  masks, which used to be ignored; one with a zero color mask now raises
  ``BmpError``, as Pillow refuses it, where earlier versions ignored the
  masks and decoded it as BGR. A file whose pixel data offset lies inside
  the three masks after a 40-byte header, so that the masks would be
  pixels, now raises ``BmpError``, where earlier versions read the first
  pixels as masks at 16 and 32 bits (imagecodecs still does) and decoded
  a 24-bit one as BGR; with four masks (BI_ALPHABITFIELDS, which earlier
  versions and imagecodecs refuse) it raises too. A channel mask whose bits are not
  contiguous, which Microsoft's header documentation forbids, and a file
  cut short in its headers, masks, color table or uncompressed rows now
  raise ``BmpError``, where some such files raised numpy's
  ``IndexError`` or ``ValueError`` or ``struct.error``, and a gapped
  mask could also decode to wrong values. An RLE stream cut short
  between codes still decodes, leaving the pixels it never reaches at
  index 0. A paletted file with a bitfield compression now raises
  ``BmpError``, as imagecodecs does, where earlier versions skipped
  three masks and decoded its palette. ``ppm`` is a format change only
  for calls that pass it: a ``ppm`` below 1 is written as 1, as
  imagecodecs writes it, and without ``ppm`` the bytes are unchanged.
- Fix: the ``numpy`` codec could not encode datetime64 or timedelta64
  arrays, wrote Fortran-ordered input in C order with ``fortran_order:
  False``, ignored ``level``, and returned an ``NpzFile`` instead of an
  array for ``.npz`` input. It now writes what ``numpy.save`` writes
  (object arrays still raise ``ValueError``, where ``numpy.save`` and
  imagecodecs pickle them). This is a format change in two places: the
  bytes for Fortran-ordered input change (to ``fortran_order: True``, as
  imagecodecs writes), and ``level``, which used to be ignored, writes a
  deflate-compressed ``.npz`` holding ``arr_0.npy`` with a fixed
  timestamp. ``decode`` returns member ``index`` (default 0) of an
  ``.npz``, raising ``KeyError`` for a member that does not exist. As in
  imagecodecs, ``level`` and ``index`` may be given by position,
  ``decode`` passes other keywords to ``numpy.load`` (``allow_pickle``
  and the like) instead of dropping them, and ``encode`` raises
  ``TypeError`` for a keyword it does not take.
- Fix: ``opencodecs.rgbe_encode``, ``rgbe_decode``, ``rgbe_imread`` and
  ``rgbe_imwrite`` were a second, pure-Python RGBE implementation that
  disagreed with the ``rgbe`` codec. It wrote a ``GAMMA=1.0`` header
  line, which is not a Radiance header variable, and decoded ``+Y`` and
  ``-X`` files unflipped and refused X-first (transposed) files, though
  the resolution string defines the scan order. They now call the codec.
  This is a format change for ``rgbe_encode`` and ``rgbe_imwrite``: no
  ``GAMMA`` line and different RLE run choices, so the bytes are now
  identical to imagecodecs' ``rgbe_encode``. The codec gains
  imagecodecs' ``header`` and ``rle`` options, which it used to ignore:
  ``header=False`` writes and, given ``out``, reads a bare pixel stream;
  ``rle=False`` writes flat pixels, also after a header, where
  imagecodecs writes RLE regardless, so that one combination writes
  different bytes than imagecodecs. With the default ``header=None``, a
  bare stream is read only when ``out`` is given and the data neither
  starts with the ``#?`` magic nor holds a header that parses without it
  (one with a ``FORMAT`` line). A bare stream must fill ``out`` exactly;
  input left over raises ``ValueError``, as in imagecodecs, which also
  catches header text with neither the magic nor a ``FORMAT`` line. The
  codec's own bytes change only for calls that pass ``header=False`` or
  ``rle=False``, which it used to ignore (a format change for those
  calls). The codec now also reads a ``#?`` header with no ``FORMAT``
  line, as Radiance's own reader and the old helpers did, and reads the
  ``FORMAT`` value as Radiance's ``formatval`` does, skipping whitespace
  after ``FORMAT=``; it used to refuse both. A ``FORMAT`` line naming
  only another pixel format (not RGBE or XYZE) raises ``RgbeError``, as
  Radiance's picture readers refuse one, where the old helpers decoded
  those pixels as RGBE. The codec's output buffer was 4 bytes per pixel,
  but run-length encoding a noisy scanline takes up to one more byte per
  128 per channel, so such images raised ``RgbeError`` (imagecodecs
  fails on them the same way); the buffer is now sized for that worst
  case, and every image the old helpers wrote can still be written.
- Fix: BC3 (DXT5) alpha truncated its interpolated values while BC4,
  whose block is the same 8 bytes, rounded them, so the same bytes
  decoded one apart. The Khronos Data Format Specification defines both
  with the same real-valued formulas, so BC3 alpha now goes through the
  rounding BC4 kernel. About a third of interpolated alpha samples
  decode one higher than before and than imagecodecs, which truncates;
  BC3 decode is slower for it (see Speed). BC4 and BC5 already rounded, and
  BC6H keeps its float32 default (``fp16=True`` is bit-identical to
  imagecodecs); both are now documented.

0.4.0 (2026-09-29)
------------------

How opencodecs spends threads, especially when several of your threads
call it at once, and one place for each thing it does differently by
operating system. Figures compare against 0.3.1 on a 20-core arm64 Mac and
a 64-core x86-64 Linux workstation: both builds in fresh processes,
alternating, each process one sample, identical decoded pixels required.

- **One shared pool.** ``run_batched``, ``map_batches`` and ``map_bounded``
  (under TIFF, DICOM, EER, FITS, HDF5, CZI, Zarr and more) built a
  ThreadPoolExecutor per call. They now share one process-wide pool. A
  144-tile TIFF reads 1.34x faster on Linux (9.2 to 6.9 ms) and 1.11x on
  the Mac.
- **Concurrent callers share instead of multiplying.** A call counts as
  in flight for its whole duration, and an automatic worker count is its
  share of what the calls in flight divide: the pool helpers' workers,
  which hand the GIL back and forth, share 32 between calls (each call
  still takes at most 16), and native codec threads share the CPU count.
  A lone call keeps full width and an explicit ``numthreads`` is honored
  as given. With the TIFF work below, two and eight threads reading a
  tiled deflate TIFF at once: 3.46x and 4.63x on Linux, 3.61x and 4.01x
  on the Mac; eight JPEG XL readers 1.86x on Linux and 1.03x on the Mac.
- **JPEG XL sizes its threads to the image.** One-shot ``decode`` used
  libjxl's default of one thread per hardware thread, all created for
  each call whatever the image size. It now reads the size from the
  header and uses about one thread per 64K pixels, up to 20. 512 x 512:
  2.54x faster on Linux (12.9 to 5.1 ms, 5 threads instead of 129) and
  1.24x on the Mac; 2048 x 2048: 1.10x for one caller and 1.56x for
  eight on Linux, parity on the Mac.
- **AVIF decode uses up to 8 threads by default** instead of one per
  core: AV1 decodes in parallel across tiles, and past 8 threads there
  was nothing left to gain. 1.07 to 1.13x on Linux, 1.00 to 1.03x on the
  Mac. ``numthreads=0`` still means every core.
- **TIFF segments decode, un-predict and land in one native call.** A
  tile used to be inflated in one call, un-predicted in a second and
  copied into the output with numpy, with a GIL handoff between each,
  which is what capped concurrent readers. Now a batch of segments goes
  through all three under one GIL release (``_tiff.decode_segments_into``),
  for no compression, deflate, zstd, LZW and PackBits with predictors 1
  to 3, and a whole strip decompresses straight into its output rows.
  zlib and zstd are reached through a C table their own modules export,
  so ``_tiff`` links neither. Separate planes, bilevel and byte-swapped
  files keep the general path, as does any segment the native call
  rejects, which also keeps its exact error. Against 0.3.1, a tiled 4096 x
  4096 uint16 image with the horizontal predictor: one reader 2.94x on the
  Mac and 2.38x on Linux, eight readers 3.65x on both; a 144-tile image
  1.98x and 1.96x; the same pixels in strips 1.28x and 1.37x for one
  reader, 1.59x and 1.94x for eight. Serial reads are unchanged or faster.
- **Uncompressed TIFF strips laid out end to end go straight into the
  output** (``core.io.read_file_into``): positioned reads split across
  the shared pool, where one thread used to copy them out of the file
  mapping. One reader 2.29x on the Mac and 3.79x on Linux; with
  ``numthreads=1``, 1.65x on the Mac and level on Linux; eight readers
  1.66x on Linux and level on the Mac, where both versions copy from the
  mapping. One choice depends on the kernel, and is named for it: macOS
  copies one thread's worth of bytes out of the reader's file mapping
  faster than it reads them, and Linux reads faster, alone and with
  eight readers alike, so that is what each does.
- The README compares TIFF reads directly with tifffile: 2.7 to 4.3x
  faster on the Mac and 4.0 to 7.7x on Linux for one reader of a tiled
  compressed image, up to 13.4x for eight concurrent readers.
- **Ultra HDR** runs its kernels on the shared pool and its encode and
  decode steps on a persistent pool of their own, instead of building up
  to five pools per call: encode 1.06x on the Mac and 1.30x on Linux,
  decode 1.09x on both.
- **Each operating-system difference is named once.** ``setup.py``
  states its platform conventions in one place and links the library
  files it finds by path on every platform; descriptor reads and writes,
  Windows' binary-mode flag and uncached opens each have one
  implementation; capabilities are probed instead of inferred from the
  platform. A test lists the nine operating-system branches left in the
  package and ``setup.py``, each with the reason it cannot be a probe,
  and fails on a new one.
- **A CZI file's compression can be rewritten without losing its
  container.** ``czi_recompress(src, dst)`` re-encodes every sub-block
  and keeps its dimensions, scene, mosaic index and pyramid level, and
  the metadata XML; ``write_frame`` and ``write_many`` take a sub-block's
  own dimension list, and ``subblock_dims`` builds one from a reader's
  entry. libCZI reads the output of a 481 sub-block slide scan and
  agrees with the input. 185 MB takes 122 ms against 340 ms for ZEISS's
  czicompress, and 297 ms with ``verify=True``, which decodes every
  sub-block back before it is written.
- **Windows wheels use libdeflate**, as macOS and Linux wheels already
  did. conda-forge names its import library ``deflate.lib``, which the
  old probe did not recognize, so Windows fell back to zlib. On a Windows
  VM, against 0.3.1: deflate encode 2.79x and decode 1.79x, PNG encode
  2.82x (decode stays on zlib, level), a tiled deflate TIFF read 1.33x;
  output at the default level is 0.7% (deflate) to 1.5% (PNG) larger,
  the same trade macOS and Linux make.
- ``import opencodecs`` on macOS ran ``mount`` once per extension, about
  4.5 ms each; the mount table is now read once, about 200 ms per import.
- Fix: the DICOM codec dropped ``numthreads``, so ``numthreads=1`` still
  decoded on several threads.
- Fix: the NDTiff writer counted the buffers after a partial ``writev``
  twice, so its position ran ahead of the file, and ``tiff_reader`` took
  one ``os.pread`` as complete even when it returned short.
- Fix (building from source): with a per-user cache of source-built
  libraries present, a Windows build handed MSVC ``.so`` paths and
  ``-Wl,-rpath`` flags; the MozJPEG extension on Linux had no rpath to its
  own library, so the loader could take the system's libturbojpeg for it;
  ``bench/build_codec_libs.sh`` passed ``-mcpu=apple-m1`` on Intel Macs.
- Fix: pools created before ``os.fork()`` hung in the child, which
  inherits the pool object but none of its threads (the first submit
  in a forked DataLoader-style worker waited forever). A forked child
  now builds fresh pools.

0.3.1 (2026-09-28)
------------------

A CZI read scheduling fix, and a clear refusal for the files that cannot
stack. Figures compare against 0.3.0 on the same machines.

**CZI whole-stack reads hand out work by bytes**

``read`` derived its batch size from the worker count, giving each worker
``ceil(n/workers)`` consecutive sub-blocks. That makes the read wait on
that many sub-blocks even when most workers have already finished: nine
8 MiB sub-blocks over eight workers is two rounds, and measured 15.1 ms
against 7.9 ms for nine one-sub-block tasks. Tasks are now sized by
output bytes, about 8 MiB each, so a large sub-block gets a task to
itself and small ones ride together. That keeps the reason batching
existed, which is that a 0.7 MiB tile cannot pay for its own future,
without paying the rounding. ``_READ_MAX_WORKERS`` goes 8 to 32: the old
value was measured against the dispatch above, where extra workers could
not help. With byte-sized tasks, 20 to 32 measured fastest on both a
20-core arm64 Mac and a 128-core x86-64 Linux host, and past 32 the Linux
host gives it back to memory-bandwidth contention. A semaphore keeps the
resolved worker count a real bound when the byte sizing yields more tasks
than that.

Eight files of 2000 x 2000 uint16 ZSTDHDR sub-blocks, read back to back:
22.1 to about 15 ms per file on macOS, which moves a whole-stack read
from a little behind aicspylibczi to about 1.3x ahead of it.

**A mixed sub-block CZI says so, instead of failing inside the codec**

``read`` stacks every sub-block into one array, and sized that array from
the first sub-block, which is wrong for any file whose sub-blocks differ:
a pyramidal slide scan interleaves down-scaled sub-blocks with
full-resolution ones, and a mosaic can clip its edge tiles. The decode
destination check then failed with an element count the caller never
chose, naming neither the file's shape mix nor a way forward. It now
refuses up front, listing the shapes it cannot stack and the APIs that do
work (``CziPyramidReader.read_region``, ``entries_at_level``,
``read_tile`` and ``iter_tiles``), and the new ``CziReader.is_uniform``
lets a caller branch before asking. Decoding was never at fault: JPEG XR
sub-blocks are pixel-exact against aicspylibczi.

0.3.0 (2026-09-27)
------------------

Mostly a speed release. Figures below compare against 0.2.0 unless they
say otherwise; the README's Performance section has the full table for a
Mac and a Linux workstation.

**Shared bounded pipeline across readers and writers**

A common layer in ``opencodecs.core`` now carries the work that each
format used to do its own way: ordered, byte-bounded parallel
scheduling (``core.pipeline.map_bounded``) with a shared worker
budget, per-worker scratch buffers, caller-owned output buffers and
sinks for the native codecs, stateful streaming iterators for seven
byte codecs, native PNG row sessions, selective range sources, and
opt-in bit-exact verification of written segments. Each piece of work
is recorded, with its measured limits, in ``pipeline_catalog.toml``
and checked by ``ci/check_pipeline_catalog.py``.

**Zarr, TIFF and blosc2 under many threads**

- Zarr regional reads can run in parallel, with ``num_workers`` on
  ``read_region`` or ``OmeZarrArray``; the default stays serial. Chunks
  are grouped into tasks of about 2 MiB on a persistent pool, chunk files
  are read with os-level calls, and plain zstd chunks decompress straight
  into the output. A 4096 x 4096 array in 256 x 256 chunks, 0.2.0's serial
  read against eight workers: zstd 72.9 to 19.6 ms on Linux (50.1 to
  14.9 ms on macOS), zlib 184 to 38.8 ms (80.5 to 25.9 ms).
- Zarr v2 ``blosc`` chunks decode through the native blosc2 extension
  instead of numcodecs, whose binding serializes calls across threads:
  the default serial read goes 37.5 to 20.0 ms on macOS and 35.1 to
  29.4 ms on Linux.
- Fix: blosc2 encode selected its compressor through process-global
  state, so concurrent encodes could use another thread's compressor.
  Encode, decode and partial decode now each use a context of their own,
  which also stops them queuing on blosc2's global mutex: from eight
  threads, 64 chunks of 1 MiB decode 6.2 to 6.5x and encode 6.6 to 7.1x
  faster. Chunks written with item size 1 and shuffle carry slightly
  different header flags; old and new chunks decode in both versions.
- TIFF tiles with the horizontal predictor decode into a writable array,
  so the predictor no longer copies each tile first (1.1 to 1.2x on a
  tiled 4096 x 4096 deflate image).

**CZI: faster reads and writes, and real slides read correctly**

- JPEG XR sub-blocks (compression 4, used by most Zeiss slide scans) now
  decode through a new optional ``_jpegxr`` extension over jxrlib, built
  when jxrlib is installed. Every tile of the Axioscan corpus slide and its
  whole 20684 x 32751 level 0 match czifile exactly. The wheels build
  jxrlib from source (conda-forge on Windows) and link it statically.
- Fix: overlapping mosaic tiles composed in directory order. Zen writes
  tiles out of mosaic-index order, and libCZI and czifile draw the higher
  mosaic index on top; each level now composes in that order, which on the
  corpus slide changes 8.9 million overlap pixels at level 0.
- Fix: ``CziPyramidReader`` took sub-block starts as level pixels.
  Zen stores them in full-resolution slide coordinates, often far from
  zero and negative, so a real slide reported every level as ``(N, 0)``.
  Levels now count from the slide origin (``reader.origin``, matching
  czifile) and divide positions by the level's scale.
- Sub-blocks decode straight into their destination. Whole-stack
  ``read`` accepts ``out=`` (``numpy.memmap`` included). zstd
  decompression and the byte unshuffle run as one native call, which
  let several threads decode small tiles at once: a stack read with eight
  workers goes 12.0 to 3.9 ms on Linux (2.5 to 1.7 ms on macOS).
- Regional reads use a lazily built spatial index and write each tile
  straight into the output; overlapping tiles still resolve to the
  later one in the directory. ``PyramidReader.read_regions`` decodes
  the union of a batch of boxes once, and ``decoded_cache_bytes`` adds
  an opt-in decoded-tile cache.
- Opening is faster: the directory parses in about a third of the
  time and the pyramid reader no longer rescans it per level (180 ms to
  3 ms on a 12k-sub-block slide).
- ``read_regions`` on CZI decodes each tile shared by several boxes once,
  straight into every box's output: 50 disjoint crops take 5.1 ms on
  Linux against 29.8 ms one call at a time in 0.2.0 (3.1 against 17.2 ms
  on macOS).
- The writer shuffles natively, emits sub-blocks as parts without
  copying unverified payloads, and ``write_many`` compresses frames on
  workers while keeping the file byte-identical to sequential writes.
  Eight 2048 x 2048 frames with zstd: 189 to 89 ms one frame at a time on
  Linux and 41 ms with ``write_many`` (82 to 54 and 20 ms on macOS).
- ``CziWriter(background_encode=True)`` compresses frames of 1 MiB or
  more on a background thread while the caller prepares the next one.
  Opt-in: with work between frames it wrote 1.44 to 1.46x faster, with
  frames ready and nothing to overlap it was 0.87 to 0.96x.

**Build and tests**

- Every wheel now carries the same 40 compiled extensions, the 39 of
  0.2.0 plus ``_jpegxr``, and ``ci/check_wheel_contents.py`` requires it.
- SZ3 3.3.2, since upstream deleted the 3.3.1 tag.
- The DICOMweb client is tested over HTTP against a local server
  (``tests/_dicomweb_server.py``) serving a synthetic study written by
  reference encoders, instead of a public demo server.
- Fix: the pipeline catalog check accepted a rooted evidence path such
  as ``/etc/passwd`` on Windows, where ``Path.is_absolute()`` needs a
  drive letter.

0.2.0 (2026-09-09)
------------------

**AVIF: tile-parallel encoding, chroma layout, encoder backend**

``_avif.encode`` gains ``tile_cols_log2`` / ``tile_rows_log2`` /
``auto_tiling``, ``yuv_format``, ``codec`` and ``codec_options``.
Tiling now defaults to 4x4 once the long axis reaches 1024 px, which
measured ~2x faster encode for +4.4% bytes at unchanged PSNR on a
2048x2048 RGB frame. ``bench/build_codec_libs.sh`` builds SVT-AV1 and
wires it into libavif, so ``codec='svt'`` is selectable, though libaom
remains the default and the faster one for stills. See
``docs/avif_backends_and_tiling.md`` for the measurements, the reason
SVT loses, and the Homebrew link-order wart that makes ``codec='svt'``
fail with "No codec available" on a dev machine.

**Fix: _plio bypassed the shadow-copy loader and hung the test suite**

``_plio`` was missing from ``_EXTENSIONS`` in
``opencodecs/codecs/__init__.py``, so it was imported straight off the
network mount instead of the local cache copy. On macOS that parks
inside dyld indefinitely with no error:
``tests/test_fits.py::test_compressed_fits_plio_1_roundtrip`` hung
forever rather than failing. Listing it takes ``tests/test_fits.py``
from a hang to 31 passed in 3 s. Also dropped the stale ``_ultrahdr``
entry left over from the ``_uhdr`` rename. New
``tests/test_codec_registry.py`` asserts that ``_EXTENSIONS`` and the
set of ``.pyx`` sources agree in both directions, so the next
occurrence is a failing test rather than a hang.

**RGBE: accept every Radiance resolution-line orientation**

A reader that matches only ``-Y H +X W`` rejects legal files. The signs
select the direction of each axis and putting the X token first stores
the image column-major, so the resolution line is now parsed generically
and the decoder flips or transposes to suit. Five forms are accepted,
one more than OpenImageIO, the extra being ``+Y H -X W``.

Implemented from the Radiance format rather than ported from
OpenImageIO. Their reader is welded to OIIO's own I/O abstraction, so
there was nothing to lift, and importing Apache-2.0 code would have put
a NOTICE requirement and a patent clause on a file that carries neither.

Fair before-and-after for the exponent tables, both builds warmed and
measured over 20 runs, since the first numbers reported for this work
were taken cold and understated it:

.. code-block:: text

    decode 1 MP    2.57 ms -> 1.36 ms    408 -> 774 MP/s   1.89x
    decode 4 MP   10.51 ms -> 5.75 ms    399 -> 730 MP/s   1.83x
    encode 1 MP    5.72 ms -> 5.03 ms    183 -> 208 MP/s   1.14x
    encode 4 MP   22.99 ms -> 20.42 ms   183 -> 205 MP/s   1.13x

**Audited every vendored source against upstream; RGBE decode 1.7x faster**

Generalizing the cfitsio finding, each third-party source under
``3rdparty/`` was diffed against its current upstream:

.. code-block:: text

    libspng     spng.h identical; spng.c current (upstream unchanged
                since 2023) plus our own SIMD defilter
    qoi         identical to upstream
    bitshuffle  three lint casts, no logic change
    bcdec       one cast, verified equivalent under -funsigned-char;
                now byte-identical to upstream
    cfitsio     two missing bounds checks, fixed separately
    rgbe        no live upstream

``rgbe`` deserves the note. It is not stale for want of a maintainer:
the format was frozen in 1991 and the maintained implementations all
descend from Bruce Walter's 1995 reference. OpenImageIO's
``hdrinput.cpp`` says so in its own header, and three.js's RGBELoader is
adapted from the same file. There is nothing newer to move to.

Compared against OpenImageIO rather than assumed, since it is the most
actively maintained C++ version. Its RLE validation turns out to be
check-for-check identical to ours, so the fixes this copy already
carried had brought it to par. It did have one thing we lacked.

``rgbe2float`` called ``ldexp()`` once per pixel, a libm call in the hot
loop of every decode. The exponent is one byte, so all 256 scale factors
fit in a compile-time table, which OpenImageIO already does. Entry 0 is
set to zero rather than ``2**-136`` so the nonzero-pixel branch folds
away as well, which OpenImageIO still pays for:

.. code-block:: text

    1024x1024 (1 MP)    2.6 ms -> 2.2 ms   404 -> 486 MP/s
    2048x2048 (4 MP)   10.6 ms -> 6.0 ms   396 -> 695 MP/s   1.77x

Output is bit-identical, cross-checked against imagecodecs' independent
per-pixel ``ldexp`` decoder over inputs spanning 40 orders of magnitude
plus zeros, which exercise the ``e == 0`` entry. Exactness matters here
because entries below 2**-126 are denormal.

**Security: vendored cfitsio was missing two upstream bounds checks**

Auditing what remained of imagecodecs-derived code turned up a broader
problem: the vendored ``3rdparty/cfitsio/`` sources predate fixes that
upstream has since shipped.

* ``pliocomp.c`` had no bounds checking at all. cfitsio added a
  ``srclen`` argument to ``pl_l2pi`` plus two range checks on 2026-07-21
  ("Added checks for group key overflow and for plio compression"). Our
  copy predated it, so a crafted or truncated PLIO line list read past
  the payload. Against the pre-fix decoder a 4000-case fuzz corpus
  killed the interpreter with SIGBUS; with the upstream file and the
  length passed through, all 4000 are handled cleanly. Reachable from
  any FITS file, since ``_fits_compressed.py`` hands heap bytes straight
  to the decoder.
* ``ricecomp.c`` was missing the guard upstream added on 2025-03-03
  ("Added a buffer size check to fits_rdecomp"). The first pixel of a
  Rice stream is stored unencoded, so a payload shorter than one pixel
  was read past. Applied to all three decoders, not just the 4-byte one
  upstream guards. Not reachable through ``rcomp_decode``, whose framing
  header rejects short blobs first, but ``decode_raw`` is public.

``fits_hcompress.c`` is byte-identical to upstream 4.7.0 and
``fits_hdecompress.c`` differs only in comments, so those needed nothing.

The lesson is recorded in THIRD-PARTY.md: a vendored copy is a snapshot,
and pinning to someone else's fork of upstream can mean inheriting their
snapshot date rather than upstream's current state.

**EER decoder is now ours, and 1.7x faster**

Replaces the vendored excerpt of imagecodecs' ``imcd.c`` with an
implementation in ``3rdparty/oc_eer/``, written against the bitstream
layout documented by RELION's ``renderEER.cpp`` and validated on genuine
Falcon 4 output. ``3rdparty/imcd_eer/`` is deleted. No imagecodecs code
remains in the repository.

Two format details were established from the reference implementation
rather than guessed:

* **Sub-pixel inversion is width-independent.** RELION XORs the packed
  symbol with ``0x0A`` for 2+2 bit fields and ``0x03`` for 1+1, and both
  constants flip exactly the MSB of each field. imagecodecs applies that
  inversion at widths 1 and 2 but not 3 and 4. Falcon hardware only ever
  emits 1 or 2, so real data is unaffected, but we apply the rule
  uniformly and pin the divergence in a test.
* **A frame ends by landing exactly on its last cell.** All 12
  EMPIAR-10568 frames checked terminate with position exactly equal to
  the frame size and 1 to 120 bits of footer left over, so leftover bits
  cannot indicate a bad shape. Overshooting the frame does.

Measured against ``imagecodecs.eer_decode`` on a real 4096x4096 frame:

.. code-block:: text

    superres=0  ->  4096x4096     1.15 ms vs 2.02 ms   1.76x
    superres=1  ->  8192x8192     2.55 ms vs 4.27 ms   1.67x
    superres=2  -> 16384x16384    5.73 ms vs 9.55 ms   1.67x

Output is byte-identical to imagecodecs on all 12 real frames at every
super-resolution level.

``tests/download_test_corpus.sh --eer`` fetches the EMPIAR-10568
micrograph (CC0, ~220 MB) into the gitignored ``.test_data/``. The real
-data tests skip cleanly without it, so CI is unaffected; the synthetic
frame generator in ``tests/test_eer.py`` covers the same paths without a
download.

**TIFF LZW encoder is now ours, and 1.4x faster**

Replaces the vendored excerpt of imagecodecs' ``imcd.c`` with an
implementation written against TIFF 6.0 section 13, living beside our
existing decoder in ``3rdparty/oc_tifflzw/``. ``3rdparty/imcd_lzw/`` is
deleted.

The dictionary lookup is an open-addressed hash table over the 20-bit
(prefix, suffix) key, sized at 8192 so the load factor stays at 0.5 and
the index is a mask rather than a modulo. Each slot packs the epoch it
was written in alongside the key, so CLEAR is an increment instead of a
32 KiB memset, and a stale slot reads as both "absent" and "reusable"
in the comparison the lookup already performs.

Measured against ``imagecodecs.lzw_encode`` on 4 MB inputs (Apple
Silicon), with output within 0.03% of its size in every case:

.. code-block:: text

    all zeros            22.8 ms vs 22.6 ms   0.99x
    photo-like           25.2 ms vs 36.7 ms   1.45x
    long runs            15.2 ms vs 30.1 ms   1.98x
    incompressible       25.3 ms vs 29.2 ms   1.15x
    16-bit image         17.9 ms vs 34.5 ms   1.93x
    total               106.3 ms vs 152.9 ms  1.44x  (188 vs 131 MB/s)

Epoch tagging is what makes the incompressible case work: it resets the
table every ~3836 codes, and with a memset-per-CLEAR that path measured
0.82x, slower than the code it replaced.

20 new tests in ``tests/test_tiff_codec_encode.py`` cover the empty
input, every code-width boundary, table overflow and the expansion
bound, and check each stream decodes correctly through
``imagecodecs.lzw_decode`` as an independent reader.

**Licensing: no imagecodecs-derived Cython source remains (#1)**

Follow-up to the attribution commit. Rather than keep the two copied
``.pxd`` files under a BSD-3-Clause notice, they are gone:

* ``codecs/libjxl.pxd`` rewritten from the libjxl 0.11.2 public headers
  (``jxl/types.h``, ``color_encoding.h``, ``codestream_header.h``,
  ``decode.h``, ``encode.h``, ``parallel_runner.h``,
  ``thread_parallel_runner.h``). It declares only the surface
  ``_jxl.pyx`` actually calls, 113 of the 332 symbols the old file
  carried, and structs list only the fields we touch. 1191 lines down
  to 346. What still matches imagecodecs' version is enum member lists,
  which the C headers fix and which cannot be renamed or reordered.
* ``codecs/libultrahdr.pxd`` deleted. Nothing imported it. ``_uhdr.pyx``
  uses ``libuhdr.pxd``, which is our own; the copied file was left
  behind by the ``_ultrahdr`` to ``_uhdr`` rename, the same rename that
  left the stale ``_EXTENSIONS`` entry fixed earlier.

The fourteen other ``.pxd`` files keep a note pointing at their
imagecodecs counterpart, but the note no longer carries a copyright
line or implies the file is derived. They transcribe the same upstream
C headers, which is why they resemble each other.

Full suite unchanged at 1696 passed, 37 skipped, 5 xfailed.

**Licensing: restore attribution for imagecodecs-derived files (#1)**

Christoph Gohlke reported (#1) that Cython sources derived from
``imagecodecs`` carried no copyright notice, in breach of clause 1 of
its BSD-3-Clause license. Audited the tree against imagecodecs
2026.6.6 and corrected it:

* ``codecs/libjxl.pxd`` and ``codecs/libultrahdr.pxd`` were taken from
  imagecodecs (the latter byte for byte, the former with one added
  declaration). Both now carry the full BSD-3-Clause notice and a
  statement of what was changed.
* Fourteen further ``.pxd`` files declare the same upstream C APIs that
  imagecodecs also declares and overlap with its counterparts as a
  result. Each now credits the corresponding imagecodecs file.
* Added a root ``LICENSE``. ``pyproject.toml`` declared BSD-3-Clause
  and ``MANIFEST.in`` referenced a ``LICENSE`` that did not exist, so
  no license text was shipped in the sdist or the wheels at all.
* Added ``THIRD-PARTY.md``: full inventory of vendored source, the
  imagecodecs-derived declarations, and the codec libraries bundled
  into the binary wheels. Wired both files plus every
  ``3rdparty/*/LICENSE`` into ``license-files`` (PEP 639, so the build
  now needs setuptools >= 77) and ``MANIFEST.in``.

The ``.pyx`` implementations, the pure-Python package and the test
suite were checked and are not derived: the longest shared run of
non-comment lines between any ``.pyx`` file and its imagecodecs
namesake is six lines of ``opj_*`` call boilerplate in ``_jpeg2k.pyx``.


**Reading multi-image files: AVIF sequences, HEIF sets, animated WebP**

AVIF carries image sequences and HEIF carries sets of top-level images
(a burst, a Live Photo's stills, a depth map beside its color image);
both were decoded as a single picture. Animated WebP was not read at
all -- ``decode`` failed with the bare message "WebP decode failed",
which told a caller nothing. ``open()`` now returns a frame-oriented
reader for all three, and ``frame_count()`` answers from the container
without decoding.

**Breaking:** ``avif.decode()`` on a SEQUENCE now returns every frame,
shaped ``(frames, H, W, C)``, rather than the first image. ``webp``
does the same for an animation. That matches ``gif``, which has always
returned the stack for a time sequence, and matches what imagecodecs
returns for the same files. A still is unchanged, ``out=`` on a still
is unchanged, and ``out=`` on a sequence is refused rather than
silently filled with frame 0. If you relied on one frame, use
``open()`` and index it.

``heif`` deliberately does not follow: a HEIF is a SET with a
standard-designated primary image, not a time sequence, and its members
need not share a shape, so ``read()`` there returns a list when they
differ.

**Whole-slide pyramids: DICOM VL Whole Slide Microscopy, and VSI/ETS**

Two formats whose pyramid is not one file. A DICOM slide is a SERIES of
instances sharing a SeriesInstanceUID and distinguished by Total Pixel
Matrix Columns/Rows, so ``open_pyramid(directory, format="dicom")``
takes a directory or a list, orders levels by extent rather than by
filename, and refuses a directory holding two series instead of
building a pyramid from two slides.

VSI was recorded for months as blocked on a sample. It was not: the
pyramid lives inside each ``.ets``, indexed by the LAST coordinate of
each tile-table record, and two header fields had been misread. The
field taken for the table's byte size is a record COUNT (a 413-record
table read as 413 bytes gives up after 11 entries), and a record is
self-describing rather than fixed-width, ``4 * (ndims + 5)`` bytes. The
level axis is derived and then verified: ``parse_ets`` reports a
pyramid only when the tile population of every level equals the grid
covering the extent halved that many times, which the untiled corpus
file fails, so it still reports one level. ``read_region`` decodes only
the tiles a box covers -- a 256x256 window of an 8022x9367 slide moves
0.5 MB of a 32.6 MB file over HTTP.

**Five codecs were holding the GIL through their decode**

``jpeg2k``, ``mozjpeg``, ``charls``, ``zfp`` and ``openjph`` called
their library's decode without releasing the GIL, so every threaded
caller paid a thread pool's overhead to take turns. All measured 0.94x
to 1.12x on eight threads before, and about 7x after. Nothing else
could have caught it: a codec that serializes every threaded caller
passes every correctness test it has. ``openjph`` is the instructive
one, because its ``.pxd`` already declared the shim functions
``nogil`` -- which means "safe to call without the GIL", not "called
without it" -- so reading the declaration would have cleared it.
``tests/test_decode_releases_gil.py`` measures each codec so the next
one added gets asked the same question.

**Parallel decode where the pieces were already independent**

FITS compressed-image tiles (HCOMPRESS_1 603 -> 87 ms, RICE_1 107 -> 19
ms), HDF5 chunks (134 -> 13 ms, via a parallel path that was already
written and simply never called), DICOM frames, EER frame accumulation
(1124 -> 214 ms, bit-identical), BCn bands (BC7 4096x4096, 70 -> 6 ms),
and zfp blocks (110 -> 30 ms end to end). ``core/parallel.py`` holds
the one policy they share: how many workers to spend, and contiguous
batching per worker rather than one future per piece.

One of those needed a fix in a vendored library rather than in the
codec. cfitsio's ``fits_hdecompress`` keeps its bit-reader position in
three file-scope variables, so overlapping HCOMPRESS_1 tiles corrupted
each other and returned status 414, a FORMAT error for a CONCURRENCY
bug. The vendored copy marks them thread-local; single-threaded
behavior is byte-identical.

**Reading at offsets instead of swallowing files**

``dm`` walks its tag tree through a sliding window and reads image data
at the absolute offsets the tags already carried, so opening an 8.39 MB
dm4 moves 0.197 MB. ``nrrd`` reads a slice at its own offset,
``nifti`` a plane, ``oib`` only the streams behind one index (12.8 MB
against 25.4 MB over HTTP), and ``lif`` and ``eer`` index at their own
offsets rather than walking (EER frame 720 of 721: 1200 ms to 1.70 ms).
There is a native ``.npy`` reader. ``jpeg2k`` gains ``decode_region``
and ``decode_tile``; ``zfp`` gains ``decode_block`` for fixed-rate
streams, where one block costs 0.0010 ms against 0.209 ms for the whole
stream; ``bcn`` gains ``decode_rows``.

``gzip`` allocates its output once from the ISIZE in the trailer rather
than growing and joining buffers: 148 MB peak to 67 MB for a 67 MB
result, and 13.0 ms to 6.8 ms.

**Every codec accepts an http(s) URL**

``read_src`` treated a URL as a filename, so about thirty codecs could
not open one at all, and no per-codec test would have found it because
each looks correct given a path. Formats that reach storage by offset
fetch only what they need; the whole-codestream formats fetch once,
which is the honest thing when every byte is needed anyway. ``tiff``
and ``jxl`` had range-reading paths that ``open()`` and ``decode()``
were not routing a URL into.

**Capability manifest: no open gaps**

``capabilities.toml`` went from 47 open capabilities across 31 codecs
to none: 25 done, 35 recorded as not applicable with reasons.
Correcting it was as valuable as closing it, because the file was wrong
in both directions and none of it was visible by reading the code. Five
capabilities existed and were recorded absent (a flag never set; offset
arithmetic living in a helper named after the container, which the
manifest attributes by filename; a memory property no correctness test
can observe), and one was claimed and absent -- ``eer`` advertised cheap
random access while indexing walked the file. ``streaming_decode`` also
presupposes a frame axis, and fifteen byte-compressor and single-image
codecs carried it as a gap they could never close; the checker now
enforces that invariant.

**Corpus and tooling**

``corpus.py coverage`` reported what the manifest DECLARED rather than
what is on disk, so "every codec has a native fixture" meant "every
codec is mentioned". It checks the disk now, and separates a deliberate
opt-in (the 1.1 GB QOI set) from an oversight. ``kodak24`` was
unfetchable through the manifest because its path held a shell
placeholder that only the shell fetcher could expand. Licences: 26 of
40 datasets now carry terms with a cited source, and the other 14 each
record what was checked and why it did not resolve, rather than a bare
"unverified".

**Fix: four advertised codecs were in none of the wheels we published**

``jpegls`` (CharLS), ``snappy``, ``gif`` and ``deflate(backend="isal")``
are listed in the README as supported. Reading the published 0.1.13
wheels back shows ``_charls``, ``_isal`` and ``_snappy`` missing on all
four platforms and ``_gif`` present only on macOS, where giflib happens
to arrive as somebody else's Homebrew dependency.

The cause was that giflib, snappy, CharLS and isa-l appear in no CI
install path at all: not ``ci/environment.yml``, which drives every
tests.yml job and the Windows wheels, not the manylinux ``dnf`` list,
not the macOS ``brew`` line. setup.py's header probe then dropped each
extension, and nothing failed. ``pytest.importorskip`` at the top of
``test_charls.py``, ``test_gif.py``, ``test_isal.py`` and
``test_snappy.py`` removed 56 tests from the run, and
``test_optional_backend.py``, one of the two files the wheel smoke test
executes, exists precisely to pass when a backend is absent. The build
and the suite were both green the entire time.

All four are now installed everywhere wheels and tests are built:
conda-forge for tests.yml and the Windows wheels, brew on macOS,
``dnf`` for snappy/giflib/isa-l on manylinux with CharLS source-built
beside the other Tier 1 libs, as neither AlmaLinux 8 nor EPEL 8
packages it. ``snappy-devel`` and ``giflib-devel`` live in PowerTools,
so that repo is enabled explicitly. All four are added to
``ci/check_wheel_contents.py``'s ``MUST_SHIP_ALL_PLATFORMS``, which
fails the build rather than publishing a wheel that quietly lost a
codec, the way v0.1.2 lost ``_sperr`` and ``_brunsli`` on Windows.

Two build bugs surfaced only once the libraries were actually present,
both found on a Windows host before pushing. ``_isal`` did not compile
at all under MSVC: ``isal_shim.c`` and the verbatim C block in
``_isal.pyx`` both name ``ssize_t``, which is POSIX. Cython rewrites a
``cdef ssize_t`` to ``Py_ssize_t``, which is why ``_tiff``, ``_jxl``
and ``_eer`` were fine, but a ``cdef extern from *`` body is copied
through untouched and the shim includes no Python headers. Adding
isa-l to the conda environment without this fix would have turned every
Windows job red rather than merely dropping the codec. The shim now
carries the same ``HAVE_SSIZE_T`` guard ``3rdparty/rgbe/rgbe.h``
already used.

The other was ``_maybe_build_ext_simple``, the CharLS resolver, which
searched neither ``$CONDA_PREFIX`` nor ``$OPENCODECS_CODEC_LIBS_PREFIX``
and so could only ever match on a developer's machine. It also looked
only in ``<prefix>/lib``, missing AlmaLinux's ``lib64`` and
conda-on-Windows's ``Library/lib``, and had no Windows import-library
pattern at all. conda-forge names that file ``charls-2-x64.lib``, so
the link name is now taken from whatever is on disk. ``_isal`` had the
same shape of bug in a simpler form: a hardcoded ``libraries=["isal"]``
that is correct on POSIX and wrong on Windows, where the package ships
``isa-l.lib``; a new ``_lib_link_name`` probe picks the right one.

0.1.13 (2026-06-04)
-------------------

**UHDR fast path: parallel kernels + Chrome rendering fix**

Four Cython kernels in ``opencodecs.codecs._uhdr`` now dispatch
pixel chunks across a ``ThreadPoolExecutor`` and call their existing
``nogil`` kernel implementations in parallel:

* ``compute_gain_map_u8`` — gain-map encode kernel (~12× at 16 threads)
* ``apply_gainmap_fp32`` — gain-map decode/apply kernel (~9×)
* ``compute_sdr_base_u8`` — HDR → SDR uint8 conversion (~5×)
* ``upscale_gainmap`` — new public function, nearest-neighbor upscale
  for ``gain_scale > 1`` decode paths (~8× faster than ``np.repeat``)

Combined wall-clock for a 2000×2000 fp32 input on a 20-core machine:

* ``encode_native``: 151 ms → 43 ms (3.5× over libuhdr direct,
  matches JXL HLG end-to-end at ~128 ms when the JXL OETF
  preprocessing cost is included)
* ``decode_native``: 87 ms → 29 ms
* Output: 5.02 MB → 4.24 MB (single-channel gain map; the libuhdr
  direct encoder emits a multi-channel gain by default)

The load-bearing implementation detail: each worker function is a
regular ``def`` that opens a ``with nogil:`` block internally before
calling the ``cdef`` kernel. A ``cpdef ... noexcept nogil`` worker
holds the GIL on Python entry for argument parsing and yields 0×
speedup. Fixing that pattern unlocked the entire kernel-parallelism
win.

**Chrome UHDR compact-metadata-form auto-fix**

libuhdr's gain-map metadata serializer emits a 37-byte "compact"
form with the ``useCommonDenominator`` flag set whenever all
per-channel rationals share the same denominator (the common case
when ``max_content_boost`` is a power of 2 and all other fields are
0 or 1). The form is ISO 21496-1 spec-legal but Chrome's UHDR
parser doesn't handle it — files render as plain SDR JPEGs in
Chrome with no HDR boost.

``encode_native`` now auto-detects when ``log2(max_content_boost)``
lands on a clean integer (within 1e-9) and nudges the value by 1
part in 10^7 before passing to libuhdr. libuhdr's float → rational
continued-fraction converter then produces a non-trivial
denominator; the common-denominator detection fails; the canonical
61-byte long form gets emitted — same byte structure as libuhdr's
own direct ``encode`` output. Numerically invisible; existing
callers don't need to change anything.

The auto-detect runs in both the auto-mcb path (``max_content_boost
=None``) and the explicit-mcb path, so any caller that historically
passed a clean integer (``mcb=8.0``, ``mcb=4.0``) is now
automatically Chrome-safe.

**``gain_scale`` caveat for sharp + noisy content**

Stride decimation of the gain map (``gain_scale > 1``) can produce
visible block patches at decode for content with high-spatial-
frequency variation in dim regions (background noise + sharp
HDR-bright detail nearby): adjacent decimation chunks pick
different noise samples for the gain value, and the nearest-
neighbor upscale at decode time amplifies the discontinuity into
visible blocks. For natural / photographic content libuhdr's CLI
default ``gain_scale=4`` remains fine; for content with sharp
bright detail on noisy backgrounds, ``gain_scale=1`` is required.

**JxlWriter: expose previously-hidden libjxl knobs**

Pass-through plumbing for five fields that the Cython binding wasn't
forwarding to libjxl's ``JxlEncoderFrameSettingId`` /
``JxlBasicInfo``:

* ``color_transform`` — ``'xyb'`` (default), ``'none'`` /
  ``'rgb'``, ``'ycbcr'``, or integer 0/1/2
* ``modular`` — force VarDCT (``False``) or modular (``True``);
  default is libjxl's automatic decision
* ``min_nits`` — black-point luminance hint in the basic info
* ``relative_to_max_display`` — reinterpret ``intensity_target`` as
  a fraction of display peak rather than absolute nits
* ``linear_below`` — signal value below which the encoded transfer
  is linear

All accept ``None`` to mean "leave libjxl default", so existing
callers see no behavior change.

0.1.12 (2026-06-02)
-------------------

**Embedded EXIF thumbnail is now Ultra-HDR by default**

The thumbnail introduced in v0.1.11 was a plain SDR JPEG, which on an
HDR display renders at SDR-white (~200 nits) — a visible brightness
gap against the main image's full HDR boost (e.g. ~1600 nits at
``sdr_white_nits=1600``). For sparse-bright content (fluorescence
dye spots, specular highlights) the brightness gap can be ~10× and is
the dominant artifact when previewing.

``encode_native(thumbnail_size=N)`` now embeds a mini Ultra-HDR
(SDR base + downsampled gain map + MPF) by default, so HDR-aware
viewers preserve peak brightness. SDR-only EXIF readers see the SDR
base layer as a plain JPEG and ignore the MPF block, so backward
compatibility is preserved.

* New kwarg ``thumbnail_hdr=True`` (default) — set to ``False`` to
  fall back to the v0.1.11 plain-SDR-thumbnail behavior. Useful when
  targeting legacy EXIF readers that don't tolerate MPF segments in
  thumbnail bytes.
* Both the SDR base and gain map are decimated by **centered**
  integer stride (offset = stride // 2) so the thumbnail's
  coordinate origin lines up with the full-res image at scale
  ``1/stride``. The previous top-left stride introduced a half-stride
  pixel bias.
* The same centered-stride fix is applied to the main image's
  gain-map downscale (``gain_scale=2/4/...``).
* Thumbnail-gain JPEG uses the full-resolution computed gain map
  (pre-``gain_scale``) so the thumbnail's peak fidelity isn't
  compounded by the main image's gain-scale loss.

Read side adds ``opencodecs.uhdr.read_thumbnail_hdr(data)`` —
returns the thumbnail as a fp32 HDR ndarray. For UHDR thumbnails this
routes through :func:`decode_native` (peak HDR pixels). For plain-SDR
thumbnails (legacy or ``thumbnail_hdr=False``) it returns
``sdr_u8 / 255``.

Size cost: ~30 KB per file vs the v0.1.11 SDR thumbnail (e.g. 53 KB
vs 23 KB at ``thumbnail_size=250, thumbnail_quality=85``). The
read-side decode time is unchanged (~1 ms).

5 new tests in ``tests/test_uhdr.py``:
``test_thumbnail_is_uhdr_by_default``,
``test_thumbnail_hdr_false_emits_plain_sdr_jpeg``,
``test_read_thumbnail_hdr_preserves_main_peak``,
``test_read_thumbnail_hdr_fallback_on_sdr_thumb``,
``test_read_thumbnail_hdr_none_when_no_thumbnail``.

0.1.11 (2026-06-02)
-------------------

**Embedded EXIF thumbnail on Ultra-HDR (opt-in)**

``encode_native`` and ``encode_to`` gained two kwargs:

* ``thumbnail_size`` — if set, embeds a square thumbnail of up to
  ``thumbnail_size`` px on each side inside an APP1 EXIF segment of
  the same file (no sidecar). ``None`` (default) embeds nothing.
* ``thumbnail_quality`` — JPEG quality for the thumbnail layer
  (default 80).

Read side:

* ``opencodecs.uhdr.read_thumbnail_bytes(data)`` returns the
  embedded thumbnail as raw JPEG bytes (or ``None``). Sub-millisecond
  — parses only the APP1 segment, never touches the SDR base or
  gain-map JPEGs.
* ``opencodecs.uhdr.read_thumbnail(data)`` decodes the same to a
  ``(h, w, 3) uint8`` ndarray.

The thumbnail is built from the (peak-normalised) SDR base raster
via integer stride decimation, then encoded as a small independent
JPEG and stuffed into a standard EXIF "1st IFD" / APP1 marker
prepended to the main file. Every existing photo viewer that
honours EXIF thumbnails (file managers, Finder, phone galleries,
OS thumbnail caches) picks it up automatically — no opencodecs
required on the read side.

Bench on a 2k² Ultra-HDR file (M-series Mac):

* ``read_thumbnail_bytes`` (just the byte-slice):  ~4 μs
* ``read_thumbnail`` (slice + decode 250×250):    ~0.2 ms
* ``decode_native(scale=1/8)`` (full entropy decode at 1/8 IDCT): ~25 ms
* ``decode_native`` (full):                       ~54 ms

That's a **~250× speedup over full decode** for archive-browsing
workflows. The win compounds on cloud-stored archives: an HTTP
range request for the first ~64 KB of a file gets you the
thumbnail without transferring the full multi-MB payload.

File-size overhead at default settings (``thumbnail_size=256``,
``thumbnail_quality=80``): ~15-25 KB per file.

Preserves Ultra-HDR conformance: ``is_uhdr``, ``probe``, libuhdr's
own ``decode``, and our ``decode_native`` all unchanged on the
thumbnail-augmented file. The APP1 EXIF segment is standard JPEG
metadata; it doesn't interact with the MPF / XMP gain-map blocks.

0.1.10 (2026-06-02)
-------------------

**HDR-fidelity bug fix — every prior ``encode_native`` output was
under-boosting HDR by roughly 50%**

The polynomial approximation in ``_fast_log2`` (used by
``_gain_map_kernel`` to encode the gain map, and by ``_fast_pow``
in ``_apply_gainmap_kernel`` for the gamma-non-1 decode case) had
systematically wrong coefficients. The polynomial was exact at
powers of 2 (m=1 exactly) but dipped up to **−1.15** in log₂ units
mid-octave — e.g. ``_fast_log2(7.5)`` returned 1.754 instead of
2.907.

Effect on encoded Ultra-HDR files: the per-pixel gain values were
quantised against a too-shallow log₂ curve, so a pixel that should
have written ``gain_u8 = 255`` (full boost) instead wrote
``gain_u8 ≈ 139``. Decoded HDR brightness landed at ~43% of
intended. macOS Preview / Chrome / libuhdr all faithfully rendered
the buggy gain map → every Ultra-HDR ``encode_native`` ever wrote
was effectively a "half-strength HDR" file.

**Fix**: replace ``_fast_log2`` → ``libc.math.log2f`` in
``_gain_map_kernel`` and ``_fast_pow`` → ``libc.math.powf`` in
``_apply_gainmap_kernel``. Costs ~3-5 ns/pixel of extra encode +
decode time (~12 ms on a 2k² raster) — well worth the fidelity.
``_fast_exp2`` was verified correct and is retained.

Verified round-trip on a 2k² fluorescence tilescan:

* input HDR peak = 1.0000, mean = 0.0530
* ``decode_native(display_boost=1)``: peak = 1.000, mean = 0.0523
* ``decode_native`` (full HDR, default): peak = 7.841 — matches
  the encoded ``hdr_capacity_max`` = 7.88 exactly
* cross-check vs libuhdr's own ``decode``: peak = 1.0000, mean =
  0.0526 (both implementations now agree)

If you have v0.1.5-v0.1.9 ``.jpg`` files in production: they're
still valid Ultra-HDR JPEGs (libuhdr / Apple / Chrome decode them
without error), but they carry roughly half the HDR signal the
encoder intended to store. Re-encoding from source produces files
with the full boost.

The earlier earlier "chroma subsampling 4:2:0 → 7.84 → 3.67 peak
attenuation" measurement in v0.1.9's notes was a *symptom* of this
log₂ bug, not the cause. With this fix the chroma-subsampling
choice (4:2:0 vs 4:4:4 vs 4:4:0) becomes the smaller signal it
should be.

0.1.9 (2026-06-02)
------------------

**Lossless SDR base layer on encode (opt-in)**

``opencodecs.codecs._jpeg.encode`` gained a ``lossless=False`` kwarg
that switches libjpeg-turbo into predictive lossless mode
(``TJPARAM_LOSSLESS=1``, PSV=1, forced 4:4:4 chroma). Output is
bit-exact through the libjpeg-turbo decoder, at the cost of ~3-5×
larger files vs ``level=95`` baseline DCT on natural-image content.

``opencodecs.uhdr.encode_native(lossless=True)`` routes the SDR base
layer through that lossless path while keeping the gain map lossy
(there's no perceptual benefit to making a band-limited gain map
lossless, and it would balloon the file).

Empirical caveats — verified on macOS 15 with an HDR-capable
display:

* macOS Preview, Chrome 116+, and any viewer that parses the MPF /
  XMP gain-map block directly **HDR-render the output normally**.
* libuhdr's reference ``uhdr_decode`` (and therefore
  ``opencodecs.uhdr.decode``) **rejects** the file. It enforces a
  strict ``JCS_YCbCr`` / ``JCS_GRAYSCALE`` colorspace check that
  lossless-mode JPEG (which writes ``JCS_RGB``) fails.
* Apple's ``CGImageSource`` ``Headroom`` property and
  ``kCGImageAuxiliaryDataTypeISOGainMap`` aux-data accessors miss
  the HDR signal (same root cause), even though Preview's actual
  render path composites the gain map correctly.
* ``opencodecs.uhdr.decode_native`` reads it cleanly — it uses our
  own gain-application kernel without libuhdr's colorspace gate.

Use only when exact SDR-base-pixel preservation is a hard
requirement (archival, scientific imaging) and every consumer in
your pipeline is either Preview / Chrome / ``decode_native``. The
``v0.1.8`` note that called lossless "not a conforming Ultra-HDR
file" was too strong — it's non-conforming to libuhdr's reference
decoder and Apple's narrow ImageIO API, but visual HDR consumers
accept it just fine.

0.1.8 (2026-06-02)
------------------

**DCT-domain decode-time scale on JPEG + decode_native**

``opencodecs.codecs._jpeg.decode`` now exposes libjpeg-turbo's
``tj3SetScalingFactor`` knob via a new ``scale=`` kwarg (plus
explicit ``scale_num`` / ``scale_denom`` for advanced callers). The
decoder skips the inverse-DCT work for high-frequency coefficients
when the caller requests a downsampled output, so a 1/8 decode runs
~2× faster than a full decode and produces a 64× smaller raster.

Supported factors are libjpeg-turbo's full set of 16 ratios N/8 for
N ∈ {1, 2, …, 16} — i.e. anywhere between 1/8 and 2/1. The new
``opencodecs.codecs._jpeg.supported_scaling_factors()`` returns them
as ``(num, denom)`` pairs. ``scale=`` accepts:

* an integer N (interpreted as ``1/N``)
* a float (snapped to the closest supported ratio)
* a tuple ``(num, denom)`` verbatim

``opencodecs.uhdr.decode_native(..., scale=...)`` threads the same
knob through both the SDR base and the gain-map JPEG decodes. The
gain-application kernel runs on the smaller rasters, so the whole
pipeline scales — a 2k² Ultra-HDR decode at ``scale=8`` returns a
``(250, 250, 3)`` fp16 raster in ~25 ms vs ~53 ms for the full
decode. Useful for thumbnail / preview pipelines that want real
HDR pixels off cloud-stored Ultra-HDR without paying the megapixel
JPEG decode cost. Lossless source not required — the JPEG file is
unchanged on disk; the savings come from skipping IDCT work.

(libuhdr lossless support is unchanged: ISO 21496-1's SDR base layer
is required to be baseline DCT JPEG, so lossless-mode JPEG would
not be a conforming Ultra-HDR file.)

0.1.7 (2026-06-02)
------------------

**Patched libultrahdr bundle**

Since v0.1.6 we build libuhdr ourselves; v0.1.7 starts carrying
local patches via ``patches/libultrahdr/*.patch``, applied by
``bench/build_codec_libs.sh::build_libultrahdr`` after the source
fetch. Five upstream post-v1.4.0 cherry-picks land first:

* ``5ed39d6`` — fix ``CLIP3`` parameter order in libuhdr's own
  ``applyGainMap``. The bug clamped a constant ``0.0f`` instead of
  the gainmap weight whenever ``display_boost ≠ hdr_capacity_max``,
  silently producing wrong HDR output on libuhdr's wrapped
  ``decode()`` path. ``opencodecs.uhdr.decode_native`` is unaffected
  (it uses our own Cython gain-application kernel).
* ``7088ca7`` — error-message typo in ``jpegr.cpp``.
* ``13a058f`` — ``icc.h`` Endian_Swap macros now respect actual
  host endianness instead of an unconditionally-true ``USE_BIG_ENDIAN_IN_ICC``.
  No effect on x86_64/aarch64 (little-endian); fixes PowerPC/s390x.
* ``5fa99b5`` — add missing ``<cstdint>`` include for GCC 15.
* ``8cbc983`` — same ``CLIP3`` fix as ``5ed39d6`` in the GPU path
  (we don't link the GPU path; carried for hygiene).

**New API: ``opencodecs.uhdr.probe(data) -> dict``**

Parse an Ultra-HDR container's MPF metadata without any pixel
decode. Returns base + gainmap dimensions plus the gainmap
metadata block (``max_content_boost`` / ``min_content_boost`` /
``gamma`` / capacity). Wraps libuhdr's existing ``uhdr_dec_probe``
+ accessors. ~180× faster than ``decode()`` for any HDR-aware
flow that only needs dimensions or capacity — image indexing,
thumbnail generation, HTTP HEAD-style inspection, routing batches
by content-boost.

**Gain-map tunables on ``encode_native``: ``gain_quality`` + ``gain_scale``**

Two new kwargs let callers shrink the gain-map layer
independently of the SDR base:

* ``gain_quality`` — JPEG quality for the gain-map layer only
  (defaults to track ``quality``). The gain map is heavily
  band-limited so ``gain_quality=70`` is visually equivalent to
  ``q95`` and cuts ~30% off the gain-map bytes.
* ``gain_scale`` — integer power-of-2 downsample factor for the
  gain-map raster (``1`` keeps full resolution, ``2`` halves both
  axes for quarter-area, etc.). Stride decimation; ~5 ms on a 2k²
  uint8 gain map. Round-trips through ``probe`` / ``decode_native``
  cleanly — the container records the actual gain-map dimensions
  and the decoder upscales on apply.

**True streaming encode: ``encode_native(..., out=fp)``**

The old ``encode_to(fp, hdr, ...)`` was a forward-compatible
``fp.write(encode_native(hdr, **kw))`` wrapper. v0.1.7 makes both
genuinely streaming: ``encode_assembled`` (and by extension
``encode_native`` / ``encode_to``) now accept an ``out=`` file-
like, and when given they hand a zero-copy ``memoryview`` over
libuhdr's internal output buffer to ``out.write()``. Skips the
final ``PyBytes_FromStringAndSize`` allocation + memcpy. Saves
~1× output-size peak memory and ~3 ms wall-clock on a 5 MB encode.
``encode_to`` now returns ``None`` instead of the byte count.

0.1.6 (2026-06-01)
------------------

**libultrahdr now bundled (fixes missing ``_uhdr`` on PyPI wheels)**

* ``bench/build_codec_libs.sh::build_libultrahdr`` builds libuhdr
  v1.4.0 (Apache-2.0) into ``$OPENCODECS_LIBS_PREFIX/{include,lib}``
  alongside the other source-built codec libs. The Linux + Windows
  CI ``--only=...`` lists pick it up; the macOS ``before-all`` brew
  install now includes ``libultrahdr`` as well.
* setup.py's ``_uhdr`` Extension is now built by a probe
  (``_maybe_build_uhdr_ext``) that finds the cached prefix, sets
  ``library_dirs``, and bakes the rpath into the resulting ``.so``
  so delocate / auditwheel / delvewheel can bundle ``libuhdr``'s
  dylib/.so/.dll into the wheel.
* ``_uhdr`` is now in ``MUST_SHIP_ALL_PLATFORMS`` in
  ``ci/check_wheel_contents.py``; a wheel without the Ultra-HDR
  extension fails the wheel-coverage step.
* The stale ``_ultrahdr.pyx`` source file (orphaned in commit
  ``c9347f5`` when the direct libuhdr binding shipped as ``_uhdr``)
  is removed.

**New API: ``opencodecs.uhdr.decode_native``**

Fused-Cython fast-path Ultra-HDR decoder. Uses libuhdr's parser to
pull out the compressed SDR base + gain-map JPEGs + metadata (no
pixel decode), then decodes both JPEGs in parallel via
``imagecodecs.jpeg_decode`` (libjpeg-turbo SIMD, GIL released) and
applies the gain map in a Cython kernel that uses the same sRGB
EOTF LUT + IEEE-754 polynomial ``exp2`` the encoder uses. ~1.4×
faster than libuhdr's reference decode on a 2k² float HDR (M-series
Mac: ~52 ms vs ~72 ms). Output matches libuhdr's decode to within
JPEG-q95 + 8-bit gain-quantisation noise. ``display_boost`` kwarg
exposes the ISO 21496-1 headroom scaler — default is full HDR
(``hdr_capacity_max``), pass ``1.0`` to match libuhdr's default
SDR-equivalent decode.

Backed by two new Cython helpers exposed for advanced callers:

* ``opencodecs.codecs._uhdr.extract_layers(data)`` — parse the
  container, return ``{base_jpeg, gainmap_jpeg, gainmap_metadata,
  width, height, gainmap_{width,height}}`` without decoding pixels.
* ``opencodecs.codecs._uhdr.apply_gainmap_fp32(sdr_u8, gain_u8,
  metadata, display_boost=...)`` — the per-pixel gain-application
  kernel.

**New API: ``opencodecs.uhdr.encode_to``**

Streaming variant of :func:`encode_native` — writes Ultra-HDR bytes
directly to a file-like (anything with ``write(bytes)``: open file,
``io.BytesIO``, HTTP upload streamer). Returns the byte count.
Forward-compatible alias: the libuhdr api-4 path we currently use
for container assembly doesn't expose a streaming writer, so the
function is ``fp.write(encode_native(...))`` today; the API exists
so callers can adopt it now and pick up any future libuhdr
streaming write-out without changing their code.

**CI / build robustness**

* ``_uhdr`` extension's ``-ffast-math`` is now narrowed to
  ``-ffast-math -fno-finite-math-only
  -fno-unsafe-math-optimizations -fno-math-errno
  -fno-trapping-math``. The ``-funsafe-math-optimizations``
  sub-flag is what tells GCC to replace libm calls with libmvec's
  vectorised ``_ZGV*`` variants, which link-fail on Ubuntu (Ubuntu's
  default gcc doesn't auto-link libmvec like manylinux_2_28 does)
  and aren't available at all on aarch64. Same flag set the edt
  extension uses for the same reason. Keeps FMA + reordering
  perf; eliminates the libmvec dependency entirely.
* The libaec source URL moved from ``gitlab.dkrz.de`` (now
  auth-gated; anonymous requests redirect to ``/users/sign_in``)
  to DKRZ's GitHub mirror at
  ``github.com/Deutsches-Klimarechenzentrum/libaec``.
* ``fetch_tar`` now has a 4-attempt shell retry loop with empty-
  extract detection — covers transient mirror outages that
  curl's own ``--retry`` doesn't see (e.g. 200 OK with a
  truncated body).

0.1.5 (2026-05-31)
------------------

**New codecs**

* ``opencodecs.uhdr`` — Ultra-HDR / ISO 21496-1 (gainmap JPEG) via a
  direct Cython binding to Google's libultrahdr. ``encode(hdr, ...)``
  wraps libuhdr's full pipeline; ``encode_native(hdr, sdr=None, ...)``
  is a fused-Cython fast path that computes SDR base + gain map in
  ``nogil`` kernels (IEEE-754 polynomial log2/pow, cross-platform —
  no Apple Accelerate intrinsics), JPEG-encodes both layers in
  parallel via a 3-worker ThreadPoolExecutor, then hands the
  pre-encoded JPEGs to libuhdr's api-4 for container assembly.
  Measured 31 ms median for a 2000² float HDR on M-series Mac vs
  ~173 ms for the libuhdr reference path — 5.5× faster. The
  ``sdr=`` argument lets callers (notably tilescan pipelines) supply
  their own SDR base instead of accepting the peak-normalised default.

* ``PLIO_1`` — IRAF run-length mask coding closes the last FITS
  tile-compression gap. Vendors cfitsio's ``pliocomp.c`` (Doug Tody /
  NRAO, public-domain) and adds the tiny
  ``opencodecs.codecs._plio.decode_raw`` shim. Round-trip tested
  against astropy. The ``COMPRESSED_DATA`` BINTABLE column walker
  now tracks per-column element byte width so the ``1PI`` (int16
  opcodes) layout PLIO uses works alongside the ``1PB`` byte VLA the
  other compressors use.

**New API**

* ``EerReader.sum(start, stop, *, weights=None, dtype=...)`` —
  per-frame dose curve. ``weights[k]`` multiplies the k-th frame in
  the requested range; output promotes to float64 when weights are
  given so fractional contributions don't truncate. Use case:
  beam-induced-motion correction and temporal-binning schemes that
  emphasise different exposures across the acquisition.

**Bug fixes**

* HEIF encode against libheif ≥ 1.18 — newer libheif calls
  ``strlen()`` on the user write-callback's ``heif_error.message``
  even on success and rejects a ``NULL`` pointer with *"heif_writer
  callback returned a null error text"*. The callback now hands back
  a static empty string on success and a descriptive string on OOM.
  Six tests (``test_heif_*``, ``test_phase5_icc.py[heif]``) go back
  to green across every CI matrix entry.

**CI / build**

* ``bench/build_codec_libs.sh`` — the ``is_built`` cache-marker now
  records the install dir alongside the version and re-verifies the
  install dir exists before short-circuiting a recipe. Without this,
  cibuildwheel's manylinux cache (which only covers ``/cibw-jxl-prefix``)
  preserved the marker for recipes that install to
  ``~/.cache/opencodecs/<lib>/`` outside the cached prefix —
  ``lerc``, ``mozjpeg``, ``brotli``, ``zstd``, ``giflib`` — so the
  recipe thought it was done while the actual library files were
  gone. ``_lerc`` + ``_mozjpeg`` had been silently dropped from
  Linux wheels for this reason; the fix restores them.

* ``test_omezarr.py`` skips at module level on zarr-python < 3 —
  fixtures use the v3 API surface, which doesn't exist on the v2
  branch pip resolves on Python 3.10.

* ``test_eer_imagecodecs_cross_validate`` is now symmetric on
  decoder error: if either implementation raises, both must raise
  for the combo to count. Tolerates imagecodecs ≥ 2026.5's tighter
  output-buffer sizing.

0.1.4 (2026-05-25)
------------------

**libvips-inspired streaming improvements**

* ``HTTPDataSource(access='sequential', sequential_chunk_bytes=4*1024*1024)``
  — opt-in libvips-style sequential read mode. Replaces the LRU +
  adaptive read-ahead with a single rolling buffer that slides forward
  as the caller reads. Memory stays bounded to one chunk regardless of
  file size. Target workload: tile-by-tile raster scan over a huge
  COG / OME-TIFF served over HTTP. Backward seeks still work but
  invalidate the buffer; ``stats['sequential_backward_seeks']`` tracks
  the count so users can spot a workload that's actually random and
  would benefit from ``access='random'``.

* ``TiffPyramidReader.read_region`` now parallelises tile decode
  across a thread pool (default ``min(cpu_count(), 8)``). The
  compressed-tile decoders (JPEG, JPEG-2000, deflate, zstd, LZW, WebP,
  LERC) all release the GIL in their C path, so threads scale across
  cores on CPython. Kicks in only when the region covers 4+ tiles —
  fewer than that and thread-pool spin-up cost dominates. Pass
  ``num_decode_workers=1`` to force the serial path for benchmarks /
  regression-diff.

* ``TiffWriter.write_pyramid_auto`` gained ``stream_levels=True``
  (default for COG layout). Computes and writes one level at a time
  via a new ``iter_pyramid_levels`` generator, dropping each finished
  level before computing the next. Peak memory drops from
  ``~1.33 × level0`` (entire geometric series materialised) to
  ``~2 × current_level``. Useful when level 0 already dominates RAM
  (whole-slide pathology, cryo-EM tomograms). Output file is
  byte-identical to the materialize-all path. SubIFD layout
  (``subifds=True``) still uses the materialize path because that
  layout writes sub-resolution IFD offsets into the main IFD's tag
  330 — sub-level offsets have to be known up front.

  Validated via tests:
  ``test_http_sequential_serves_from_rolling_buffer``,
  ``test_http_sequential_backward_seek_counted``,
  ``test_pyramid_reader_parallel_decode_matches_serial``,
  ``test_pyramid_auto_stream_matches_materialize``,
  ``test_iter_pyramid_levels_matches_make_pyramid_levels``.

**MozJPEG ships on every wheel**

* Added MozJPEG (Mozilla's libjpeg-turbo fork) to the cibuildwheel
  codec-lib build set on Linux, macOS, and Windows. MozJPEG produces
  JPEG files ~10-15% smaller than baseline libjpeg-turbo at the same
  quality setting (progressive encoding + trellis quantization +
  better quantization tables). The ``_mozjpeg`` extension has shipped
  on macOS wheels for a while via brew; v0.1.4 brings it to
  Linux + Windows so the codec is available everywhere.
* New ``build_mozjpeg`` recipe in ``bench/build_codec_libs.sh``
  installs into a keg-style ``mozjpeg/`` subdir under the prefix
  (rather than the shared ``$PREFIX/{include,lib}``) to avoid the
  libturbojpeg / libjpeg name collision with the libjpeg-turbo 3.x
  install. setup.py's mozjpeg probe was extended to recognise the
  new candidate paths on Linux (``~/.cache/opencodecs/mozjpeg``),
  Windows (``$CONDA_PREFIX/Library/mozjpeg``), and the Windows
  fallback (no ``nm``: trust the keg-style directory name).
* Locally validated on a Windows VM (MSVC 14.44 — same as CI) and
  a Linux x86_64 host: ``bash bench/build_codec_libs.sh
  --only=mozjpeg`` followed by ``setup.py build_ext --inplace``
  produces ``_mozjpeg.{pyd,so}`` cleanly. This iteration was validated
  end-to-end through the same script CI runs, with the same env vars,
  before any push.

**Post-publish wheel coverage check (CI hardening)**

* New ``ci/check_wheel_contents.py`` runs in the wheel-build job
  immediately after ``cibuildwheel`` produces each wheel. It unzips
  the wheel, walks ``opencodecs/codecs/``, and asserts every codec
  in the ``MUST_SHIP_ALL_PLATFORMS`` set has a ``.pyd``/``.so``
  present. Missing codec → matrix job fails → publish step gated.
* Catches the v0.1.2-style "changelog overclaim" silent-drop class
  of bugs at build time instead of after publish. v0.1.2's Windows
  wheels shipped without ``_sperr``/``_brunsli`` despite the
  changelog claim; this check would have failed that build.


0.1.3 (2026-05-22)
------------------

**SPERR + brunsli land on every wheel**

* Added ``SPERR`` and ``brunsli`` to the cibuildwheel codec-lib
  build recipe for Linux, macOS, and Windows. Previously the
  ``build_codec_libs.sh --only=...`` selection for cibuildwheel
  ran only SZ3 + pcodec, so the SPERR / brunsli ``cmake_build``
  paths (already in the script) never fired on CI — setup.py's
  header probe dropped ``_sperr`` and ``_brunsli`` from every
  wheel released through v0.1.2.
* brunsli vendors its own brotli submodule via CMake, so no
  additional system-brotli dependency at link time. Both libs
  install into the per-user opencodecs cache and get bundled
  into the wheel via auditwheel / delocate / delvewheel.
* v0.1.3 ships ``_sperr`` (NCAR wavelet error-bounded compressor)
  and ``_brunsli`` (Google lossless JPEG transcoder, ~22% smaller
  storage) on all four wheel cells (Linux x86_64 + aarch64,
  macOS arm64, Windows AMD64) × Python 3.10-3.13.

**Errata for v0.1.2**

* The v0.1.2 CHANGES claimed "Restored ``_sz3``, ``_pcodec``,
  ``_sperr``, ``_brunsli`` on Windows." In reality only ``_sz3``
  and ``_pcodec`` were restored — the SPERR / brunsli libs never
  got into the ``--only=`` build selection, so their extensions
  silently dropped via the missing-header probe. v0.1.3 closes
  the gap.


0.1.2 (2026-05-22)
------------------

**Windows wheels get _sz3 and _pcodec back**

* Restored ``_sz3`` and ``_pcodec`` on Windows. Root cause was
  conda's bash putting ``gcc.exe`` ahead of ``cl.exe`` on PATH;
  CMake then produced gnu-format ``libSZ3c.dll.a`` import
  libraries that cibuildwheel's MSVC link.exe couldn't consume.
* Workflow now uses ``ilammy/msvc-dev-cmd`` to source vcvars64.bat
  before the SZ3+pcodec source-build step; CMake's auto-detect picks
  cl.exe and produces MSVC-format ``SZ3c.lib`` / ``cpcodec.lib``.
* Cargo's MSVC linker pinned via
  ``CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER`` so rustc doesn't
  PATH-resolve to GNU coreutils' ``link.exe`` at
  ``C:\Program Files\Git\usr\bin\link.exe``.
* Validated end-to-end on a Windows 11 VM (clean SZ3 install with
  Ninja + cl.exe + vcvars-sourced env).

**README rewrite for the released project**

* ``pip install opencodecs`` + PyPI badge at the top.
* New "Why opencodecs" table mapping common scientific-imaging needs
  to concrete shipping capabilities.
* New "Streaming-reader examples" section with 3 copy-paste recipes:
  HTTP region-fetch from a remote Aperio TIFF, TIFF → OME-Zarr v3
  sharded conversion, and the native progressive JXL thumbnail path.
* Status / Install sections updated for the 0.1.x cadence.


0.1.1 (2026-05-21)
------------------

Supersedes 0.1.0 (yanked). Same codec coverage as 0.1.0 plus the
work since, with source-comment metadata scrubbed.

**CMS — sRGB ↔ Display-P3 fast converter**

* Add ``opencodecs._cms_codec.srgb_to_display_p3_uint8(arr)``
  convenience for the gallery / Jupyter / web-display case: convert
  sRGB-encoded uint8 RGB(A) → Display-P3-encoded uint8 in ~28 ms
  for a 2Kx2K image on macOS arm64 (vs ~110 ms for an equivalent
  numpy LUT + matmul pipeline).
* Add ``_builtin_profile_icc(name)`` returning ICC bytes for the
  built-in profiles ``"srgb"`` and ``"display-p3"``. Display-P3 is
  synthesized via ``cmsBuildParametricToneCurve`` +
  ``cmsCreateRGBProfile`` rather than depending on lcms2 ≥2.16's
  ``cmsCreate_DisplayP3`` — works on older liblcms2 too.
* Lcms2 ``COPY_ALPHA`` flag doesn't combine with manually-built
  RGB-only profiles, so the RGBA path transforms RGB in a
  contiguous temporary and stitches the alpha channel back.
* 9 new tests under tests/test_phase7_cms.py cover canonical
  primary-color transforms, gray invariance, alpha preservation,
  and input validation.

**OME-Zarr v3 sharded write**

* Add ``shards=`` kwarg to ``opencodecs.write_zarr_array``,
  ``write_omezarr_pyramid``, and ``write_omezarr_pyramid_auto``.
  Enables the Zarr v3 ``sharding_indexed`` codec: each shard file
  on disk packs ``prod(shards / chunks)`` inner sub-chunks plus
  a trailing ``uint64`` ``(offset, nbytes)`` index. The reader
  (which already supported sharded *reads* with HTTP-range
  fetches) now has a writer counterpart, closing the OME-Zarr
  v3 write story.
* Validation: ``shards=`` requires ``zarr_format=3`` and each
  shard axis must be a multiple of the corresponding chunk axis
  (so each shard holds a whole number of inner chunks).
* Pyramid writers auto-adapt ``shards`` per level — when a
  downsampled level is smaller than the requested shard shape,
  the shard clamps to the largest multiple of ``chunks`` that
  fits, and falls back to per-chunk layout when a level is too
  small to hold even one chunk.
* Verified pixel-equal round-trip via the reference zarr-python
  reader (including the edge-of-array case where the array
  dimensions aren't a multiple of the shard shape — those slots
  use the standard Zarr empty-chunk sentinel of ``2**64 - 1``).
* Index uses bytes-only encoding (no CRC32C yet — would add a
  ``crc32c`` runtime dep and the reader already handles its
  absence gracefully).

**JXL ``subsample`` kwarg for downsample positioning**

* Add ``subsample={'top-left', 'center'}`` kwarg to
  ``opencodecs.jxl.read`` / ``open`` / ``iter_frames`` and the
  underlying ``JxlReader`` / ``decode``.
* ``'top-left'`` (default) keeps the historical
  ``arr[::N, ::N]`` semantic — back-compat with imagecodecs and
  every existing caller.
* ``'center'`` takes ``arr[N//2::N, N//2::N]`` so each output
  pixel represents the geometric centroid of its source NxN
  block. The right choice for visual thumbnails: when the
  downsampled raster gets drawn in an NxN region of an output
  canvas (SVG ``<image>``, GL texture upload, …), centroid
  semantics keep the thumb positionally self-consistent with
  the full-res source. Top-left semantics shift features by
  ~(N-1)/2 source-pixels.
* Output shape is ``ceil(src/N)`` per axis in both modes.
  When source dimensions aren't divisible by N, the centered
  slice would otherwise come up one row/col short; we replicate
  the bottom/right edge to preserve the shape contract.
* No perf change — both modes are pure index/copy on the
  already-decoded buffer. The decode path is unchanged.

**Tier 3 streaming-reader plumbing**

* Add ``HTTPDataSource`` covering-cache lookup: when ``read_at(off, n)``
  misses the exact ``(off, n)`` LRU key, scan for any cached blob that
  fully covers the requested range and slice it out. Compounds for
  free — any time a reader fetches a big region, later small reads
  inside it don't round-trip.
* Add adaptive read-ahead trigger. When 3 consecutive small cache
  misses fall within ``adaptive_locality`` (64 KB) of one another —
  the signature of a FITS HDU walk, h5py B-tree traversal, or TIFF
  IFD chain — the next miss gets bumped to ``adaptive_window``
  (64 KB) so subsequent adjacent reads serve from cache. Scattered
  reads keep the streak at 1 and never trigger.
* ``HTTPDataSource`` now returns ``b""`` (not a raised
  ``HTTPException``) for ``Range`` requests past end-of-file. Matches
  file-like ``read()`` behavior and lets the FITS HDU walker probe
  the next-HDU offset without a try/except dance.
* Add ``read_hdf5_slice(dataset, sel)`` convenience wrapper that
  bundles ``prefetch_hdf5_chunks`` + the actual read into one call.
  Falls through cleanly when the dataset isn't backed by
  ``open_remote_hdf5`` (local file path).

**FITS reader**

* Native FITS reader (``opencodecs._fits.FitsStream``). Multi-HDU
  parsing, BITPIX 8/16/32/64/-32/-64, BSCALE/BZERO with the FITS
  unsigned-int convention (BZERO=2**(N-1) → uintN). HTTP-range
  friendly: one Range request per HDU header at open time, image
  data fetched lazily on ``asarray()``.
* Compressed-image decode for BINTABLE+ZIMAGE HDUs. Supported
  ZCMPTYPE: RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1, NOCOMPRESS.
  Per-tile ZSCALE / ZZERO quantization for floats. Fall-back path:
  when the primary COMPRESSED_DATA descriptor is empty, decode from
  GZIP_COMPRESSED_DATA (lossless gzipped original bytes) instead.
  HCOMPRESS_1 uses cfitsio's ``fits_hdecompress`` vendored under
  ``3rdparty/cfitsio/`` (BSD-style license).
* astropy.io.fits is the cross-validation oracle: every supported
  ZCMPTYPE × dtype combination decodes pixel-equal.

**Ultra HDR / Radiance HDR**

* Add ``rgbe`` codec (Radiance ``.hdr``) — Cython binding to the
  vendored Bruce Walter / Greg Ward C library. RLE-compressed by
  default; cross-validated against ``imagecodecs.rgbe_encode/decode``.
* Add ``ultrahdr`` codec (ISO 21496 gainmap JPEG) — Cython binding
  to Google's libultrahdr 1.4.0. Default encode: ``(H, W, 4) float16``
  linear BT.2100 RGBA at quality 95; decode returns the same. Tested
  against imagecodecs's ultrahdr binding for pixel-equal interop.

**Intel ISA-L deflate (opt-in)**

* Add ``backend="isal"`` option to ``DeflateCodec.encode/decode``.
  ISA-L's igzip is ~4× faster than libdeflate at encode but
  produces ~19% bigger output and is slightly slower at decode —
  not strictly Pareto-better, so opt-in via the backend kwarg rather
  than the new default.

**Compressor coverage**

* Add ``gzip`` and ``none`` codecs (stdlib-based wrappers); add
  ``zlibng`` as an alias of ``deflate``. Lets ic-compatible callers
  (tifffile, zarr filter chains) name backends explicitly.

**Pareto-default closures vs imagecodecs**

* ``aec`` encode: 3.51× → 1.74× on the 200 KB uint16 bench workload.
  Streaming ``aec_encode_init`` / ``aec_encode(AEC_FLUSH)`` /
  ``aec_encode_end`` API plus correct worst-case output cap
  (``srcsize * 67/64 + 257``).
* ``zfp`` encode: 1.19× → 0.98× on Mac (parity). Switched to
  absolute-dylib link to bypass macOS sysconfig's prepended
  ``/opt/homebrew/lib`` ordering; cache builds via
  ``bench/build_codec_libs.sh --only=zfp`` now win.
* ``blosc2``: 2.12× → 1.06× by matching imagecodecs's
  ``typesize=8`` default for bytes input (rather than 1). Output is
  bit-identical to ic on default settings. Also added the cached
  c-blosc2 2.23 build to the build recipe.

**Build infrastructure**

* ``bench/build_codec_libs.sh``: add ``-DCMAKE_POLICY_VERSION_MINIMUM=3.5``
  to CMAKE_COMMON so cmake 4.x can configure projects with pre-3.5
  ``cmake_minimum_required`` (x265, libheif).
* Add ``setup.py`` ``_user_cache_rpath_args()`` helper that auto-bakes
  the per-user cache lib dirs into every extension's DT_RUNPATH /
  LC_RPATH. Closes a class of "import works in tests but breaks at
  runtime" bugs.

**Live archive smoke tests**

* Add ``tests/test_live_archives.py`` (marked ``slow``) — open and
  read a 700 KB NASA GSFC FITS file and an EMBL-EBI IDR OME-Zarr
  dataset over HTTPS. Skips cleanly when the network is unreachable.
  Catches the class of bugs synthetic tests can't (Content-Type
  quirks, server recompression, real CDN HTTPS retry behavior).
