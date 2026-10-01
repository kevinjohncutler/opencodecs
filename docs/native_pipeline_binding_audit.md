# Native pipeline binding audit

This audit records what the checked-in bindings can actually call. A whole-image
wrapper does not prove that a format cannot stream. Conversely, an upstream
function name does not prove bounded memory, useful first output, or independent
random access. Measurements distinguish caller-owned input/output, Python traced
allocations, native working memory, and bytes requested from storage.

## Row and progressive sessions

| Catalog target | Source evidence | Outcome and boundary |
|---|---|---|
| Portable Network Graphics (PNG) | `codecs/_png.pyx:RowDecoder`, `RowEncoder`; `spng.pxd`; `core/rows.py` | Implemented actual libspng scanline sessions. Ordinary rows and Adam7 sparse pass samples carry explicit placement coordinates. Returned arrays own pixels. Encoding requires known geometry and sequential noninterlaced rows. Native metadata and compression dictionaries are additional memory. Independent libpng fixtures cover interlace and tiny images. |
| Joint Photographic Experts Group (JPEG) | `codecs/_jpeg.pyx:decode`; `turbojpeg.pxd:tj3Decompress8` | Current TurboJPEG application programming interface (API) consumes a full compressed image and writes a full output. Native reduced-resolution decode is forwarded publicly. True scanline sessions require a separate lower-level libjpeg binding with source/destination managers, suspension, error handling and progressive-scan semantics. Restart markers alone are not a row index. |
| MozJPEG | `codecs/_mozjpeg.pyx:decode`; `mozjpeg.pxd:tjDecompress2` | Same distinction as JPEG, with the older TurboJPEG interface. Decode-time scaling is forwarded. No callable row session is declared in this binding. |
| Red, green, blue and exponent (RGBE) | `rgbe.pxd:RGBE_ReadPixels_RLE`, `RGBE_WritePixels_RLE`; `_rgbe.pyx:decode` | Native calls accept a scanline count, but their `rgbe_stream_t` stores contiguous encoded memory. An owned cursor can emit rows while retaining encoded input; selective storage requires changing the native stream abstraction. Orientation can reverse rows or transpose axes, so source scanline order must be explicit. |
| Windows bitmap (BMP) | `codecs/_bmp.pyx:decode_bgr24_to_rgb`, `decode_bgra32_to_rgba`; `_bmp_codec.py` | Uncompressed pixel rows have computable byte offsets and padding. A row API can use those offsets, preserving bottom-up/top-down orientation and channel conversion. Paletted and run-length encoded modes need separate state handling. This is a new parser-facing row API, not an existing native suspend/resume function. |
| High Throughput JPEG 2000 (HTJ2K) | `codecs/openjph_shim.cpp:oc_openjph_encode`, decode implementation; `codestream::exchange`, `pull` | Callable line processing is proven. Current shim enforces planar component order and owns `mem_infile`/`mem_outfile` within one call. A persistent C++ object and component-aware update contract are prerequisites. Claiming complete interleaved color rows would otherwise require retaining earlier component planes. Resolution reduction is already public. |
| Quite OK Image (QOI) | `qoi.pxd:qoi_decode`; `codecs/_qoi.pyx:decode` | Reference API allocates a full decoded image. Incremental rows require an actual stateful decoder retaining previous pixel, index table and run state across input boundaries. Slicing eager output would not satisfy the catalog. |
| JPEG lossless (JPEG-LS) | `charls.pxd:charls_jpegls_decoder_set_source_buffer`, `decode_to_buffer` | Declared CharLS interface is whole source/destination with explicit stride and interleave mode. No feed/drain or row callback is bound. Lower-level native support must be established before promotion; resetting per row changes prediction state and is not equivalent. |
| WebP | `webp.pxd`; `webp_shim.h:oc_webp_encode` | Existing still decode is whole-image and animation decode has canvas/disposal dependencies. Incremental native decode needs additional declarations and output-lifetime tests; an animation frame iterator is not a scanline session. The current encoder shim uses `WebPPicture` and `WebPMemoryWriter`, not a row source. |
| AV1 Image File Format (AVIF) | `avif.pxd:avifDecoderParse`, `avifDecoderNthImage`, `avifImageYUVToRGB` | Implemented selective input callbacks for the persistent sequence reader. The native decoder still owns complete decoded planes; input callbacks do not make this a row decoder. Custom input must return each requested contiguous extent until the next call, and nonpersistent input can cause native sample copies. |
| High Efficiency Image File Format (HEIF) | `heif.pxd:heif_context_read_from_reader`, `heif_decode_image` | Implemented seekable input callbacks and encoded output callbacks. Decoding still produces a native image plane copied to caller output. No row-finality API is declared; collections and tiled images are separate native concepts. |
| Ultra high dynamic range (Ultra HDR) adapter | `_ultrahdr_codec.py:UltraHdrCodec`; `codecs/libuhdr.pxd`; `_uhdr.pyx` | Direct adapter inventory entry outside the codec manifest. Native decode combines base image, gain map and high dynamic range metadata. A base JPEG row alone is not equivalent output. Requires an explicit native pipeline for gain-map availability and reconstruction before any row claim. |

## Native indexes and selective sources

| Catalog target | Source evidence | Outcome and boundary |
|---|---|---|
| JPEG 2000 | `_jpeg2k.pyx:decode_region`, `decode_tile`; OpenJPEG read/skip/seek callbacks | Implemented `NativeSource` callbacks for region/tile requests. Native source length remains 64-bit. The codec chooses which tile packets to fetch; storage-byte savings require a suitable tiled codestream and are tested separately from decoded-pixel savings. The source adapter does not synthesize a missing packet index. |
| B2nd | `b2nd_helpers.c:oc_b2nd_open`, `oc_b2nd_read_slice`; `_b2nd.pyx:FileReader` | Implemented local persistent file reader using `b2nd_open` and a retained array/decompression context. Repeated boxes reuse metadata and touch native indexed chunks. Independent multi-chunk/block fixtures verify edge and interior boxes. A per-reader lock prevents simultaneous context mutation. Remote `DataSource` integration remains separate work; this path is explicitly local storage. |
| Blosc2 | `_blosc2.pyx:decode_partial`; `blosc2.pxd:blosc2_getitem_ctx` | Existing partial decode skips unrelated decompression but accepts a complete compressed chunk. The raw chunk API is not the B2nd persistent container. Storage selection requires a lazy-chunk/storage API binding or a validated block-offset parser; partial CPU work alone does not justify a range-read claim. |
| HEIF | Native `heif_reader` version 1 callbacks and `heif_context_get_image_handle` | Implemented source callbacks and metadata-only open geometry. Borrowed sources stay open. Native version 2 range-preload hints exist in the inspected installed header but are not required for version 1 correctness and are not advertised as integrated scheduling. Collections may contain differently sized images. |
| AVIF | `avifIO.read`, `avifDecoderSetIO`, `AvifSequence` | Implemented custom input for public `open` and metadata frame count. One native decoder retains the parsed sample table. Inter-frame dependencies may require reference samples; frame indexing does not guarantee independently decodable frames. Scratch can grow to the largest requested compressed extent. Eager decode retains its existing fast path. |
| Lempel-Ziv-Markov chain algorithm (LZMA)/XZ | `core/streaming.py:_StdlibDecoder`; `_lzma_codec.py` | Implemented genuine incremental decode/encode through standard-library state. Arbitrary block selection needs XZ footer/index parsing plus block/filter reset and checksum handling, or additional liblzma index APIs. Incremental sequential input is not indexed access. |
| SPERR | `sperr.pxd:sperr_trunc_3d`; `_sperr.pyx:decode_native` | A truncation function is already declared, but not exposed publicly. It produces a reduced-information progressive stream, which changes reconstruction quality. This is a quality/size tradeoff requiring its own public contract and error-bound tests, not an equal-quality storage optimization or an arbitrary spatial index. Native thread budgeting is now forwarded. |

## Output sinks and incremental bytes

`core/streaming.py` uses actual persistent states for Deflate, gzip, bzip2, XZ,
Brotli, LZ4 and Zstandard. The iterator bounds emitted chunks, preserves consumer
backpressure, validates finalization, and closes abandoned state. Compression
windows and some native block output remain additional working memory. The eager
APIs keep their existing defaults. Deflate's explicit incremental mode uses the
standard-library zlib engine; the eager optimized implementation remains separate.

Adaptive Entropy Coding (AEC) has a declared `aec_stream` and the encode and
decode init/process/end functions in `libaec.pxd`. `_aec.pyx:decode_raw` drives
`aec_decode_init`/`aec_decode`/`aec_decode_end` over a whole input buffer, taking
the expected byte count from the caller, since the raw stream does not supply a
self-describing array size. An incremental public protocol is still missing:
arbitrary byte chunks must preserve sample/block alignment and final padding
semantics, so it should not be added to the generic seven-codec iterator by
guessing termination from exhausted input.

HEIF output callbacks now bypass the wrapper's growing encoded buffer. Native
libheif may retain encoded image data internally. Explicit B2nd path output
(`storage_output=True`) uses
`blosc2_storage.urlpath`, writes an adjacent temporary file, then replaces the
destination. It skips `b2nd_to_cframe` and Python encoded bytes, but the current
single-chunk writer can still retain a whole native compressed chunk. The default
path writer keeps the original buffer implementation because direct storage
measured slower; the explicit option trades throughput for lower traced allocation. PNG's row
encoder writes through libspng callbacks. JPEG 2000 now writes through seekable
destination callbacks because the container writer patches offsets. Relative
positions and partial writes are preserved. Forward-only destinations retain
the eager buffer fallback. Native component planes remain additional memory.
GIF, JPEG XL and Brunsli sink implementations are covered by their separate writer
tests and measurements.

## Scratch and context reuse

| Catalog target | Boundaries and native evidence |
|---|---|
| Color Management System (CMS) | Explicit owned transform work is separate from native compressor sessions. A transform is configuration-specific and must preserve profiles, pixel layout and rendering intent. See `_cms_codec.py`. |
| JPEG, MozJPEG | Eager decode retains per-call `tj3Init`/`tj3Destroy` and `tjInitDecompress`/`tjDestroy`. Explicit `codec.decoder()` contexts reuse handles and serialize calls. JPEG scaling resets between calls; tests vary dimensions, gray/color layouts and malformed-input recovery. Adoption is explicit and throughput measurements determine whether reuse helps. No automatic global pool has been introduced. |
| Bitshuffle | `_bitshuffle.pyx` exposes caller output. Block size, item size and leftover elements define reset boundaries. Reusing output does not remove native transform scratch automatically. |
| Byteshuffle | `_byteshuffle_codec.py` forwards caller storage. The transpose is item-width dependent; metadata belongs to each call. Returned byte views must alias caller storage rather than a copied bytearray slice. |
| Delta, exclusive OR (XOR), floating-point predictor | `_predictor_codec.py` and native bytetools define axis, distance, byte order and row-reset rules. Shared scratch reuse must preserve those boundaries; root pipeline work covers the direct-output paths. |
| Packints | Native `unpackints_into` decodes packed samples directly into caller storage without a samples-by-bits matrix. Independent streams cover widths through 64 bits, tails, signed storage and byte order. One-bit input uses the faster NumPy unpackbits path; standard byte widths retain their existing view/conversion path. |
| Quantize | `_quantize_codec.py` distinguishes numeric rounding from decoding already stored values. Reusing decode output does not imply an in-place lossy encode is safe for caller arrays. |
| Digital Imaging and Communications in Medicine run-length encoding (DICOM RLE) | `_dicomrle_codec.py:_assemble_dicomrle_array` combines independently encoded byte planes. Segment boundaries and byte significance must survive scratch reuse. Native/direct assembly is a separate kernel from a generic byte-buffer pool. |
| Rcomp | `_rcomp.pyx:decode_raw` (the bare Rice stream) and `decode_framed` (blobs from 0.4.0 and earlier) call typed Rice decoders directly into validated caller storage. The element count, pixel size and block size come from the caller for a bare stream and from the validated old header otherwise, and the decoders check every input byte against the end of the stream; big-endian inputs and outputs preserve numeric values. Block prediction and external raw dimensions remain codec responsibilities; no arbitrary seek point is implied. |
| Block Compression (BCn) | `_bcdec.pyx:decode_block_rows` proves independent 4-by-4 block rows. Destination pitch and partial edge geometry are the placement constraints. Block decoder functions have no reusable heavyweight context. |
| Limited Error Raster Compression (LERC) | `lerc_decode` writes to caller output; validity-mask allocation can remain. The declared API has no persistent decode context to pool. Preserve mask and multiband geometry. |
| Pcodec | `pco_standalone_simple_decompress_into` writes directly to output. The declared C wrapper has no persistent page decoder. Rust-level page/session support would require additional bindings. |
| SZ3 | `SZ_decompress` returns native allocated storage, then the wrapper copies it into output. Public `out` saves the Python allocation, not the native intermediate. A direct destination or reusable session requires a different native interface. |
| None | Identity transport has no compressor state. Buffer ownership and copy policy determine whether reuse helps; a native pool would add overhead. |

Zstandard source records a previous automatic context-pool regression. Explicit
worker-owned context experiments must therefore retain small and large eager
controls and concurrent-caller parity. B2nd's file reader already retains its
array-owned decompression context. None of these findings supports a universal
context pool across codecs.

## Sequence writers

WebP, AVIF and HEIF common writers now reject a second frame explicitly using
`max_writer_frames=1`. This closes the false implication that sequence decoding
also provides sequence encoding. Current native writer entry points are
`oc_webp_encode`, `avifEncoderWrite`, and one `heif_context_encode_image` call.
Timed WebP animation requires the mux/animation writer API and disposal/timestamp
semantics. Timed AVIF encoding requires frame submission/finalization plus timing
and keyframe policy. HEIF can hold a collection of top-level images, which is not
automatically a timed animation. These are additional format features with their
own independent-reader conformance requirements, not missing shared scheduling.

## Acceptance and remaining feasible work

The source callback tests count requested bytes on independently encoded tiled
JPEG 2000 and metadata-only HEIF/AVIF opens. They also test short reads, original
source exceptions, caller position, and ownership. PNG tests reconstruct ordinary
and interlaced images against independent decoders. B2nd tests create independent
multi-chunk storage and prohibit the full-source helper during region reads.

Still feasible, without claiming completion prematurely: lower-level JPEG row
bindings, component-aware OpenJPH
sessions, callback-backed RGBE streams, uncompressed BMP row addressing, QOI state
machine sessions, AEC count-aware incremental streams, and the separately scoped
native/storage/index work listed above. Each needs its own measured acceptance;
an audit outcome is not a fabricated implementation or a zero-copy claim.
