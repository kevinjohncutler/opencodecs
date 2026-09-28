# Shared streaming and buffering capabilities

The optimization unit is a reusable mechanism: source access, buffer ownership,
bounded scheduling, destination handling, or native session lifetime. Format
adapters retain their indexes, predictors, filter order, channel layout,
compositing, and finalization. A stateful compressed stream cannot be split into
independently decoded pieces simply by slicing its bytes.

[capabilities.toml](../capabilities.toml) records public codec flags.
[pipeline_catalog.toml](../pipeline_catalog.toml) covers all 60 codecs and six
direct adapters, including investigated opportunities that need new native
bindings or format contracts. See the [implementation record](pipeline_optimization_plan.md)
and [native binding audit](native_pipeline_binding_audit.md) for those boundaries.

## Public flags

| Field | Meaning | Limit |
|---|---|---|
| streaming_decode | Yield frames without retaining all decoded frames | Compressed input may still be retained |
| streaming_encode | Encode submitted frames without retaining the raw stack | Encoded output may still be retained |
| streaming_output | Destination receives encoded bytes before close | Finalization may require seeking |
| decode_overlap | Bounded input fetching can overlap decode | Source and reader modes determine applicability |
| writer_buffering | all, frame, encoded, or unsupported | A bytes-returning call necessarily retains output |
| range_reads | Storage is addressed by offset | Does not promise every selection saves bytes |
| parallel_decode | Multiple workers contribute to decode | Does not imply parallel fetching |

## Shared mechanisms

The table uses these format names: TIFF (Tagged Image File Format), CZI (Carl
Zeiss Image), HDF5 (Hierarchical Data Format version 5), FITS (Flexible Image
Transport System), DICOMweb (Digital Imaging and Communications in Medicine web
services), MRC (Medical Research Council), NIfTI (Neuroimaging Informatics
Technology Initiative), GIF (Graphics Interchange Format), PNG (Portable Network
Graphics), HEIF (High Efficiency Image File Format), JPEG (Joint Photographic
Experts Group), and LZMA (Lempel-Ziv-Markov chain algorithm). Other names are codec
identifiers.

| Mechanism | Implementation | Consumers and behavior |
|---|---|---|
| Checked output storage | core.buffers | Public byte and array decoders validate writable layout and forward native destinations; native copies remain explicit |
| Byte and task backpressure | core.pipeline.map_bounded | TIFF, CZI, Zarr, N5, HDF5, FITS, VSI, OIB, NDTiff and DICOMweb adapters use ordered bounded work where appropriate |
| Worker budget | core.pipeline.WorkerBudget | Shared budgets cap concurrent outer workers; nested piece work runs inline; inner native codecs avoid multiplied worker pools |
| Typed input ownership | core.segment_compression.prepare_segment_input | Image compressors receive shaped arrays; byte compressors receive byte views; reused producer pixels and metadata are snapshotted before advancing |
| Persistent executors | core.io.get_reader_pool, core.parallel.shared_pool | run_batched, map_batches and map_bounded share one process-wide pool; readers reuse family-owned pools; closing an iterator joins only its own work; a forked child builds fresh pools |
| Fair share | core.parallel.fair_share, auto_threads | Automatic worker counts shrink while other parallel calls run: pool helpers split one call's full width (one GIL per process), native codec threads split the CPU count; explicit counts are honored |
| Offset sources | core.io, core.native_source | Checked range responses, covering caches, coalescing and private native callback cursors; borrowed sources remain open |
| Direct placement | Format-specific adapters | TIFF regions, HDF5-family chunks, ND2 frames and fixed-rate ZFP bands land in the final destination |
| Bounded serialization | core._write_helpers | MRC, NIfTI and NumPy destination output avoids a complete serialized-volume temporary; short writes are checked |
| Encoded sinks | Native callbacks and drains | GIF, JPEG XL, PNG, HEIF and Brunsli emit to destinations; JPEG 2000 requires seeking or an explicit eager fallback |
| Stateful byte iteration | core.streaming | Deflate, gzip, bzip2, LZMA, Brotli, LZ4 and Zstandard preserve state across arbitrary input feeds |
| Row and pass updates | core.rows | PNG exposes owned ordinary rows and sparse Adam7 pass updates; no hidden full-image reconstruction |
| Explicit reusable state | core.scratch and owned contexts | Predictor scratch, color transforms, JPEG decoders and file-backed B2nd contexts have caller-visible lifetimes |
| Shards and pyramids | _omezarr_writer | Shards stream to temporary storage before their index; pyramid levels are generated lazily |
| Bit-exact verification | core.verification | Optional actual decode and pixel-bit comparison before TIFF, CZI, NDTiff and Zarr pieces are emitted |

## Overlapped bit-exact verification

Use `verify=True` on TIFF page/stream/pyramid writing, CZI frame/pyramid writers,
NDTiff writers, or Zarr array/pyramid writing to check each encoded piece through
its actual decoder. TIFF also reverses its predictors. Image decoders compare
shape, scalar type and pixel bits; byte codecs compare exact serialized pixel
bytes, with declared shape and type left to container validation. Signed zero
and NaN (not a number) payload bits are preserved. Array comparison normalizes
byte order without floating-point conversion. A lossy setting that changes any
pixel fails the check rather than receiving a tolerance-based exemption.

TIFF, NDTiff and Zarr check pieces inside their bounded worker tasks, allowing
one worker to verify while another encodes. CZI retains one owned pending piece
and checks it on one worker during compression of the next. Only checked pieces
are emitted. CZI close waits for the final check. Producer buffer reuse remains
safe, and failures join outstanding work before releasing its buffers.

Verification adds decoder output, bounded comparison scratch, and native decoder
workspace. Parallel writers increase task reservations for the known extra
storage; CZI bounds this stage to one pending piece. Comparison scratch uses
64 KiB blocks. Native workspace is additional, so this is not a hard total-memory
cap. Verification defaults to false and cannot be assumed free: overlap helps
when there is spare compute or another stage is waiting.

This checks the encoded pixel payload before output. It does not reread the
stored file or validate its completed index and metadata. Independent full-file
readback remains a separate correctness gate. Benchmarks distinguish identical
per-piece checks from full-file checks, since those perform different work.

## Memory and lifetime contracts

`map_bounded` reserves adapter-estimated owned input, scratch and retained result
bytes. It holds a yielded result's reservation until iteration resumes. A single
oversized item runs alone. The caller's final output, source caches, indexes,
native allocator overhead and native compression windows are separate costs.
These reservations are not a total-process memory limit.

Preparation runs before advancing a producer. This preserves acquisition loops
that reuse pixels or metadata. Closing an iterator cancels queued tasks and joins
running tasks before their source can close. A supplied executor and source
iterator remain caller-owned. Use `contextlib.closing` around partially consumed
internal iterators.

`out=` is an ownership contract, not a universal zero-copy claim. Some native
libraries allocate their own complete image before copying into caller storage.
The native audit identifies these cases. A reused scratch buffer belongs to one
active operation at a time. Explicit color and image decoder contexts serialize
concurrent use rather than sharing unsafe mutable native state globally.

## Selection and streaming are different

CZI `from_http(..., range_reads=True)` reads its index and selected payloads;
the existing whole-file mode remains the default. TIFF region readers enable
bounded prefetch for actual remote offset sources, while local reads retain the
fast coalesced path. Zarr and N5 use concurrent independent object fetches.
HTTP (Hypertext Transfer Protocol) range responses must actually honor offsets;
a server returning a full object is rejected before consuming that body.

JPEG 2000, HEIF and AVIF (AV1 Image File Format) callbacks allow the native
container parser to choose source extents. Savings depend on the actual index
and dependencies. AVIF may request a complete compressed sample and retain its
extent. B2nd selective file access currently uses local native storage.

PNG interlace updates include coordinates and pass identity. They are sparse
updates, not necessarily final rows. Its row encoder requires known geometry
and sequential noninterlaced rows. Sequence reading does not imply sequence
writing: WebP, AVIF and HEIF common writers reject a second frame explicitly.

## Performance gates and opt-in tradeoffs

Default fast local paths remain controls. HDF5/FITS byte-budget modes and OIB
prefetch are explicit because scheduling can cost more than it saves on fast
memory. NDTiff retains its fast count-prefetch reader when no byte or worker
budget is requested. Raw copying remains serial where workers do not help.

TIFF also writes floating-point predictor 3 for supported compressed float layouts,
with byte-for-byte independent predictor checks and bitwise file round trips.

Forward-only TIFF can explicitly spool an encoded page using `spool_threshold`.
This exchanges temporary memory for disk traffic; it is not advertised as a
throughput improvement. The default remains in memory. Stateful byte iteration
likewise adds bounded output and early delivery alongside existing eager engines,
without replacing those engines globally.

Rejected experiments and workload-specific measurements are recorded alongside
successful results. A faster delayed-source simulation does not establish a
universal network speedup. Peak Python-traced allocation excludes some native
memory and is distinct from process resident memory and reservation counters.
