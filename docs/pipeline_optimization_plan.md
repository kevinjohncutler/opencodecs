# Cross-codec pipeline implementation record

Reviewed September 22, 2026. This replaces the initial implementation plan with
an account of shared mechanisms, acceptance gates and remaining native feature
boundaries. [pipeline_catalog.toml](../pipeline_catalog.toml) covers 61 registered
codecs, six direct adapters and 17 work packages. Runtime capability flags remain
in [capabilities.toml](../capabilities.toml).

## Design ethos

1. Share scheduling, ownership, source access, destinations and worker budgets.
   Keep format indexes, prediction, checksums, compositing and finalization in
   the format adapter.
2. Distinguish independently indexed pieces from stateful compressed streams.
   Interleave independent fetching and decoding only across valid reset points.
3. Bound bytes as well as task counts. Preserve producer ownership, explicit
   oversized-item behavior, cancellation and source lifetime.
4. Preserve quality, wire semantics, axes, byte order and metadata. Check against
   independent encoders or decoders instead of trusting symmetric round trips.
5. Measure throughput, first output, storage bytes, temporary allocation and
   native memory separately. Keep fast local paths and eager native engines when
   added scheduling has no useful benefit.
6. Keep negative findings. An upstream function or partial CPU decode is not
   proof of bounded output, selective storage or useful progressive latency.

## Work package outcomes

| Work package | Implementation or assessment |
|---|---|
| Correctness | HDF5 active filter order and masks, checksum-preserving fallback, MRC/NIfTI destination ordering, shaped NDTiff image inputs |
| Output buffers | Validated common byte/array destinations, native forwarding, explicit remaining native intermediates |
| Bounded pipeline | Shared task/byte reservations, owned preparation, ordered results, oversized-item isolation, close/join and borrowed executor ownership |
| Worker budget | Shared outer budget, nested inline work, inner native limits, persistent pools, Electron Event Representation sum error propagation |
| Fetch/decode | TIFF, CZI, Zarr, N5, HDF5 family, FITS, VSI, OIB, NDTiff and DICOMweb integrations with fast local controls |
| Direct placement | TIFF segments, HDF5 chunks, ND2 frames and fixed-rate ZFP block bands |
| Bounded writes | Lazy owned NDTiff frames and Zarr chunks; existing CZI sub-block and TIFF batch output |
| Shards/pyramids | Streamed local shard payloads with final index, lazy pyramid levels; optional forward-only TIFF disk spooling |
| Array destinations | Direct MRC/NIfTI/NumPy serialization, bounded conversion and short-write handling |
| Output sinks | GIF/JPEG XL drains, Brunsli/HEIF callbacks, PNG rows, seekable JPEG 2000 output and B2nd file output |
| Incremental bytes | Seven genuine stateful byte codecs with backpressure, truncation checks and finalization |
| Row sessions | Actual PNG ordinary/interlaced updates and sequential encoding; source-grounded assessment of each other named target |
| Indexed native | B2nd persistent local slices, JPEG 2000/HEIF/AVIF source callbacks; XZ, Blosc2 and SPERR boundaries documented |
| Scratch/contexts | Direct predictor destinations, reusable float scratch, owned color/JPEG contexts, packed integer kernel, DICOM byte-plane assembly and Rice output |
| Sequence encoding | Explicit unsupported-second-frame errors; timed animation and image collection encoder requirements documented separately |
| Raw selection | Counted selected-plane reads for raw arrays; remote OIR coalescing; LIF/DM source audit and no-change decisions |
| Bit-exact verification | Optional actual decode and bounded bit comparison inside independent-piece writers; CZI overlaps one pending check with the next encode |

Format acronym definitions and public behavior are in
[shared_pipeline_capabilities.md](shared_pipeline_capabilities.md). Native target
assessments, including functionality requiring additional bindings or new format
contracts, are in [native_pipeline_binding_audit.md](native_pipeline_binding_audit.md).
Completing an investigation is not a claim that every potential format feature
has been implemented. Those prerequisites remain visible rather than becoming
false capability flags.

## Reproducible evidence

Run catalog checks without optional native libraries:

```sh
python ci/check_pipeline_catalog.py verify
python ci/check_pipeline_catalog.py report
```

A full native build can additionally run:

```sh
python ci/check_capabilities.py verify --strict
```

The validator checks every codec and adapter is assessed, dependencies have no
cycles, and source/test evidence exists. Completed work requires implementation
evidence. Runtime flags cannot be inferred from an unmeasured format possibility.

Each round was measured against a preserved copy of the source tree taken
before any edit, with alternating paired runs in fresh processes, and the
raw timings and scripts were kept with the working copy rather than in the
repository. Results describe synthetic, delayed-source, native and local
controls explicitly. The final combined selection passed 1240 tests with 12
skips; strict runtime capability verification found zero discrepancies
across 60 built codecs. The per-record `limits` in `pipeline_catalog.toml`
carry the measured figures and their boundaries.

## How to read the measurements

A throughput ratio is baseline elapsed time divided by new elapsed time; 2x
means the same verified work took half as long. Peak traced allocation is the
largest live allocation total observed by Python's tracer during the operation,
including supported NumPy allocations but excluding some native-library memory.
Process resident memory includes more native storage but also allocator and
runtime overhead. Source bytes count bytes requested or fetched from storage;
they are distinct from decoded pixels and output size.

Sparse cache misses are requests for scattered uncached extents. Reads inside
cached ranges are requests served by a larger already-cached extent without a
new source read. Adaptive retransition means read-ahead changes mode as access
returns from scattered offsets to nearby sequential requests. Cache measurements
from earlier rounds must not be labeled as new results of this implementation.
