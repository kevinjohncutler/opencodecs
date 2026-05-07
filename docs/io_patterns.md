# opencodecs I/O patterns — design notes

Internal notes from the tifffile/CZI benchmarking exercises. The TL;DR
is below; rationale and measured numbers follow.

## TL;DR

1. **Coalesced sequential reads beat per-chunk pread** on modern fast
   storage (NVMe + 10 G NAS). The kernel prefetcher and the Python
   `BufferedReader` are both very good — fight them at your peril.
2. **Parallel decode is independent of parallel I/O.** Most workloads
   want one stream of I/O feeding many CPU-bound decoders. Don't
   parallelise the I/O unless you've measured it.
3. **mmap is "let the kernel prefetch" with extra steps.** Use it when
   the file fits in address space and you'll touch most bytes; let the
   kernel decide what to read ahead.
4. **`bytes(memoryview)` is a real copy.** When chunks reach the codec,
   they should arrive as memoryview / mmap slices, not as fresh bytes.
   Our codec adapters now accept buffer-protocol input directly.
5. **Streaming readers (`BackgroundChunkReader`) are for streamed
   formats only.** JXL frame-by-frame benefits from overlap. TIFF/CZI
   don't, because slurp-then-parallel-decode is faster when the file
   fits in RAM.

## Background: where this came from

Two benchmark exercises drove these conclusions:

* **tifffile patches** (May 2026). Tried three patches against tifffile:
  routing codec dispatch through opencodecs, Cython-izing
  `read_segments`, and replacing the I/O scheduler with parallel
  `os.pread`. Best result was 1.20× on warm-cache local disk; multi-page
  stack reading was *slower* than tifffile (0.65×). Conclusion: don't
  ship the patches; tifffile is already near-optimal on fast storage.

* **Native CZI reader** (May 2026). Built `opencodecs._czi_reader`
  targeting compression types 0 and 6 (the ones empirically present in
  the lab archive). Achieved 6.2× over `czifile` and within 1.4× of
  `aicspylibczi` (a well-tuned C++ implementation) using
  mmap + parallel zstd decode.

Numbers from the CZI bench (66 MB file, 14 sub-blocks of 2000×2000
uint16, ZSTDHDR compression, M3 Mac, 2 × cpu_count workers):

| Reader                            | Local warm | Local cold | NAS warm | NAS cold |
|-----------------------------------|------------|------------|----------|----------|
| czifile (Python)                  | 152 ms     | 158 ms     | 160 ms   | 152 ms   |
| opencodecs native (1 worker)      | 111 ms     |  99 ms     | 117 ms   | 104 ms   |
| **opencodecs native (parallel)**  | **14 ms**  | **14 ms**  | **18 ms**| **18 ms**|
| aicspylibczi (C++)                |  17 ms     |  17 ms     |  19 ms   |  18 ms   |

Net: opencodecs is **~20% faster than aicspylibczi on local disk** (the
nogil byteshuffle is the difference) and **essentially tied on NAS**
(both bottleneck on SMB I/O latency).

Pipeline benchmark (8 CZIs back-to-back from NAS, 795 MB total):

| Reader              | Pass total | Per-file | Throughput |
|---------------------|-----------:|---------:|-----------:|
| czifile             |   1603 ms  |   229 ms |  0.50 GB/s |
| opencodecs          |    198 ms  |    22 ms |  4.01 GB/s |
| aicspylibczi        |    191 ms  |    22 ms |  4.16 GB/s |

The 4% gap to aicspylibczi at this point is noise. Both readers are
bottlenecked on the NAS link at ~4 GB/s.

Linux x86_64 (Threadripper, 64 cores) on a single 195 MB CZI shows the
parallel-decode advantage scaling out:

| Reader          |   Time |
|-----------------|-------:|
| czifile         | 414 ms |
| aicspylibczi    | 140 ms |
| **opencodecs**  |  46 ms |

3× ahead of aicspylibczi here — more cores → more benefit from the
nogil-byteshuffle + persistent-pool combination.

## What worked, what didn't

### Worked

* **mmap + parallel decode** for self-contained file formats. CZI's
  ~14 sub-blocks per file are independent zstd streams — exactly the
  shape parallelism likes. We saturate ~7 of the M3's performance cores
  during decode and stop being I/O-bound.

* **Zero-copy memoryview through the codec layer.** Slicing an mmap
  with `view[a:b]` and passing the slice straight to `zstd_decode` saved
  ~17% on warm cache for CZI. The bytes() copy was visible at the level
  of "memcpy 8 MB at NVMe-bandwidth ≈ 3 ms" per sub-block.

* **`madvise(MADV_SEQUENTIAL)`** as a hint when we know we'll touch
  most of the file front-to-back. Free win for CZI on local disk; the
  kernel prefetches more aggressively.

* **Replacing `numpy.ascontiguousarray(arr.T)` with a nogil Cython
  byte-shuffle.** This was the single biggest win in the CZI work.
  cProfile on the serial decode showed 60% of time in
  `np.ascontiguousarray` for an 8 MB transpose. The transposed view has
  rows that are 8 MB apart in memory, so the copy is full-cache-miss
  per byte — ~9.6 ms per sub-block. A 30-line tight C loop in
  `_bytetools.pyx` brings that down to ~0.2 ms (memory-bandwidth
  limited) AND releases the GIL so the work parallelises across decode
  threads — which the numpy version couldn't because it holds the GIL.
  This single change took us from being 1.4× behind aicspylibczi to
  ~5% ahead on local-disk warm cache.

  General rule: any time you see numpy's `transpose() / .T /
  np.ascontiguousarray() / np.swapaxes()` on a multi-MB array in a hot
  path, profile it. The high-level numpy API papers over a real cost
  that scales with array bytes; in tight loops where you know the
  dimensions and dtype, hand-written Cython matches or beats it
  trivially. The win is doubled when the loop can be `nogil` so it
  parallelises across worker threads.

* **Persistent module-level thread pool.** The CZI reader originally
  used `with ThreadPoolExecutor(...) as ex:` per `read()` call.
  `shutdown()` blocks until every worker thread has exited the pool —
  ~1-2 ms in steady state. For a workload that calls `read()` many
  times (e.g. a pipeline opening 100s of CZI files) that's pure
  overhead. A module-level `_POOL = ThreadPoolExecutor(...)` created on
  first use brought our local-warm CZI from 19 ms median (with high
  variance) to 13.7 ms median (variance ≈ 0.3 ms), pushing us ~20%
  ahead of aicspylibczi. The pool sticks around for the life of the
  Python process; it's daemon-thread-backed so it doesn't block exit.

* **Don't blindly use `MADV_SEQUENTIAL`.** It tells the kernel to
  evict pages aggressively after read. On warm-cache scenarios (calling
  the reader repeatedly on the same file, common in pipelines) that's
  exactly the wrong hint — every call re-fetches from the SMB server.
  We removed the `madvise(MADV_SEQUENTIAL)` call from `CziReader`.
  Default kernel behaviour is fine for files that fit in RAM, which is
  ≥ 99% of the lab's CZI files. The hint *is* useful for one-shot
  reads of files larger than RAM, but that's not what we're optimising
  for.

### Didn't work (negative results worth keeping)

* **Per-chunk `os.pread` for tiled TIFF.** Issuing 256-1024 individual
  preads for tiles defeats the OS prefetcher (each pread looks like a
  random seek). Even if the tiles ARE contiguous on disk, pread doesn't
  signal that. Lost to tifffile's BufferedReader-based sequential reads.

* **Cythonizing tifffile's `read_segments`.** Profile showed the Python
  scheduler at 32 ms / 412 ms ≈ 8% of the total — small enough that even
  a perfect Cython port wouldn't materially help.

* **Repointing tifffile's codec dispatch at opencodecs.** Both libraries
  call the same system `libzstd` / `libdeflate`. Nothing to gain from
  the indirection layer — which is on the order of nanoseconds anyway.

## How to design a new reader

For a new format wrapper or native parser, ask in this order:

1. **Does the file fit in RAM?** If yes, `mmap` it and stop thinking
   about I/O. The kernel handles prefetch, page eviction, NUMA
   placement. Use `madvise(MADV_SEQUENTIAL)` if you'll scan front-to-
   back, `MADV_RANDOM` for chunk-index access patterns. You should
   probably never call `os.pread` directly.

2. **Is the format self-describing in a way that lets you find chunks
   without reading sequentially?** If yes, parse the directory once,
   then dispatch chunk-decode to a thread pool. Don't issue parallel
   READS — issue one big read or trust mmap, then run the decoders in
   parallel.

3. **Is the format truly streaming (decoder needs frame N before
   frame N+1)?** Use `BackgroundChunkReader`: a background thread
   reads chunks into a bounded queue, the foreground thread decodes
   them. This is what JXL streaming uses. It's NOT the right pattern
   for self-describing chunked formats — you'd be paying overhead for
   parallelism you don't get.

4. **Do you really need parallel I/O?** Almost never on local NVMe or
   10 G NAS. Almost always on cloud object storage (S3/GCS) where every
   GET request has 30-100 ms of round-trip latency. Match the pattern
   to the storage.

## Codec adapter pattern: zero-copy buffer protocol

Cython's typed memoryview `const uint8_t[::1] src` accepts any buffer-
protocol object that's 1D and contiguous. Use `try`/`except` to fall
back to `bytes()` for non-buffer inputs:

```cython
def encode(data, *, level=None) -> bytes:
    cdef:
        const uint8_t[::1] src
        ...
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    ...
```

That single 4-line block accepts: `bytes`, `bytearray`, `memoryview`,
`mmap.mmap`, numpy `uint8` 1D arrays, and anything else with a uint8
buffer protocol — all without an intermediate copy. The fall-back
handles BytesIO-like and string-coercible inputs.

Applied across `_zstd`, `_lz4`, `_brotli`, `_blosc2`, `_deflate` as of
this writing. The image codecs (`_jpeg`, `_jpeg2k`, `_avif`, `_heif`,
`_png`, `_webp`) already accept memoryview by default.

## When parallelism actually helps

Empirical rules from the benchmarks:

* **Pays off**: when each chunk decode dominates per-chunk I/O time AND
  the chunks are independent. CZI sub-blocks (~10 ms zstd each, ~14
  per file): parallelism delivers 5-6× over serial.
* **Doesn't pay off**: when the format already saturates I/O bandwidth
  with stock-Python sequential reads. Tiled TIFF on NVMe: stock
  tifffile already hits ~1.5 GB/s; parallelism gives 1.0-1.2× best case.
* **Hurts**: when fine-grained tasks pay more in dispatch overhead than
  they save in CPU time. Submitting 5000 50-µs decode tasks to a
  ThreadPoolExecutor: dispatch overhead dominates.

Default to a single-threaded implementation, then measure, then
parallelise only if the workload has independent chunks taking ≥ 1 ms
each.
