# opencodecs bench — threadripper (x86_64)

- Run at: `20260511T070021Z`
- opencodecs: `0.2.0.dev0` (git: `261a5be`)
- Python: 3.12.13, CPU: AMD Ryzen Threadripper PRO 3995WX 64-Cores × 128
- Reference libraries: tifffile 2026.5.2, ndstorage 0.1.18, czifile 2026.4.30, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| tiff_random_tile_read | 1.20 | 1.17 | 1.25 | 0.01 |  | 37.60× |
| tiff_pyramid_crop_from_fullres | 0.86 | 0.83 | 0.89 | 0.04 |  | 22.12× |
| ndtiff_index_parse_synthetic_10k | 5.30 | 5.27 | 5.46 | 0.05 |  | 6.56× |
| ndtiff_random_frame_read | 3.18 | 3.07 | 3.51 | 0.36 |  | 1.13× |
| ndtiff_write_50_frames | 12.82 | 12.28 | 13.34 | 0.63 |  | 0.97× |
| ndtiff_write_compressed_zstd | 61.75 | 60.31 | 63.10 | 1.76 |  | 5.01× of uncompressed |
| tier1_codecs_roundtrip_10mb/zstd | 12.70 | 12.24 | 12.99 | 0.48 |  | 826 MB/s |
| tier1_codecs_roundtrip_10mb/lz4 | 4.07 | 4.05 | 4.12 | 0.02 |  | 2575 MB/s |
| tier1_codecs_roundtrip_10mb/brotli | 87.24 | 82.47 | 96.07 | 5.04 |  | 120 MB/s |
| tier1_codecs_roundtrip_10mb/blosc2 | 6.47 | 6.43 | 6.54 | 0.04 |  | 1621 MB/s |
| tier1_codecs_roundtrip_10mb/deflate | 282.28 | 279.79 | 284.40 | 1.15 |  | 37 MB/s |
| tier1_codecs_roundtrip_10mb/bitshuffle | 4.07 | 4.05 | 4.19 | 0.12 |  | 2573 MB/s |
| tier1_codecs_roundtrip_10mb/b2nd | 12.98 | 11.63 | 14.23 | 1.37 |  | 808 MB/s |
| tier1_codecs_roundtrip_10mb/pcodec | 102.68 | 100.92 | 103.66 | 1.59 |  | 102 MB/s |
