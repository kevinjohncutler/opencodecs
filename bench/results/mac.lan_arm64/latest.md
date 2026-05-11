# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T065651Z`
- opencodecs: `0.2.0.dev0` (git: `261a5be`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| tiff_random_tile_read | 0.79 | 0.77 | 0.86 | 0.08 |  | 10.56× |
| tiff_pyramid_crop_from_fullres | 0.48 | 0.46 | 0.51 | 0.03 | ⚠️ | 19.29× |
| ndtiff_index_parse_synthetic_10k | 4.56 | 4.38 | 4.95 | 0.30 |  | 6.15× |
| ndtiff_random_frame_read | 1.22 | 1.18 | 1.32 | 0.03 |  | 1.23× |
| ndtiff_write_50_frames | 4.25 | 4.20 | 4.39 | 0.08 |  | 1.08× |
| ndtiff_write_compressed_zstd | 44.93 | 44.26 | 46.37 | 1.03 |  | 10.58× of uncompressed |
| tier1_codecs_roundtrip_10mb/zstd | 10.69 | 10.48 | 10.91 | 0.25 |  | 981 MB/s |
| tier1_codecs_roundtrip_10mb/lz4 | 2.83 | 2.70 | 3.01 | 0.19 |  | 3709 MB/s |
| tier1_codecs_roundtrip_10mb/brotli | 62.72 | 59.22 | 65.93 | 4.89 |  | 167 MB/s |
| tier1_codecs_roundtrip_10mb/blosc2 | 4.59 | 4.58 | 4.60 | 0.00 |  | 2286 MB/s |
| tier1_codecs_roundtrip_10mb/deflate | 249.68 | 236.06 | 252.80 | 12.31 |  | 42 MB/s |
| tier1_codecs_roundtrip_10mb/bitshuffle | 2.62 | 2.52 | 2.68 | 0.12 |  | 3997 MB/s |
| tier1_codecs_roundtrip_10mb/b2nd | 15.31 | 14.91 | 16.00 | 0.73 |  | 685 MB/s |
| tier1_codecs_roundtrip_10mb/pcodec | 59.13 | 58.72 | 59.29 | 0.21 |  | 177 MB/s |
