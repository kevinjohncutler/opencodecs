# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T081121Z`
- opencodecs: `0.2.0.dev0` (git: `cabd448`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_zstd_10mb/encode | 9.98 | 9.95 | 10.05 | 0.07 |  | 0.95× vs ic |
| h2h_zstd_10mb/decode | 9.24 | 9.23 | 9.27 | 0.02 |  | 1.00× vs ic |
| h2h_deflate_10mb/encode | 281.81 | 278.82 | 284.64 | 3.97 |  | 0.98× vs ic |
| h2h_deflate_10mb/decode | 23.68 | 23.59 | 23.93 | 0.10 |  | 1.73× vs ic |
| h2h_lz4_10mb/encode | 2.62 | 2.61 | 2.66 | 0.04 |  | 1.01× vs ic |
| h2h_lz4_10mb/decode | 0.25 | 0.24 | 0.28 | 0.01 | ⚠️ | 1.30× vs ic |
| h2h_brotli_10mb/encode | 56.39 | 56.07 | 57.74 | 0.21 |  | 0.95× vs ic |
| h2h_brotli_10mb/decode | 47.22 | 47.15 | 47.27 | 0.02 |  | 1.01× vs ic |
| h2h_blosc2_10mb/encode | 4.34 | 4.31 | 4.40 | 0.06 |  | 11.77× vs ic |
| h2h_blosc2_10mb/decode | 0.25 | 0.24 | 0.26 | 0.01 | ⚠️ | 27.61× vs ic |
| h2h_jpeg_4mp_rgb/encode | 16.17 | 16.09 | 16.26 | 0.09 |  | 1.08× vs ic |
| h2h_jpeg_4mp_rgb/decode | 28.87 | 28.60 | 29.96 | 1.08 |  | 0.99× vs ic |
| h2h_png_4mp_rgb/encode | 319.00 | 316.29 | 320.32 | 2.40 |  | 0.86× vs ic |
| h2h_png_4mp_rgb/decode | 6.84 | 6.84 | 6.87 | 0.03 |  | 1.60× vs ic |
| h2h_webp_4mp_rgb/encode | 585.41 | 584.61 | 587.26 | 1.70 |  | 0.94× vs ic |
| h2h_webp_4mp_rgb/decode | 142.39 | 141.54 | 144.18 | 1.68 |  | 0.98× vs ic |
| h2h_jpeg2k_4mp_u16/encode | 702.34 | 693.68 | 708.28 | 8.94 |  | 0.95× vs ic |
| h2h_jpeg2k_4mp_u16/decode | 102.30 | 102.30 | 102.31 | 0.00 |  | 6.49× vs ic |
| h2h_qoi_4mp_rgb/encode | 16.65 | 16.13 | 17.49 | 1.05 |  | 0.93× vs ic |
| h2h_qoi_4mp_rgb/decode | 7.46 | 7.42 | 7.55 | 0.04 |  | 0.95× vs ic |
| h2h_lerc_4mp_u16 | skipped: lerc static-library symbol clash with imagecodecs (cross-process bench TODO) | — | — | — | — | — |
| h2h_jxl_4mp_rgb/encode | 156.14 | 155.91 | 156.30 | 0.00 |  | 19.84× vs ic |
| h2h_jxl_4mp_rgb/decode | 25.91 | 24.77 | 26.99 | 1.57 |  | 19.21× vs ic |
