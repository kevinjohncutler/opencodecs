# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T074052Z`
- opencodecs: `0.2.0.dev0` (git: `f2407e2`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_zstd_10mb/encode | 10.23 | 9.99 | 10.70 | 0.51 |  | 0.92× vs ic |
| h2h_zstd_10mb/decode | 9.23 | 9.21 | 9.28 | 0.04 |  | 1.00× vs ic |
| h2h_deflate_10mb/encode | 283.36 | 279.04 | 299.54 | 16.21 |  | 0.98× vs ic |
| h2h_deflate_10mb/decode | 23.93 | 23.59 | 25.24 | 0.91 |  | 1.71× vs ic |
| h2h_lz4_10mb/encode | 2.62 | 2.59 | 2.69 | 0.05 |  | 1.01× vs ic |
| h2h_lz4_10mb/decode | 1.56 | 1.53 | 1.67 | 0.08 |  | 0.16× vs ic |
| h2h_brotli_10mb/encode | 56.18 | 56.07 | 57.22 | 0.53 |  | 0.95× vs ic |
| h2h_brotli_10mb/decode | 47.37 | 47.29 | 47.43 | 0.09 |  | 1.05× vs ic |
| h2h_blosc2_10mb/encode | 4.30 | 4.27 | 4.50 | 0.04 |  | 11.84× vs ic |
| h2h_blosc2_10mb/decode | 0.24 | 0.23 | 0.24 | 0.00 | ⚠️ | 29.11× vs ic |
| h2h_jpeg_4mp_rgb/encode | 32.04 | 31.94 | 32.13 | 0.07 |  | 0.55× vs ic |
| h2h_jpeg_4mp_rgb/decode | 63.28 | 56.82 | 67.06 | 4.08 |  | 0.46× vs ic |
| h2h_png_4mp_rgb/encode | 320.56 | 317.23 | 327.80 | 4.45 |  | 0.84× vs ic |
| h2h_png_4mp_rgb/decode | 6.80 | 6.76 | 6.89 | 0.05 |  | 1.62× vs ic |
| h2h_webp_4mp_rgb/encode | 573.02 | 572.64 | 573.32 | 0.31 |  | 0.91× vs ic |
| h2h_webp_4mp_rgb/decode | 142.51 | 141.63 | 145.51 | 1.70 |  | 0.50× vs ic |
| h2h_jpeg2k_4mp_u16/encode | 690.62 | 689.40 | 694.67 | 1.64 |  | 0.97× vs ic |
| h2h_jpeg2k_4mp_u16/decode | 102.68 | 102.13 | 103.15 | 0.50 |  | 6.49× vs ic |
| h2h_qoi_4mp_rgb/encode | 16.41 | 16.26 | 16.48 | 0.07 |  | 0.96× vs ic |
| h2h_qoi_4mp_rgb/decode | 7.94 | 7.91 | 8.00 | 0.07 |  | 0.93× vs ic |
| h2h_lerc_4mp_u16 | skipped: lerc static-library symbol clash with imagecodecs (cross-process bench TODO) | — | — | — | — | — |
| h2h_jxl_4mp_rgb/encode | 155.14 | 154.06 | 156.34 | 1.21 |  | 19.34× vs ic |
| h2h_jxl_4mp_rgb/decode | 32.45 | 30.40 | 35.06 | 0.94 |  | 15.34× vs ic |
