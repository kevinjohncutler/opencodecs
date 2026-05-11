# opencodecs bench — threadripper (x86_64)

- Run at: `20260511T081827Z`
- opencodecs: `0.2.0.dev0` (git: `f040fbd`)
- Python: 3.12.13, CPU: AMD Ryzen Threadripper PRO 3995WX 64-Cores × 128
- Reference libraries: tifffile 2026.5.2, ndstorage 0.1.18, czifile 2026.4.30, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_zstd_10mb/encode | 12.59 | 12.48 | 12.64 | 0.13 |  | 1.02× vs ic |
| h2h_zstd_10mb/decode | 9.67 | 9.30 | 9.90 | 0.24 |  | 1.02× vs ic |
| h2h_deflate_10mb/encode | 331.28 | 328.41 | 333.15 | 2.90 |  | 1.00× vs ic |
| h2h_deflate_10mb/decode | 46.23 | 45.75 | 46.35 | 0.39 |  | 0.85× vs ic |
| h2h_lz4_10mb/encode | 3.97 | 3.96 | 4.00 | 0.01 |  | 1.00× vs ic |
| h2h_lz4_10mb/decode | 1.16 | 1.13 | 1.17 | 0.01 |  | 1.00× vs ic |
| h2h_brotli_10mb/encode | 93.24 | 84.17 | 95.59 | 8.21 |  | 0.92× vs ic |
| h2h_brotli_10mb/decode | 51.28 | 50.81 | 52.23 | 0.98 |  | 0.91× vs ic |
| h2h_blosc2_10mb/encode | 6.27 | 6.20 | 6.32 | 0.07 |  | 11.00× vs ic |
| h2h_blosc2_10mb/decode | 1.27 | 1.23 | 1.45 | 0.08 |  | 6.27× vs ic |
| h2h_jpeg_4mp_rgb/encode | 22.16 | 21.74 | 22.81 | 0.88 |  | 1.02× vs ic |
| h2h_jpeg_4mp_rgb/decode | 29.64 | 28.64 | 29.98 | 0.46 |  | 1.03× vs ic |
| h2h_png_4mp_rgb/encode | 439.22 | 438.76 | 440.60 | 1.42 |  | 0.86× vs ic |
| h2h_png_4mp_rgb/decode | 14.85 | 14.17 | 16.00 | 0.62 |  | 1.40× vs ic |
| h2h_webp_4mp_rgb/encode | 627.68 | 625.91 | 631.08 | 0.89 |  | 1.01× vs ic |
| h2h_webp_4mp_rgb/decode | 163.74 | 162.77 | 164.08 | 0.45 |  | 0.99× vs ic |
| h2h_jpeg2k_4mp_u16/encode | 913.80 | 910.77 | 917.78 | 2.38 |  | 0.99× vs ic |
| h2h_jpeg2k_4mp_u16/decode | 120.47 | 119.73 | 121.12 | 0.61 |  | 5.86× vs ic |
| h2h_qoi_4mp_rgb/encode | 25.86 | 25.62 | 26.04 | 0.25 |  | 0.87× vs ic |
| h2h_qoi_4mp_rgb/decode | 14.29 | 14.06 | 14.60 | 0.53 |  | 0.94× vs ic |
| h2h_lerc_4mp_u16 | skipped: lerc static-library symbol clash with imagecodecs (cross-process bench TODO) | — | — | — | — | — |
| h2h_jxl_4mp_rgb/encode | 143.30 | 142.51 | 144.10 | 0.00 |  | 32.49× vs ic |
| h2h_jxl_4mp_rgb/decode | 40.68 | 40.30 | 41.66 | 1.03 |  | 22.15× vs ic |
