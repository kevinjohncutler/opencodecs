# opencodecs bench — threadripper (x86_64)

- Run at: `20260511T074352Z`
- opencodecs: `0.2.0.dev0` (git: `acfb2dc`)
- Python: 3.12.11, CPU: AMD Ryzen Threadripper PRO 3995WX 64-Cores × 128
- Reference libraries: tifffile 2026.5.2, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_zstd_10mb/encode | 13.40 | 12.67 | 13.95 | 0.82 |  | 0.98× vs ic |
| h2h_zstd_10mb/decode | 9.76 | 9.56 | 10.10 | 0.47 |  | 0.99× vs ic |
| h2h_deflate_10mb/encode | 313.18 | 311.25 | 318.39 | 3.01 |  | 1.00× vs ic |
| h2h_deflate_10mb/decode | 46.05 | 45.84 | 46.19 | 0.09 |  | 0.87× vs ic |
| h2h_lz4_10mb/encode | 4.59 | 4.46 | 4.64 | 0.13 |  | 0.99× vs ic |
| h2h_lz4_10mb/decode | 3.56 | 3.47 | 3.68 | 0.09 |  | 0.32× vs ic |
| h2h_brotli_10mb/encode | 87.57 | 87.23 | 88.37 | 0.82 |  | 1.01× vs ic |
| h2h_brotli_10mb/decode | 51.26 | 51.00 | 51.59 | 0.15 |  | 0.90× vs ic |
| h2h_blosc2_10mb/encode | 6.45 | 6.42 | 6.46 | 0.02 |  | 10.45× vs ic |
| h2h_blosc2_10mb/decode | 1.26 | 1.23 | 1.31 | 0.03 |  | 6.45× vs ic |
| h2h_jpeg_4mp_rgb/encode | 43.82 | 43.55 | 43.97 | 0.21 |  | 0.50× vs ic |
| h2h_jpeg_4mp_rgb/decode | 58.16 | 57.99 | 58.20 | 0.07 |  | 0.52× vs ic |
| h2h_png_4mp_rgb/encode | 435.18 | 431.96 | 440.04 | 3.04 |  | 0.86× vs ic |
| h2h_png_4mp_rgb/decode | 12.98 | 12.95 | 12.99 | 0.00 |  | 0.94× vs ic |
| h2h_webp_4mp_rgb/encode | 597.34 | 591.87 | 602.28 | 3.56 |  | 1.24× vs ic |
| h2h_webp_4mp_rgb/decode | 164.38 | 164.10 | 164.58 | 0.36 |  | 0.32× vs ic |
| h2h_jpeg2k_4mp_u16/encode | 894.93 | 892.71 | 897.86 | 2.97 |  | 0.93× vs ic |
| h2h_jpeg2k_4mp_u16/decode | 122.37 | 122.29 | 122.40 | 0.00 |  | 5.70× vs ic |
| h2h_qoi_4mp_rgb/encode | 25.93 | 25.77 | 26.08 | 0.17 |  | 0.84× vs ic |
| h2h_qoi_4mp_rgb/decode | 14.41 | 13.90 | 14.60 | 0.48 |  | 0.95× vs ic |
| h2h_lerc_4mp_u16 | skipped: lerc static-library symbol clash with imagecodecs (cross-process bench TODO) | — | — | — | — | — |
| h2h_jxl_4mp_rgb/encode | 182.89 | 180.18 | 184.83 | 0.00 |  | 25.44× vs ic |
| h2h_jxl_4mp_rgb/decode | 30.84 | 30.09 | 31.95 | 1.42 |  | 28.38× vs ic |
