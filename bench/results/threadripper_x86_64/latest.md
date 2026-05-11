# opencodecs bench — threadripper (x86_64)

- Run at: `20260511T085600Z`
- opencodecs: `0.2.0.dev0` (git: `dbbcb2e`)
- Python: 3.12.13, CPU: AMD Ryzen Threadripper PRO 3995WX 64-Cores × 128
- Reference libraries: tifffile 2026.5.2, ndstorage 0.1.18, czifile 2026.4.30, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_zstd_10mb/encode | 13.32 | 12.50 | 14.28 | 0.78 |  | 1.04× vs ic |
| h2h_zstd_10mb/decode | 10.08 | 9.50 | 10.79 | 0.83 |  | 0.96× vs ic |
| h2h_deflate_10mb/encode | 331.43 | 328.50 | 332.44 | 2.29 |  | 1.00× vs ic |
| h2h_deflate_10mb/decode | 45.92 | 45.84 | 46.09 | 0.10 |  | 0.86× vs ic |
| h2h_lz4_10mb/encode | 3.97 | 3.89 | 4.05 | 0.08 |  | 1.16× vs ic |
| h2h_lz4_10mb/decode | 1.49 | 1.41 | 1.53 | 0.09 |  | 1.00× vs ic |
| h2h_brotli_10mb/encode | 87.44 | 81.52 | 96.79 | 5.97 |  | 0.99× vs ic |
| h2h_brotli_10mb/decode | 51.66 | 51.45 | 51.98 | 0.24 |  | 0.89× vs ic |
| h2h_blosc2_10mb/encode | 6.50 | 6.46 | 6.58 | 0.08 |  | 10.53× vs ic |
| h2h_blosc2_10mb/decode | 1.26 | 1.25 | 1.27 | 0.01 |  | 6.41× vs ic |
| h2h_jpeg_4mp_rgb/encode | 22.34 | 22.07 | 22.60 | 0.31 |  | 0.99× vs ic |
| h2h_jpeg_4mp_rgb/decode | 29.43 | 29.37 | 29.54 | 0.13 |  | 1.04× vs ic |
| h2h_png_4mp_rgb/encode | 327.89 | 326.49 | 328.63 | 0.84 |  | 1.16× vs ic |
| h2h_png_4mp_rgb/decode | 12.87 | 12.58 | 13.19 | 0.11 |  | 1.61× vs ic |
| h2h_webp_4mp_rgb/encode | 630.32 | 626.86 | 632.19 | 2.20 |  | 0.99× vs ic |
| h2h_webp_4mp_rgb/decode | 163.23 | 162.46 | 163.59 | 0.83 |  | 1.00× vs ic |
| h2h_jpeg2k_4mp_u16/encode | 915.44 | 914.62 | 917.15 | 1.95 |  | 0.98× vs ic |
| h2h_jpeg2k_4mp_u16/decode | 121.36 | 120.69 | 122.71 | 0.66 |  | 5.83× vs ic |
| h2h_qoi_4mp_rgb/encode | 25.62 | 25.53 | 25.79 | 0.18 |  | 0.88× vs ic |
| h2h_qoi_4mp_rgb/decode | 13.95 | 13.88 | 14.14 | 0.23 |  | 0.98× vs ic |
| h2h_lerc_4mp_u16 | skipped: lerc static-library symbol clash with imagecodecs (cross-process bench TODO) | — | — | — | — | — |
| h2h_jxl_4mp_rgb/encode | 217.85 | 181.67 | 226.03 | 12.59 |  | 21.54× vs ic |
| h2h_jxl_4mp_rgb/decode | 42.26 | 40.07 | 46.53 | 4.56 |  | 21.29× vs ic |
