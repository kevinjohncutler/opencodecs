# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T085359Z`
- opencodecs: `0.2.0.dev0` (git: `5272e16`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_zstd_10mb/encode | 10.46 | 10.25 | 10.67 | 0.23 |  | 0.98× vs ic |
| h2h_zstd_10mb/decode | 9.46 | 9.27 | 9.62 | 0.15 |  | 1.00× vs ic |
| h2h_deflate_10mb/encode | 280.86 | 280.31 | 281.00 | 0.17 |  | 0.99× vs ic |
| h2h_deflate_10mb/decode | 24.51 | 24.31 | 24.60 | 0.23 |  | 1.72× vs ic |
| h2h_lz4_10mb/encode | 2.72 | 2.63 | 2.81 | 0.09 |  | 1.03× vs ic |
| h2h_lz4_10mb/decode | 0.29 | 0.25 | 0.32 | 0.07 | ⚠️ | 1.03× vs ic |
| h2h_brotli_10mb/encode | 60.02 | 57.62 | 63.06 | 5.14 |  | 0.95× vs ic |
| h2h_brotli_10mb/decode | 49.11 | 48.38 | 50.51 | 1.50 |  | 1.03× vs ic |
| h2h_blosc2_10mb/encode | 4.63 | 4.44 | 4.69 | 0.11 |  | 11.37× vs ic |
| h2h_blosc2_10mb/decode | 0.29 | 0.25 | 0.35 | 0.05 | ⚠️ | 24.77× vs ic |
| h2h_jpeg_4mp_rgb/encode | 16.81 | 16.51 | 17.49 | 0.46 |  | 1.03× vs ic |
| h2h_jpeg_4mp_rgb/decode | 29.36 | 29.10 | 29.82 | 0.37 |  | 1.00× vs ic |
| h2h_png_4mp_rgb/encode | 277.76 | 277.49 | 277.88 | 0.00 |  | 1.00× vs ic |
| h2h_png_4mp_rgb/decode | 4.53 | 4.41 | 4.64 | 0.18 |  | 2.42× vs ic |
| h2h_webp_4mp_rgb/encode | 594.53 | 587.63 | 596.14 | 6.13 |  | 0.94× vs ic |
| h2h_webp_4mp_rgb/decode | 142.68 | 142.18 | 142.72 | 0.00 |  | 0.98× vs ic |
| h2h_jpeg2k_4mp_u16/encode | 695.08 | 693.27 | 698.18 | 3.36 |  | 0.98× vs ic |
| h2h_jpeg2k_4mp_u16/decode | 106.09 | 105.99 | 106.38 | 0.00 |  | 6.33× vs ic |
| h2h_qoi_4mp_rgb/encode | 16.48 | 16.46 | 16.52 | 0.04 |  | 0.97× vs ic |
| h2h_qoi_4mp_rgb/decode | 7.66 | 7.55 | 7.73 | 0.07 |  | 0.93× vs ic |
| h2h_lerc_4mp_u16 | skipped: lerc static-library symbol clash with imagecodecs (cross-process bench TODO) | — | — | — | — | — |
| h2h_jxl_4mp_rgb/encode | 119.29 | 112.50 | 123.78 | 6.75 |  | 25.95× vs ic |
| h2h_jxl_4mp_rgb/decode | 28.66 | 27.33 | 29.77 | 0.95 |  | 17.38× vs ic |
