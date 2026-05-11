# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T071828Z`
- opencodecs: `0.2.0.dev0` (git: `5a97730`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| ndtiff_write_compressed_zstd | 8.00 | 7.94 | 8.10 | 0.14 |  | 1.91× of uncompressed |
| ndtiff_write_compressed_zstd_large | 161.37 | 158.98 | 163.77 | 1.63 |  | 3.51× |
