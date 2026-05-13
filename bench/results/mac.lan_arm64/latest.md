# opencodecs bench — mac.lan (arm64)

- Run at: `20260513T095534Z`
- opencodecs: `0.2.0.dev0` (git: `ebbbb62`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: medium)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| tiff_write_1gb | 90.90 | 85.04 | 92.09 | 0.00 |  | 0.77× |
| ndtiff_write_1gb | 222.75 | 161.33 | 247.59 | 72.95 | ⚠️ | 0.91× |
