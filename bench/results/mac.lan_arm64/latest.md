# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T090427Z`
- opencodecs: `0.2.0.dev0` (git: `a18b20d`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| tiff_write_64mb | 4.58 | 4.54 | 4.65 | 0.10 |  | 1.10× |
