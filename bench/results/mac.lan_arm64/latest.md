# opencodecs bench — mac.lan (arm64)

- Run at: `20260514T062825Z`
- opencodecs: `0.2.0.dev0` (git: `9fa1059`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_lz4_10mb/encode | 2.78 | 2.76 | 2.90 | 0.02 |  | 1.01× vs ic |
| h2h_lz4_10mb/decode | 0.27 | 0.25 | 0.31 | 0.02 | ⚠️ | 1.02× vs ic |
