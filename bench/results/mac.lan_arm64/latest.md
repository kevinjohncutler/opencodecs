# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T233801Z`
- opencodecs: `0.2.0.dev0` (git: `d90a5db`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_deflate_10mb/encode | 162.12 | 161.64 | 162.32 | 0.31 |  | 1.81× vs ic |
| h2h_deflate_10mb/decode | 35.12 | 35.05 | 35.30 | 0.21 |  | 1.25× vs ic |
