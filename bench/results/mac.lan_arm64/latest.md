# opencodecs bench — mac.lan (arm64)

- Run at: `20260511T225944Z`
- opencodecs: `0.2.0.dev0` (git: `46cca77`)
- Python: 3.12.9, CPU: Apple M1 Ultra × 20
- Reference libraries: tifffile 2026.3.3, ndstorage 0.1.18, czifile 2026.4.11, imagecodecs 2026.3.6

## Workloads (tier: fast)

| Workload | median (ms) | min | max | IQR | noisy | ratio |
|---|---:|---:|---:|---:|:-:|---:|
| h2h_lerc_4mp_u16/encode | 47.89 | 46.70 | 48.50 | 0.74 |  | 0.95× vs ic |
| h2h_lerc_4mp_u16/decode | 11.63 | 11.37 | 11.90 | 0.28 |  | 0.86× vs ic |
