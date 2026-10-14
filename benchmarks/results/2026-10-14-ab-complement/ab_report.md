# A/B report `2026-10-14-ab-complement`

Baseline binary: `C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-tidyrustmodels\521b1036-120d-477b-9c9e-583ed06ba543\scratchpad\trm-bench-prefix2.exe`; candidate binary: `C:\Users\adame\tidyrustmodels\benchmarks\rust\target\release\trm-bench.exe`; git commit of the candidate checkout: `23e54551de9880c4cc909dae665d5aad56b811de`; trial order: A-B / B-A alternating; protocol: {'trials': 6, 'warm': 5, 'v': 10, 'tune_levels': 20, 'threads': '1', 'timeout': 600.0}.

Ratio = candidate / baseline median (< 1: candidate faster or smaller). IQR is relative to the median. Drift = every fingerprint identical across all trials of both binaries (bitwise-equal values).

## ames_reg (2930 rows, linear_reg, 1 thread) — ok trials baseline 6/6, candidate 6/6; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 2.54 ms (IQR 9%, n=6) | 2.6 ms (IQR 4%, n=6) | 1.022 | within noise band (inside IQR) |
| workflow fit, warm median | 2.54 ms (IQR 12%, n=6) | 2.7 ms (IQR 9%, n=6) | 1.065 | **regression** (inside IQR) |
| prep + bake, warm median | 625 µs (IQR 16%, n=6) | 597 µs (IQR 24%, n=6) | 0.955 | within noise band (inside IQR) |
| model fit, warm median | 1.86 ms (IQR 7%, n=6) | 2.02 ms (IQR 6%, n=6) | 1.083 | **regression** |
| predict | 9.5 µs (IQR 11%, n=6) | 9.15 µs (IQR 38%, n=6) | 0.963 | within noise band (inside IQR) |
| resampling (cv) | 33.4 ms (IQR 6%, n=6) | 33 ms (IQR 2%, n=6) | 0.989 | within noise band (inside IQR) |
| wall (cold process) | 84 ms (IQR 4%, n=6) | 83.1 ms (IQR 5%, n=6) | 0.989 | within noise band (inside IQR) |
| peak working set MB | 9 MB (n=6) | 9.12 MB (n=6) | 1.013 | within noise band |
| allocations wf_fit (count) | 679 (n=6) | 679 (n=6) | 1.000 | within noise band |
| allocations wf_fit (MB) | 2.96 MB (n=6) | 2.96 MB (n=6) | 1.000 | within noise band |
| allocations cv (count) | 9,747 (n=6) | 9,747 (n=6) | 1.000 | within noise band |
| allocations cv (MB) | 38.3 MB (n=6) | 38.3 MB (n=6) | 1.000 | within noise band |

## synth_100000_reg (100000 rows, linear_reg, 1 thread) — ok trials baseline 6/6, candidate 6/6; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 35.5 ms (IQR 11%, n=6) | 34.4 ms (IQR 5%, n=6) | 0.967 | within noise band (inside IQR) |
| workflow fit, warm median | 35.8 ms (IQR 17%, n=6) | 34.6 ms (IQR 4%, n=6) | 0.967 | within noise band (inside IQR) |
| prep + bake, warm median | 21.4 ms (IQR 36%, n=6) | 18 ms (IQR 2%, n=6) | 0.839 | faster (inside IQR) |
| model fit, warm median | 20.4 ms (IQR 26%, n=6) | 16.7 ms (IQR 7%, n=6) | 0.822 | faster (inside IQR) |
| predict | 300 µs (IQR 20%, n=6) | 226 µs (IQR 65%, n=6) | 0.752 | faster (inside IQR) |
| resampling (cv) | 505 ms (IQR 11%, n=6) | 465 ms (IQR 2%, n=6) | 0.921 | faster (inside IQR) |
| wall (cold process) | 3.24 s (IQR 3%, n=6) | 1.09 s (IQR 5%, n=6) | 0.335 | faster |
| peak working set MB | 75.3 MB (n=6) | 74.9 MB (n=6) | 0.995 | within noise band |
| allocations wf_fit (count) | 465 (n=6) | 465 (n=6) | 1.000 | within noise band |
| allocations wf_fit (MB) | 56.4 MB (n=6) | 56.4 MB (n=6) | 1.000 | within noise band |
| allocations cv (count) | 6,486 (n=6) | 6,486 (n=6) | 1.000 | within noise band |
| allocations cv (MB) | 731 MB (n=6) | 731 MB (n=6) | 1.000 | within noise band |

## Summary of phases with a decision

| task | threads | phase | ratio | verdict |
|---|---|---|---|---|
| ames_reg | 1 | workflow fit, cold | 1.022 | within noise band (inside IQR) |
| ames_reg | 1 | workflow fit, warm median | 1.065 | **regression** (inside IQR) |
| ames_reg | 1 | prep + bake, warm median | 0.955 | within noise band (inside IQR) |
| ames_reg | 1 | model fit, warm median | 1.083 | **regression** |
| ames_reg | 1 | predict | 0.963 | within noise band (inside IQR) |
| ames_reg | 1 | resampling (cv) | 0.989 | within noise band (inside IQR) |
| ames_reg | 1 | peak working set MB | 1.013 | within noise band |
| synth_100000_reg | 1 | workflow fit, cold | 0.967 | within noise band (inside IQR) |
| synth_100000_reg | 1 | workflow fit, warm median | 0.967 | within noise band (inside IQR) |
| synth_100000_reg | 1 | prep + bake, warm median | 0.839 | faster (inside IQR) |
| synth_100000_reg | 1 | model fit, warm median | 0.822 | faster (inside IQR) |
| synth_100000_reg | 1 | predict | 0.752 | faster (inside IQR) |
| synth_100000_reg | 1 | resampling (cv) | 0.921 | faster (inside IQR) |
| synth_100000_reg | 1 | peak working set MB | 0.995 | within noise band |
