# A/B report `2026-10-13-ab-opts1-rf`

Baseline binary: `C:\Users\adame\tidyrustmodels-baseline\benchmarks\rust\target\release\trm-bench.exe`; candidate binary: `C:\Users\adame\tidyrustmodels\benchmarks\rust\target\release\trm-bench.exe`; git commit of the candidate checkout: `29ba04af5305ca3e2b5c433798904eb3b8a365ec`; trial order: A-B / B-A alternating; protocol: {'trials': 6, 'warm': 5, 'v': 10, 'tune_levels': 20, 'threads': '1', 'timeout': 600.0}.

Ratio = candidate / baseline median (< 1: candidate faster or smaller). IQR is relative to the median. Drift = every fingerprint identical across all trials of both binaries (bitwise-equal values).

## ames_rf_reg (2930 rows, rand_forest, 1 thread) — ok trials baseline 6/6, candidate 6/6; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 436 ms (IQR 23%, n=6) | 424 ms (IQR 15%, n=6) | 0.974 | within noise band (inside IQR) |
| workflow fit, warm median | 583 ms (IQR 42%, n=6) | 515 ms (IQR 13%, n=6) | 0.884 | faster (inside IQR) |
| prep + bake, warm median | 795 µs (IQR 16%, n=6) | 397 µs (IQR 13%, n=6) | 0.500 | faster |
| model fit, warm median | 606 ms (IQR 28%, n=6) | 484 ms (IQR 9%, n=6) | 0.798 | faster (inside IQR) |
| predict | 16.2 ms (IQR 13%, n=6) | 17.6 ms (IQR 25%, n=6) | 1.085 | **regression** (inside IQR) |
| resampling (cv) | 6.96 s (IQR 26%, n=6) | 7.01 s (IQR 7%, n=6) | 1.007 | within noise band (inside IQR) |
| wall (cold process) | 13.3 s (IQR 25%, n=6) | 13 s (IQR 3%, n=6) | 0.975 | within noise band (inside IQR) |
| peak working set MB | 40.6 MB (n=6) | 40.7 MB (n=6) | 1.003 | within noise band |
| allocations wf_fit (count) | 579,946 (n=6) | 579,808 (n=6) | 1.000 | within noise band |
| allocations wf_fit (MB) | 347 MB (n=6) | 346 MB (n=6) | 0.997 | within noise band |
| allocations cv (count) | 6,544,970 (n=6) | 6,542,850 (n=6) | 1.000 | within noise band |
| allocations cv (MB) | 4.29e+03 MB (n=6) | 4.28e+03 MB (n=6) | 0.997 | within noise band |

## synth_10000_rf_clf (10000 rows, rand_forest, 1 thread) — ok trials baseline 6/6, candidate 6/6; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 3.74 s (IQR 9%, n=6) | 3.98 s (IQR 20%, n=6) | 1.064 | **regression** (inside IQR) |
| workflow fit, warm median | 3.4 s (IQR 29%, n=6) | 3.77 s (IQR 19%, n=6) | 1.110 | **regression** (inside IQR) |
| prep + bake, warm median | 3.33 ms (IQR 26%, n=6) | 3.27 ms (IQR 28%, n=6) | 0.982 | within noise band (inside IQR) |
| model fit, warm median | 3.29 s (IQR 35%, n=6) | 3.78 s (IQR 17%, n=6) | 1.147 | **regression** (inside IQR) |
| predict | 167 ms (IQR 66%, n=6) | 210 ms (IQR 54%, n=6) | 1.258 | **regression** (inside IQR) |
| resampling (cv) | 42.4 s (IQR 33%, n=6) | 46.6 s (IQR 25%, n=6) | 1.099 | **regression** (inside IQR) |
| wall (cold process) | 82.7 s (IQR 28%, n=6) | 93.6 s (IQR 21%, n=6) | 1.131 | **regression** (inside IQR) |
| peak working set MB | 154 MB (n=6) | 154 MB (n=6) | 0.999 | within noise band |
| allocations wf_fit (count) | 1,985,410 (n=6) | 1,985,278 (n=6) | 1.000 | within noise band |
| allocations wf_fit (MB) | 1.11e+03 MB (n=6) | 1.11e+03 MB (n=6) | 0.997 | within noise band |
| allocations cv (count) | 22,947,577 (n=6) | 22,945,667 (n=6) | 1.000 | within noise band |
| allocations cv (MB) | 1.35e+04 MB (n=6) | 1.35e+04 MB (n=6) | 0.997 | within noise band |

## Summary of phases with a decision

| task | threads | phase | ratio | verdict |
|---|---|---|---|---|
| ames_rf_reg | 1 | workflow fit, cold | 0.974 | within noise band (inside IQR) |
| ames_rf_reg | 1 | workflow fit, warm median | 0.884 | faster (inside IQR) |
| ames_rf_reg | 1 | prep + bake, warm median | 0.500 | faster |
| ames_rf_reg | 1 | model fit, warm median | 0.798 | faster (inside IQR) |
| ames_rf_reg | 1 | predict | 1.085 | **regression** (inside IQR) |
| ames_rf_reg | 1 | resampling (cv) | 1.007 | within noise band (inside IQR) |
| ames_rf_reg | 1 | peak working set MB | 1.003 | within noise band |
| synth_10000_rf_clf | 1 | workflow fit, cold | 1.064 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | workflow fit, warm median | 1.110 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | prep + bake, warm median | 0.982 | within noise band (inside IQR) |
| synth_10000_rf_clf | 1 | model fit, warm median | 1.147 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | predict | 1.258 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | resampling (cv) | 1.099 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | peak working set MB | 0.999 | within noise band |
