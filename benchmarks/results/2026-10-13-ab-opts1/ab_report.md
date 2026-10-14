# A/B report `2026-10-13-ab-opts1`

Baseline binary: `C:\Users\adame\tidyrustmodels-baseline\benchmarks\rust\target\release\trm-bench.exe`; candidate binary: `C:\Users\adame\tidyrustmodels\benchmarks\rust\target\release\trm-bench.exe`; git commit of the candidate checkout: `29ba04af5305ca3e2b5c433798904eb3b8a365ec`; trial order: A-B / B-A alternating; protocol: {'trials': 5, 'warm': 30, 'v': 10, 'tune_levels': 20, 'threads': '4', 'timeout': 600.0}.

Ratio = candidate / baseline median (< 1: candidate faster or smaller). IQR is relative to the median. Drift = every fingerprint identical across all trials of both binaries (bitwise-equal values).

## ames_reg (2930 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 5.09 ms (IQR 11%, n=5) | 5.02 ms (IQR 14%, n=5) | 0.987 | within noise band (inside IQR) |
| workflow fit, warm median | 5.34 ms (IQR 2%, n=5) | 4.61 ms (IQR 2%, n=5) | 0.863 | faster |
| prep + bake, warm median | 1.75 ms (IQR 2%, n=5) | 1.15 ms (IQR 10%, n=5) | 0.656 | faster |
| model fit, warm median | 3.48 ms (IQR 3%, n=5) | 3.46 ms (IQR 4%, n=5) | 0.994 | within noise band (inside IQR) |
| predict | 22.6 µs (IQR 27%, n=5) | 24.9 µs (IQR 33%, n=5) | 1.102 | **regression** (inside IQR) |
| resampling (cv) | 67 ms (IQR 5%, n=5) | 58.6 ms (IQR 5%, n=5) | 0.875 | faster |
| wall (cold process) | 430 ms (IQR 3%, n=5) | 381 ms (IQR 9%, n=5) | 0.886 | faster |
| peak working set MB | 9.2 MB (n=5) | 9.15 MB (n=5) | 0.995 | within noise band |
| allocations wf_fit (count) | 817 (n=5) | 679 (n=5) | 0.831 | smaller |
| allocations wf_fit (MB) | 4.01 MB (n=5) | 2.96 MB (n=5) | 0.737 | smaller |
| allocations cv (count) | 11,867 (n=5) | 9,747 (n=5) | 0.821 | smaller |
| allocations cv (MB) | 52.1 MB (n=5) | 38.3 MB (n=5) | 0.735 | smaller |

## ames_tree_reg (2930 rows, decision_tree, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 8.06 ms (IQR 22%, n=5) | 7.75 ms (IQR 26%, n=5) | 0.961 | within noise band (inside IQR) |
| workflow fit, warm median | 8.25 ms (IQR 19%, n=5) | 7.31 ms (IQR 6%, n=5) | 0.885 | faster (inside IQR) |
| prep + bake, warm median | 1.42 ms (IQR 53%, n=5) | 928 µs (IQR 10%, n=5) | 0.652 | faster (inside IQR) |
| model fit, warm median | 6.91 ms (IQR 11%, n=5) | 6.39 ms (IQR 5%, n=5) | 0.925 | faster (inside IQR) |
| predict | 29.1 µs (IQR 14%, n=5) | 30.7 µs (IQR 12%, n=5) | 1.055 | **regression** (inside IQR) |
| resampling (cv) | 103 ms (IQR 5%, n=5) | 104 ms (IQR 13%, n=5) | 1.010 | within noise band (inside IQR) |
| tune | 2.6 s (IQR 11%, n=5) | 2.19 s (IQR 17%, n=5) | 0.844 | faster |
| select + last_fit | 14.7 ms (IQR 14%, n=5) | 16.2 ms (IQR 24%, n=5) | 1.099 | **regression** (inside IQR) |
| wall (cold process) | 3.28 s (IQR 13%, n=5) | 2.82 s (IQR 15%, n=5) | 0.860 | faster |
| peak working set MB | 9.42 MB (n=5) | 8.71 MB (n=5) | 0.925 | smaller |
| allocations wf_fit (count) | 895 (n=5) | 757 (n=5) | 0.846 | smaller |
| allocations wf_fit (MB) | 4.14 MB (n=5) | 3.08 MB (n=5) | 0.745 | smaller |
| allocations cv (count) | 12,626 (n=5) | 10,506 (n=5) | 0.832 | smaller |
| allocations cv (MB) | 53.6 MB (n=5) | 39.8 MB (n=5) | 0.743 | smaller |

## ames_tree_reg (2930 rows, decision_tree, 4 threads) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 6.11 ms (IQR 18%, n=5) | 5.18 ms (IQR 22%, n=5) | 0.848 | faster (inside IQR) |
| workflow fit, warm median | 4.72 ms (IQR 4%, n=5) | 4.62 ms (IQR 17%, n=5) | 0.979 | within noise band (inside IQR) |
| prep + bake, warm median | 358 µs (IQR 17%, n=5) | 269 µs (IQR 13%, n=5) | 0.751 | faster |
| model fit, warm median | 4.19 ms (IQR 10%, n=5) | 4.29 ms (IQR 25%, n=5) | 1.023 | within noise band (inside IQR) |
| predict | 24.1 µs (IQR 24%, n=5) | 19.5 µs (IQR 28%, n=5) | 0.809 | faster (inside IQR) |
| resampling (cv) | 30.2 ms (IQR 23%, n=5) | 34.6 ms (IQR 12%, n=5) | 1.143 | **regression** (inside IQR) |
| tune | 654 ms (IQR 25%, n=5) | 515 ms (IQR 11%, n=5) | 0.787 | faster (inside IQR) |
| select + last_fit | 10.3 ms (IQR 49%, n=5) | 10.4 ms (IQR 34%, n=5) | 1.006 | within noise band (inside IQR) |
| wall (cold process) | 1.02 s (IQR 9%, n=5) | 891 ms (IQR 14%, n=5) | 0.872 | faster |
| peak working set MB | 17 MB (n=5) | 16.8 MB (n=5) | 0.987 | within noise band |
| allocations wf_fit (count) | 895 (n=5) | 757 (n=5) | 0.846 | smaller |
| allocations wf_fit (MB) | 4.14 MB (n=5) | 3.08 MB (n=5) | 0.745 | smaller |
| allocations cv (count) | 12,680 (n=5) | 10,560 (n=5) | 0.833 | smaller |
| allocations cv (MB) | 53.7 MB (n=5) | 39.9 MB (n=5) | 0.743 | smaller |

## ames_xgb_reg (2930 rows, boost_tree, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 42.8 ms (IQR 12%, n=5) | 38.9 ms (IQR 15%, n=5) | 0.909 | faster (inside IQR) |
| workflow fit, warm median | 49.3 ms (IQR 5%, n=5) | 46.6 ms (IQR 3%, n=5) | 0.945 | faster |
| prep + bake, warm median | 1.74 ms (IQR 13%, n=5) | 1.05 ms (IQR 9%, n=5) | 0.603 | faster |
| model fit, warm median | 46.3 ms (IQR 13%, n=5) | 47.2 ms (IQR 2%, n=5) | 1.018 | within noise band (inside IQR) |
| predict | 1.85 ms (IQR 13%, n=5) | 1.8 ms (IQR 8%, n=5) | 0.974 | within noise band (inside IQR) |
| resampling (cv) | 477 ms (IQR 16%, n=5) | 431 ms (IQR 27%, n=5) | 0.905 | faster (inside IQR) |
| tune | 11.9 s (IQR 3%, n=5) | 11.9 s (IQR 3%, n=5) | 1.005 | within noise band (inside IQR) |
| select + last_fit | 63.5 ms (IQR 13%, n=5) | 63.8 ms (IQR 8%, n=5) | 1.005 | within noise band (inside IQR) |
| wall (cold process) | 15.7 s (IQR 3%, n=5) | 15.5 s (IQR 4%, n=5) | 0.989 | within noise band (inside IQR) |
| peak working set MB | 14.9 MB (n=5) | 14.2 MB (n=5) | 0.949 | smaller |
| allocations wf_fit (count) | 21,932 (n=5) | 21,794 (n=5) | 0.994 | within noise band |
| allocations wf_fit (MB) | 43.8 MB (n=5) | 42.8 MB (n=5) | 0.976 | within noise band |
| allocations cv (count) | 243,717 (n=5) | 241,597 (n=5) | 0.991 | within noise band |
| allocations cv (MB) | 536 MB (n=5) | 522 MB (n=5) | 0.974 | within noise band |

## penguins_reg (342 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 164 µs (IQR 15%, n=5) | 151 µs (IQR 16%, n=5) | 0.922 | faster (inside IQR) |
| workflow fit, warm median | 81.7 µs (IQR 43%, n=5) | 93.4 µs (IQR 28%, n=5) | 1.145 | **regression** (inside IQR) |
| prep + bake, warm median | 53.4 µs (IQR 57%, n=5) | 61.8 µs (IQR 40%, n=5) | 1.156 | **regression** (inside IQR) |
| model fit, warm median | 27.6 µs (IQR 20%, n=5) | 33.2 µs (IQR 17%, n=5) | 1.205 | **regression** (inside IQR) |
| predict | 4.7 µs (IQR 30%, n=5) | 3.3 µs (IQR 18%, n=5) | 0.702 | faster (inside IQR) |
| resampling (cv) | 2.32 ms (IQR 33%, n=5) | 2.17 ms (IQR 7%, n=5) | 0.936 | faster (inside IQR) |
| wall (cold process) | 37.8 ms (IQR 4%, n=5) | 37.9 ms (IQR 2%, n=5) | 1.003 | within noise band (inside IQR) |
| peak working set MB | 5.31 MB (n=5) | 5.29 MB (n=5) | 0.997 | within noise band |
| allocations wf_fit (count) | 375 (n=5) | 290 (n=5) | 0.773 | smaller |
| allocations wf_fit (MB) | 0.189 MB (n=5) | 0.0977 MB (n=5) | 0.516 | smaller |
| allocations cv (count) | 5,446 (n=5) | 4,256 (n=5) | 0.781 | smaller |
| allocations cv (MB) | 2.65 MB (n=5) | 1.47 MB (n=5) | 0.555 | smaller |

## penguins_tree_clf (333 rows, decision_tree, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 380 µs (IQR 33%, n=5) | 451 µs (IQR 11%, n=5) | 1.187 | **regression** (inside IQR) |
| workflow fit, warm median | 369 µs (IQR 12%, n=5) | 292 µs (IQR 20%, n=5) | 0.792 | faster |
| prep + bake, warm median | 109 µs (IQR 9%, n=5) | 78.7 µs (IQR 16%, n=5) | 0.723 | faster |
| model fit, warm median | 239 µs (IQR 5%, n=5) | 202 µs (IQR 24%, n=5) | 0.846 | faster (inside IQR) |
| predict | 33.6 µs (IQR 36%, n=5) | 32 µs (IQR 6%, n=5) | 0.952 | within noise band (inside IQR) |
| resampling (cv) | 5.85 ms (IQR 33%, n=5) | 6.05 ms (IQR 3%, n=5) | 1.035 | **regression** (inside IQR) |
| tune | 118 ms (IQR 33%, n=5) | 74.7 ms (IQR 11%, n=5) | 0.631 | faster |
| select + last_fit | 669 µs (IQR 4%, n=5) | 618 µs (IQR 5%, n=5) | 0.923 | faster |
| wall (cold process) | 202 ms (IQR 20%, n=5) | 159 ms (IQR 20%, n=5) | 0.786 | faster |
| peak working set MB | 5.36 MB (n=5) | 5.34 MB (n=5) | 0.995 | within noise band |
| allocations wf_fit (count) | 511 (n=5) | 423 (n=5) | 0.828 | smaller |
| allocations wf_fit (MB) | 0.192 MB (n=5) | 0.103 MB (n=5) | 0.535 | smaller |
| allocations cv (count) | 7,717 (n=5) | 6,487 (n=5) | 0.841 | smaller |
| allocations cv (MB) | 2.72 MB (n=5) | 1.57 MB (n=5) | 0.578 | smaller |

## penguins_tree_clf (333 rows, decision_tree, 4 threads) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 256 µs (IQR 6%, n=5) | 284 µs (IQR 32%, n=5) | 1.109 | **regression** (inside IQR) |
| workflow fit, warm median | 229 µs (IQR 12%, n=5) | 173 µs (IQR 68%, n=5) | 0.755 | faster (inside IQR) |
| prep + bake, warm median | 77.5 µs (IQR 12%, n=5) | 48.9 µs (IQR 59%, n=5) | 0.631 | faster (inside IQR) |
| model fit, warm median | 155 µs (IQR 18%, n=5) | 125 µs (IQR 67%, n=5) | 0.809 | faster (inside IQR) |
| predict | 17.8 µs (IQR 47%, n=5) | 17.3 µs (IQR 62%, n=5) | 0.972 | within noise band (inside IQR) |
| resampling (cv) | 2.48 ms (IQR 13%, n=5) | 2.3 ms (IQR 40%, n=5) | 0.927 | faster (inside IQR) |
| tune | 31.2 ms (IQR 5%, n=5) | 19.8 ms (IQR 4%, n=5) | 0.633 | faster |
| select + last_fit | 537 µs (IQR 9%, n=5) | 418 µs (IQR 6%, n=5) | 0.777 | faster |
| wall (cold process) | 83.1 ms (IQR 1%, n=5) | 61.6 ms (IQR 24%, n=5) | 0.741 | faster |
| peak working set MB | 6.29 MB (n=5) | 5.99 MB (n=5) | 0.952 | smaller |
| allocations wf_fit (count) | 511 (n=5) | 423 (n=5) | 0.828 | smaller |
| allocations wf_fit (MB) | 0.192 MB (n=5) | 0.103 MB (n=5) | 0.535 | smaller |
| allocations cv (count) | 7,771 (n=5) | 6,541 (n=5) | 0.842 | smaller |
| allocations cv (MB) | 2.75 MB (n=5) | 1.61 MB (n=5) | 0.583 | smaller |

## synth_100000_reg (100000 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 80.6 ms (IQR 14%, n=5) | 64.6 ms (IQR 9%, n=5) | 0.802 | faster |
| workflow fit, warm median | 112 ms (IQR 4%, n=5) | 90.7 ms (IQR 3%, n=5) | 0.813 | faster |
| prep + bake, warm median | 69.4 ms (IQR 8%, n=5) | 48.5 ms (IQR 9%, n=5) | 0.699 | faster |
| model fit, warm median | 41.3 ms (IQR 1%, n=5) | 41 ms (IQR 9%, n=5) | 0.992 | within noise band (inside IQR) |
| predict | 412 µs (IQR 23%, n=5) | 458 µs (IQR 24%, n=5) | 1.113 | **regression** (inside IQR) |
| resampling (cv) | 1.44 s (IQR 2%, n=5) | 1.17 s (IQR 4%, n=5) | 0.815 | faster |
| wall (cold process) | 13.2 s (IQR 10%, n=5) | 11.9 s (IQR 2%, n=5) | 0.898 | faster |
| peak working set MB | 73 MB (n=5) | 75.2 MB (n=5) | 1.030 | within noise band |
| allocations wf_fit (count) | 573 (n=5) | 465 (n=5) | 0.812 | smaller |
| allocations wf_fit (MB) | 93.8 MB (n=5) | 56.4 MB (n=5) | 0.601 | smaller |
| allocations cv (count) | 8,036 (n=5) | 6,486 (n=5) | 0.807 | smaller |
| allocations cv (MB) | 1.21e+03 MB (n=5) | 731 MB (n=5) | 0.604 | smaller |

## synth_10000_clf (10000 rows, logistic_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 24.5 ms (IQR 26%, n=5) | 18.6 ms (IQR 9%, n=5) | 0.759 | faster (inside IQR) |
| workflow fit, warm median | 27.5 ms (IQR 5%, n=5) | 26.6 ms (IQR 6%, n=5) | 0.967 | within noise band (inside IQR) |
| prep + bake, warm median | 4.73 ms (IQR 9%, n=5) | 3.88 ms (IQR 6%, n=5) | 0.820 | faster |
| model fit, warm median | 22.5 ms (IQR 8%, n=5) | 22.1 ms (IQR 7%, n=5) | 0.981 | within noise band (inside IQR) |
| predict | 131 µs (IQR 8%, n=5) | 84.7 µs (IQR 24%, n=5) | 0.647 | faster |
| resampling (cv) | 329 ms (IQR 17%, n=5) | 247 ms (IQR 8%, n=5) | 0.750 | faster |
| tune | 6.84 s (IQR 19%, n=5) | 4.93 s (IQR 5%, n=5) | 0.720 | faster |
| wall (cold process) | 9.12 s (IQR 18%, n=5) | 6.92 s (IQR 4%, n=5) | 0.759 | faster |
| peak working set MB | 15.2 MB (n=5) | 14.1 MB (n=5) | 0.924 | smaller |
| allocations wf_fit (count) | 668 (n=5) | 536 (n=5) | 0.802 | smaller |
| allocations wf_fit (MB) | 15.4 MB (n=5) | 11.9 MB (n=5) | 0.771 | smaller |
| allocations cv (count) | 9,386 (n=5) | 7,476 (n=5) | 0.797 | smaller |
| allocations cv (MB) | 201 MB (n=5) | 156 MB (n=5) | 0.775 | smaller |

## synth_10000_clf (10000 rows, logistic_reg, 4 threads) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 18.9 ms (IQR 47%, n=5) | 16.8 ms (IQR 8%, n=5) | 0.888 | faster (inside IQR) |
| workflow fit, warm median | 16 ms (IQR 29%, n=5) | 18.5 ms (IQR 6%, n=5) | 1.153 | **regression** (inside IQR) |
| prep + bake, warm median | 2.35 ms (IQR 40%, n=5) | 2.79 ms (IQR 13%, n=5) | 1.186 | **regression** (inside IQR) |
| model fit, warm median | 13.3 ms (IQR 27%, n=5) | 15.7 ms (IQR 12%, n=5) | 1.182 | **regression** (inside IQR) |
| predict | 91.3 µs (IQR 37%, n=5) | 104 µs (IQR 40%, n=5) | 1.142 | **regression** (inside IQR) |
| resampling (cv) | 90.8 ms (IQR 36%, n=5) | 69.7 ms (IQR 46%, n=5) | 0.767 | faster (inside IQR) |
| tune | 1.74 s (IQR 30%, n=5) | 1.25 s (IQR 15%, n=5) | 0.721 | faster (inside IQR) |
| wall (cold process) | 2.89 s (IQR 47%, n=5) | 2.53 s (IQR 21%, n=5) | 0.874 | faster (inside IQR) |
| peak working set MB | 33.2 MB (n=5) | 31.1 MB (n=5) | 0.938 | smaller |
| allocations wf_fit (count) | 668 (n=5) | 536 (n=5) | 0.802 | smaller |
| allocations wf_fit (MB) | 15.4 MB (n=5) | 11.9 MB (n=5) | 0.771 | smaller |
| allocations cv (count) | 9,440 (n=5) | 7,530 (n=5) | 0.798 | smaller |
| allocations cv (MB) | 201 MB (n=5) | 156 MB (n=5) | 0.775 | smaller |

## synth_10000_reg (10000 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 7.94 ms (IQR 6%, n=5) | 6.59 ms (IQR 8%, n=5) | 0.830 | faster |
| workflow fit, warm median | 8.47 ms (IQR 4%, n=5) | 6.18 ms (IQR 2%, n=5) | 0.730 | faster |
| prep + bake, warm median | 5.38 ms (IQR 6%, n=5) | 3.08 ms (IQR 5%, n=5) | 0.572 | faster |
| model fit, warm median | 2.8 ms (IQR 7%, n=5) | 2.85 ms (IQR 2%, n=5) | 1.017 | within noise band (inside IQR) |
| predict | 31.6 µs (IQR 45%, n=5) | 33.4 µs (IQR 72%, n=5) | 1.057 | **regression** (inside IQR) |
| resampling (cv) | 111 ms (IQR 3%, n=5) | 78.7 ms (IQR 9%, n=5) | 0.711 | faster |
| wall (cold process) | 721 ms (IQR 3%, n=5) | 556 ms (IQR 2%, n=5) | 0.772 | faster |
| peak working set MB | 12.7 MB (n=5) | 12.6 MB (n=5) | 0.993 | within noise band |
| allocations wf_fit (count) | 553 (n=5) | 445 (n=5) | 0.805 | smaller |
| allocations wf_fit (MB) | 9.03 MB (n=5) | 5.27 MB (n=5) | 0.584 | smaller |
| allocations cv (count) | 7,886 (n=5) | 6,336 (n=5) | 0.803 | smaller |
| allocations cv (MB) | 124 MB (n=5) | 75.8 MB (n=5) | 0.612 | smaller |

## synth_10000_reg_boot (10000 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 9.38 ms (IQR 9%, n=5) | 7.12 ms (IQR 10%, n=5) | 0.759 | faster |
| workflow fit, warm median | 8.42 ms (IQR 11%, n=5) | 6.69 ms (IQR 8%, n=5) | 0.794 | faster |
| prep + bake, warm median | 4.69 ms (IQR 5%, n=5) | 3.35 ms (IQR 6%, n=5) | 0.714 | faster |
| model fit, warm median | 2.93 ms (IQR 5%, n=5) | 3.15 ms (IQR 7%, n=5) | 1.075 | **regression** (inside IQR) |
| predict | 35.4 µs (IQR 16%, n=5) | 31.7 µs (IQR 11%, n=5) | 0.895 | faster (inside IQR) |
| resampling (cv) | 333 ms (IQR 14%, n=5) | 262 ms (IQR 16%, n=5) | 0.785 | faster |
| wall (cold process) | 934 ms (IQR 3%, n=5) | 758 ms (IQR 12%, n=5) | 0.811 | faster |
| peak working set MB | 14.5 MB (n=5) | 15.2 MB (n=5) | 1.045 | **larger** |
| allocations wf_fit (count) | 553 (n=5) | 445 (n=5) | 0.805 | smaller |
| allocations wf_fit (MB) | 9.03 MB (n=5) | 5.27 MB (n=5) | 0.584 | smaller |
| allocations cv (count) | 19,694 (n=5) | 15,819 (n=5) | 0.803 | smaller |
| allocations cv (MB) | 379 MB (n=5) | 226 MB (n=5) | 0.596 | smaller |

## synth_10000_reg_wfreq (10000 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 10.1 ms (IQR 14%, n=5) | 7.37 ms (IQR 8%, n=5) | 0.732 | faster |
| workflow fit, warm median | 9.61 ms (IQR 4%, n=5) | 7.88 ms (IQR 10%, n=5) | 0.820 | faster |
| prep + bake, warm median | 6.2 ms (IQR 8%, n=5) | 4.03 ms (IQR 12%, n=5) | 0.651 | faster |
| model fit, warm median | 2.91 ms (IQR 3%, n=5) | 2.95 ms (IQR 6%, n=5) | 1.012 | within noise band (inside IQR) |
| predict | 37.6 µs (IQR 15%, n=5) | 29.4 µs (IQR 25%, n=5) | 0.782 | faster |
| resampling (cv) | 124 ms (IQR 1%, n=5) | 99.6 ms (IQR 4%, n=5) | 0.806 | faster |
| wall (cold process) | 803 ms (IQR 6%, n=5) | 644 ms (IQR 9%, n=5) | 0.802 | faster |
| peak working set MB | 14 MB (n=5) | 14 MB (n=5) | 0.997 | within noise band |
| allocations wf_fit (count) | 661 (n=5) | 545 (n=5) | 0.825 | smaller |
| allocations wf_fit (MB) | 11.4 MB (n=5) | 7.23 MB (n=5) | 0.632 | smaller |
| allocations cv (count) | 9,166 (n=5) | 7,516 (n=5) | 0.820 | smaller |
| allocations cv (MB) | 160 MB (n=5) | 106 MB (n=5) | 0.663 | smaller |

## synth_10000_reg_wfs (10000 rows, linear_reg, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 9.42 ms (IQR 7%, n=5) | 6.88 ms (IQR 6%, n=5) | 0.730 | faster |
| workflow fit, warm median | 6.25 ms (IQR 22%, n=5) | 5.51 ms (IQR 6%, n=5) | 0.882 | faster (inside IQR) |
| prep + bake, warm median | 3.01 ms (IQR 11%, n=5) | 2.48 ms (IQR 5%, n=5) | 0.823 | faster |
| model fit, warm median | 2.66 ms (IQR 27%, n=5) | 2.9 ms (IQR 4%, n=5) | 1.091 | **regression** (inside IQR) |
| predict | 39.1 µs (IQR 30%, n=5) | 32.5 µs (IQR 16%, n=5) | 0.831 | faster (inside IQR) |
| resampling (cv) | 111 ms (IQR 5%, n=5) | 87.9 ms (IQR 6%, n=5) | 0.792 | faster |
| workflow set | 1.57 s (IQR 15%, n=5) | 1.52 s (IQR 13%, n=5) | 0.966 | within noise band (inside IQR) |
| wall (cold process) | 2.26 s (IQR 13%, n=5) | 2.11 s (IQR 10%, n=5) | 0.933 | faster (inside IQR) |
| peak working set MB | 13.9 MB (n=5) | 13.5 MB (n=5) | 0.976 | within noise band |
| allocations wf_fit (count) | 553 (n=5) | 445 (n=5) | 0.805 | smaller |
| allocations wf_fit (MB) | 9.03 MB (n=5) | 5.27 MB (n=5) | 0.584 | smaller |
| allocations cv (count) | 7,886 (n=5) | 6,336 (n=5) | 0.803 | smaller |
| allocations cv (MB) | 124 MB (n=5) | 75.8 MB (n=5) | 0.612 | smaller |

## synth_10000_reg_wfs (10000 rows, linear_reg, 4 threads) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 7.3 ms (IQR 0%, n=5) | 5.96 ms (IQR 20%, n=5) | 0.817 | faster |
| workflow fit, warm median | 4.94 ms (IQR 6%, n=5) | 4.42 ms (IQR 9%, n=5) | 0.895 | faster |
| prep + bake, warm median | 2.44 ms (IQR 10%, n=5) | 2.09 ms (IQR 16%, n=5) | 0.856 | faster |
| model fit, warm median | 2.35 ms (IQR 3%, n=5) | 2.3 ms (IQR 5%, n=5) | 0.981 | within noise band (inside IQR) |
| predict | 28.8 µs (IQR 35%, n=5) | 23.1 µs (IQR 10%, n=5) | 0.802 | faster (inside IQR) |
| resampling (cv) | 41.9 ms (IQR 9%, n=5) | 30.6 ms (IQR 4%, n=5) | 0.730 | faster |
| workflow set | 559 ms (IQR 4%, n=5) | 481 ms (IQR 10%, n=5) | 0.861 | faster |
| wall (cold process) | 1.01 s (IQR 1%, n=5) | 871 ms (IQR 4%, n=5) | 0.862 | faster |
| peak working set MB | 28.1 MB (n=5) | 27.9 MB (n=5) | 0.991 | within noise band |
| allocations wf_fit (count) | 553 (n=5) | 445 (n=5) | 0.805 | smaller |
| allocations wf_fit (MB) | 9.03 MB (n=5) | 5.27 MB (n=5) | 0.584 | smaller |
| allocations cv (count) | 7,940 (n=5) | 6,390 (n=5) | 0.805 | smaller |
| allocations cv (MB) | 124 MB (n=5) | 75.8 MB (n=5) | 0.612 | smaller |

## synth_10000_rf_clf (10000 rows, rand_forest, 1 thread) — ok trials baseline 5/5, candidate 5/5; drift: **PASS**

| phase | baseline | candidate | ratio | verdict |
|---|---|---|---|---|
| workflow fit, cold | 3.89 s (IQR 51%, n=5) | 3.33 s (IQR 36%, n=5) | 0.857 | faster (inside IQR) |
| workflow fit, warm median | 2.52 s (IQR 19%, n=5) | 3.56 s (IQR 29%, n=5) | 1.412 | **regression** (inside IQR) |
| prep + bake, warm median | 2.85 ms (IQR 11%, n=5) | 3.04 ms (IQR 27%, n=5) | 1.065 | **regression** (inside IQR) |
| model fit, warm median | 2.5 s (IQR 23%, n=5) | 3.58 s (IQR 29%, n=5) | 1.430 | **regression** |
| predict | 147 ms (IQR 60%, n=5) | 162 ms (IQR 11%, n=5) | 1.102 | **regression** (inside IQR) |
| resampling (cv) | 41.1 s (IQR 34%, n=5) | 46.6 s (IQR 26%, n=5) | 1.134 | **regression** (inside IQR) |
| wall (cold process) | 223 s (IQR 12%, n=5) | 254 s (IQR 24%, n=5) | 1.142 | **regression** (inside IQR) |
| peak working set MB | 154 MB (n=5) | 154 MB (n=5) | 0.999 | within noise band |
| allocations wf_fit (count) | 1,985,410 (n=5) | 1,985,278 (n=5) | 1.000 | within noise band |
| allocations wf_fit (MB) | 1.11e+03 MB (n=5) | 1.11e+03 MB (n=5) | 0.997 | within noise band |
| allocations cv (count) | 22,947,577 (n=5) | 22,945,667 (n=5) | 1.000 | within noise band |
| allocations cv (MB) | 1.35e+04 MB (n=5) | 1.35e+04 MB (n=5) | 0.997 | within noise band |

## Summary of phases with a decision

| task | threads | phase | ratio | verdict |
|---|---|---|---|---|
| ames_reg | 1 | workflow fit, cold | 0.987 | within noise band (inside IQR) |
| ames_reg | 1 | workflow fit, warm median | 0.863 | faster |
| ames_reg | 1 | prep + bake, warm median | 0.656 | faster |
| ames_reg | 1 | model fit, warm median | 0.994 | within noise band (inside IQR) |
| ames_reg | 1 | predict | 1.102 | **regression** (inside IQR) |
| ames_reg | 1 | resampling (cv) | 0.875 | faster |
| ames_reg | 1 | peak working set MB | 0.995 | within noise band |
| ames_tree_reg | 1 | workflow fit, cold | 0.961 | within noise band (inside IQR) |
| ames_tree_reg | 1 | workflow fit, warm median | 0.885 | faster (inside IQR) |
| ames_tree_reg | 1 | prep + bake, warm median | 0.652 | faster (inside IQR) |
| ames_tree_reg | 1 | model fit, warm median | 0.925 | faster (inside IQR) |
| ames_tree_reg | 1 | predict | 1.055 | **regression** (inside IQR) |
| ames_tree_reg | 1 | resampling (cv) | 1.010 | within noise band (inside IQR) |
| ames_tree_reg | 1 | tune | 0.844 | faster |
| ames_tree_reg | 1 | select + last_fit | 1.099 | **regression** (inside IQR) |
| ames_tree_reg | 1 | peak working set MB | 0.925 | smaller |
| ames_tree_reg | 4 | workflow fit, cold | 0.848 | faster (inside IQR) |
| ames_tree_reg | 4 | workflow fit, warm median | 0.979 | within noise band (inside IQR) |
| ames_tree_reg | 4 | prep + bake, warm median | 0.751 | faster |
| ames_tree_reg | 4 | model fit, warm median | 1.023 | within noise band (inside IQR) |
| ames_tree_reg | 4 | predict | 0.809 | faster (inside IQR) |
| ames_tree_reg | 4 | resampling (cv) | 1.143 | **regression** (inside IQR) |
| ames_tree_reg | 4 | tune | 0.787 | faster (inside IQR) |
| ames_tree_reg | 4 | select + last_fit | 1.006 | within noise band (inside IQR) |
| ames_tree_reg | 4 | peak working set MB | 0.987 | within noise band |
| ames_xgb_reg | 1 | workflow fit, cold | 0.909 | faster (inside IQR) |
| ames_xgb_reg | 1 | workflow fit, warm median | 0.945 | faster |
| ames_xgb_reg | 1 | prep + bake, warm median | 0.603 | faster |
| ames_xgb_reg | 1 | model fit, warm median | 1.018 | within noise band (inside IQR) |
| ames_xgb_reg | 1 | predict | 0.974 | within noise band (inside IQR) |
| ames_xgb_reg | 1 | resampling (cv) | 0.905 | faster (inside IQR) |
| ames_xgb_reg | 1 | tune | 1.005 | within noise band (inside IQR) |
| ames_xgb_reg | 1 | select + last_fit | 1.005 | within noise band (inside IQR) |
| ames_xgb_reg | 1 | peak working set MB | 0.949 | smaller |
| penguins_reg | 1 | workflow fit, cold | 0.922 | faster (inside IQR) |
| penguins_reg | 1 | workflow fit, warm median | 1.145 | **regression** (inside IQR) |
| penguins_reg | 1 | prep + bake, warm median | 1.156 | **regression** (inside IQR) |
| penguins_reg | 1 | model fit, warm median | 1.205 | **regression** (inside IQR) |
| penguins_reg | 1 | predict | 0.702 | faster (inside IQR) |
| penguins_reg | 1 | resampling (cv) | 0.936 | faster (inside IQR) |
| penguins_reg | 1 | peak working set MB | 0.997 | within noise band |
| penguins_tree_clf | 1 | workflow fit, cold | 1.187 | **regression** (inside IQR) |
| penguins_tree_clf | 1 | workflow fit, warm median | 0.792 | faster |
| penguins_tree_clf | 1 | prep + bake, warm median | 0.723 | faster |
| penguins_tree_clf | 1 | model fit, warm median | 0.846 | faster (inside IQR) |
| penguins_tree_clf | 1 | predict | 0.952 | within noise band (inside IQR) |
| penguins_tree_clf | 1 | resampling (cv) | 1.035 | **regression** (inside IQR) |
| penguins_tree_clf | 1 | tune | 0.631 | faster |
| penguins_tree_clf | 1 | select + last_fit | 0.923 | faster |
| penguins_tree_clf | 1 | peak working set MB | 0.995 | within noise band |
| penguins_tree_clf | 4 | workflow fit, cold | 1.109 | **regression** (inside IQR) |
| penguins_tree_clf | 4 | workflow fit, warm median | 0.755 | faster (inside IQR) |
| penguins_tree_clf | 4 | prep + bake, warm median | 0.631 | faster (inside IQR) |
| penguins_tree_clf | 4 | model fit, warm median | 0.809 | faster (inside IQR) |
| penguins_tree_clf | 4 | predict | 0.972 | within noise band (inside IQR) |
| penguins_tree_clf | 4 | resampling (cv) | 0.927 | faster (inside IQR) |
| penguins_tree_clf | 4 | tune | 0.633 | faster |
| penguins_tree_clf | 4 | select + last_fit | 0.777 | faster |
| penguins_tree_clf | 4 | peak working set MB | 0.952 | smaller |
| synth_100000_reg | 1 | workflow fit, cold | 0.802 | faster |
| synth_100000_reg | 1 | workflow fit, warm median | 0.813 | faster |
| synth_100000_reg | 1 | prep + bake, warm median | 0.699 | faster |
| synth_100000_reg | 1 | model fit, warm median | 0.992 | within noise band (inside IQR) |
| synth_100000_reg | 1 | predict | 1.113 | **regression** (inside IQR) |
| synth_100000_reg | 1 | resampling (cv) | 0.815 | faster |
| synth_100000_reg | 1 | peak working set MB | 1.030 | within noise band |
| synth_10000_clf | 1 | workflow fit, cold | 0.759 | faster (inside IQR) |
| synth_10000_clf | 1 | workflow fit, warm median | 0.967 | within noise band (inside IQR) |
| synth_10000_clf | 1 | prep + bake, warm median | 0.820 | faster |
| synth_10000_clf | 1 | model fit, warm median | 0.981 | within noise band (inside IQR) |
| synth_10000_clf | 1 | predict | 0.647 | faster |
| synth_10000_clf | 1 | resampling (cv) | 0.750 | faster |
| synth_10000_clf | 1 | tune | 0.720 | faster |
| synth_10000_clf | 1 | peak working set MB | 0.924 | smaller |
| synth_10000_clf | 4 | workflow fit, cold | 0.888 | faster (inside IQR) |
| synth_10000_clf | 4 | workflow fit, warm median | 1.153 | **regression** (inside IQR) |
| synth_10000_clf | 4 | prep + bake, warm median | 1.186 | **regression** (inside IQR) |
| synth_10000_clf | 4 | model fit, warm median | 1.182 | **regression** (inside IQR) |
| synth_10000_clf | 4 | predict | 1.142 | **regression** (inside IQR) |
| synth_10000_clf | 4 | resampling (cv) | 0.767 | faster (inside IQR) |
| synth_10000_clf | 4 | tune | 0.721 | faster (inside IQR) |
| synth_10000_clf | 4 | peak working set MB | 0.938 | smaller |
| synth_10000_reg | 1 | workflow fit, cold | 0.830 | faster |
| synth_10000_reg | 1 | workflow fit, warm median | 0.730 | faster |
| synth_10000_reg | 1 | prep + bake, warm median | 0.572 | faster |
| synth_10000_reg | 1 | model fit, warm median | 1.017 | within noise band (inside IQR) |
| synth_10000_reg | 1 | predict | 1.057 | **regression** (inside IQR) |
| synth_10000_reg | 1 | resampling (cv) | 0.711 | faster |
| synth_10000_reg | 1 | peak working set MB | 0.993 | within noise band |
| synth_10000_reg_boot | 1 | workflow fit, cold | 0.759 | faster |
| synth_10000_reg_boot | 1 | workflow fit, warm median | 0.794 | faster |
| synth_10000_reg_boot | 1 | prep + bake, warm median | 0.714 | faster |
| synth_10000_reg_boot | 1 | model fit, warm median | 1.075 | **regression** (inside IQR) |
| synth_10000_reg_boot | 1 | predict | 0.895 | faster (inside IQR) |
| synth_10000_reg_boot | 1 | resampling (cv) | 0.785 | faster |
| synth_10000_reg_boot | 1 | peak working set MB | 1.045 | **larger** |
| synth_10000_reg_wfreq | 1 | workflow fit, cold | 0.732 | faster |
| synth_10000_reg_wfreq | 1 | workflow fit, warm median | 0.820 | faster |
| synth_10000_reg_wfreq | 1 | prep + bake, warm median | 0.651 | faster |
| synth_10000_reg_wfreq | 1 | model fit, warm median | 1.012 | within noise band (inside IQR) |
| synth_10000_reg_wfreq | 1 | predict | 0.782 | faster |
| synth_10000_reg_wfreq | 1 | resampling (cv) | 0.806 | faster |
| synth_10000_reg_wfreq | 1 | peak working set MB | 0.997 | within noise band |
| synth_10000_reg_wfs | 1 | workflow fit, cold | 0.730 | faster |
| synth_10000_reg_wfs | 1 | workflow fit, warm median | 0.882 | faster (inside IQR) |
| synth_10000_reg_wfs | 1 | prep + bake, warm median | 0.823 | faster |
| synth_10000_reg_wfs | 1 | model fit, warm median | 1.091 | **regression** (inside IQR) |
| synth_10000_reg_wfs | 1 | predict | 0.831 | faster (inside IQR) |
| synth_10000_reg_wfs | 1 | resampling (cv) | 0.792 | faster |
| synth_10000_reg_wfs | 1 | workflow set | 0.966 | within noise band (inside IQR) |
| synth_10000_reg_wfs | 1 | peak working set MB | 0.976 | within noise band |
| synth_10000_reg_wfs | 4 | workflow fit, cold | 0.817 | faster |
| synth_10000_reg_wfs | 4 | workflow fit, warm median | 0.895 | faster |
| synth_10000_reg_wfs | 4 | prep + bake, warm median | 0.856 | faster |
| synth_10000_reg_wfs | 4 | model fit, warm median | 0.981 | within noise band (inside IQR) |
| synth_10000_reg_wfs | 4 | predict | 0.802 | faster (inside IQR) |
| synth_10000_reg_wfs | 4 | resampling (cv) | 0.730 | faster |
| synth_10000_reg_wfs | 4 | workflow set | 0.861 | faster |
| synth_10000_reg_wfs | 4 | peak working set MB | 0.991 | within noise band |
| synth_10000_rf_clf | 1 | workflow fit, cold | 0.857 | faster (inside IQR) |
| synth_10000_rf_clf | 1 | workflow fit, warm median | 1.412 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | prep + bake, warm median | 1.065 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | model fit, warm median | 1.430 | **regression** |
| synth_10000_rf_clf | 1 | predict | 1.102 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | resampling (cv) | 1.134 | **regression** (inside IQR) |
| synth_10000_rf_clf | 1 | peak working set MB | 0.999 | within noise band |
