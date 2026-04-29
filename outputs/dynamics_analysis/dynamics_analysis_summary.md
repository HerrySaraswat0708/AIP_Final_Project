# Dynamics Analysis Summary
## 1. Adaptation Dynamics
| dataset   |   final_clip_acc |   final_tda_acc |   final_ftta_acc |   stability_clip |   stability_tda |   stability_ftta |   time_to_beat_clip_tda_pct |   time_to_beat_clip_ftta_pct |
|:----------|-----------------:|----------------:|-----------------:|-----------------:|----------------:|-----------------:|----------------------------:|-----------------------------:|
| dtd       |           43.936 |          45.851 |           45.479 |           16.192 |          17.981 |           16.675 |                           0 |                            0 |
| pets      |           88.389 |          88.416 |           88.962 |            7.23  |           9.366 |            7.286 |                           0 |                            0 |

## 2. Uncertainty Analysis — Entropy Statistics
| dataset   | method   |   mean_entropy_correct |   mean_entropy_wrong |   entropy_gap |   spearman_rho |   spearman_pval |
|:----------|:---------|-----------------------:|---------------------:|--------------:|---------------:|----------------:|
| dtd       | clip     |                 3.85   |               3.85   |        0      |         0.261  |          0      |
| dtd       | tda      |                 0.5062 |               1.1503 |        0.644  |         0.4161 |          0      |
| dtd       | freetta  |                 1.4065 |               2.3053 |        0.8988 |         0.4452 |          0      |
| pets      | clip     |                 3.6098 |               3.6098 |       -0      |        -0.028  |          0.0904 |
| pets      | tda      |                 0.2885 |               0.9389 |        0.6504 |         0.4191 |          0      |
| pets      | freetta  |                 0.302  |               1.0018 |        0.6998 |         0.4351 |          0      |

### Entropy–Accuracy Calibration
| dataset   | method   |   low_ent_acc |   high_ent_acc |   entropy_acc_rho |
|:----------|:---------|--------------:|---------------:|------------------:|
| dtd       | clip     |        63.298 |         26.596 |             0.261 |
| dtd       | tda      |        85.106 |         15.957 |             0.416 |
| dtd       | freetta  |        97.34  |         15.426 |             0.445 |
| pets      | clip     |        87.738 |         89.373 |            -0.028 |
| pets      | tda      |       100     |         51.499 |             0.419 |
| pets      | freetta  |       100     |         50.136 |             0.435 |

## 3. Distribution Modeling — Geometry
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |    GAS |   mu_drift_final |   cache_pressure |
|:----------|----------------------:|-----------------:|-------:|-----------------:|-----------------:|
| caltech   |                97.444 |           91.846 |  0.056 |            1.134 |            4.93  |
| dtd       |                73.191 |           63.617 |  0.096 |            1.155 |            5.714 |
| eurosat   |                78.259 |           89.802 | -0.115 |            1.142 |          162     |
| pets      |                91.496 |           83.456 |  0.08  |            1.122 |           19.832 |
| imagenet  |                82.72  |           41.89  |  0.408 |            1.128 |            2     |

## 4. Computational Efficiency
| dataset   |   n_samples |   tda_ms_per_sample |   ftta_ms_per_sample |   speedup_ftta_vs_tda |
|:----------|------------:|--------------------:|---------------------:|----------------------:|
| caltech   |        2465 |               0.394 |                0.294 |                 1.341 |
| dtd       |        1880 |               0.375 |                0.187 |                 2.006 |
| eurosat   |        8100 |               0.353 |                0.169 |                 2.09  |
| pets      |        3669 |               0.381 |                0.192 |                 1.987 |
| imagenet  |       10000 |               1.961 |                0.703 |                 2.788 |
