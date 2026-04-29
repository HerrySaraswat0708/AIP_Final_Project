# Final Research Report
## CLIP vs TDA vs FreeTTA: Deep Comparative Analysis

*Generated: 2026-04-26 19:15*

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Section 2: Baseline Accuracy](#section-2-baseline-accuracy)
3. [Section 3: Prediction Change Analysis](#section-3-prediction-change-analysis)
4. [Section 4: Entropy & Confidence](#section-4-entropy--confidence)
5. [Section 7: Disagreement Analysis](#section-7-disagreement-analysis)
6. [Section 8: Failure Buckets](#section-8-failure-buckets)
7. [Section 10: Novel Metrics](#section-10-novel-metrics)
8. [Section 12: Improvement Attempts](#section-12-improvement-attempts)
9. [Section 11: Synthesis & Insights](#section-11-synthesis--insights)

---

## Executive Summary

| dataset   |   n_samples |   clip_acc |   tda_acc |   freetta_acc |   conf_ftta_acc |   ent_tda_acc |   hybrid_acc |   tda_gain |   freetta_gain |   conf_ftta_gain |   ent_tda_gain |   hybrid_gain |
|:----------|------------:|-----------:|----------:|--------------:|----------------:|--------------:|-------------:|-----------:|---------------:|-----------------:|---------------:|--------------:|
| caltech   |        2465 |     0.9355 |    0.9343 |        0.9375 |          0.9375 |        0.9359 |       0.9387 |    -0.0012 |         0.0020 |           0.0020 |         0.0004 |        0.0032 |
| dtd       |        1880 |     0.4394 |    0.4606 |        0.4596 |          0.4378 |        0.4638 |       0.4479 |     0.0213 |         0.0202 |          -0.0016 |         0.0245 |        0.0085 |
| eurosat   |        8100 |     0.4843 |    0.6215 |        0.5974 |          0.5864 |        0.6226 |       0.6156 |     0.1372 |         0.1131 |           0.1021 |         0.1383 |        0.1312 |
| imagenet  |       10000 |     0.6235 |    0.6289 |        0.6268 |          0.6254 |        0.6298 |       0.6250 |     0.0054 |         0.0033 |           0.0019 |         0.0063 |        0.0015 |
| pets      |        3669 |     0.8839 |    0.8964 |        0.8893 |          0.8874 |        0.8934 |       0.8907 |     0.0125 |         0.0055 |           0.0035 |         0.0095 |        0.0068 |

- TDA outperforms CLIP on **4/5** datasets.
- FreeTTA outperforms CLIP on **5/5** datasets.


## Section 2: Baseline Accuracy

| dataset   |   n_samples |   clip_acc |   tda_acc |   freetta_acc |   conf_ftta_acc |   ent_tda_acc |   hybrid_acc |   tda_gain |   freetta_gain |   conf_ftta_gain |   ent_tda_gain |   hybrid_gain |
|:----------|------------:|-----------:|----------:|--------------:|----------------:|--------------:|-------------:|-----------:|---------------:|-----------------:|---------------:|--------------:|
| caltech   |        2465 |     0.9355 |    0.9343 |        0.9375 |          0.9375 |        0.9359 |       0.9387 |    -0.0012 |         0.0020 |           0.0020 |         0.0004 |        0.0032 |
| dtd       |        1880 |     0.4394 |    0.4606 |        0.4596 |          0.4378 |        0.4638 |       0.4479 |     0.0213 |         0.0202 |          -0.0016 |         0.0245 |        0.0085 |
| eurosat   |        8100 |     0.4843 |    0.6215 |        0.5974 |          0.5864 |        0.6226 |       0.6156 |     0.1372 |         0.1131 |           0.1021 |         0.1383 |        0.1312 |
| imagenet  |       10000 |     0.6235 |    0.6289 |        0.6268 |          0.6254 |        0.6298 |       0.6250 |     0.0054 |         0.0033 |           0.0019 |         0.0063 |        0.0015 |
| pets      |        3669 |     0.8839 |    0.8964 |        0.8893 |          0.8874 |        0.8934 |       0.8907 |     0.0125 |         0.0055 |           0.0035 |         0.0095 |        0.0068 |

## Section 3: Prediction Change Analysis

| dataset   | method   |   change_rate |   beneficial_rate |   harmful_rate |   net_correction_rate |   correction_efficiency |
|:----------|:---------|--------------:|------------------:|---------------:|----------------------:|------------------------:|
| caltech   | tda      |        0.0187 |            0.0077 |         0.0089 |               -0.0012 |                  0.4130 |
| caltech   | freetta  |        0.0020 |            0.0020 |         0.0000 |                0.0020 |                  1.0000 |
| dtd       | tda      |        0.1266 |            0.0378 |         0.0165 |                0.0213 |                  0.2983 |
| dtd       | freetta  |        0.1676 |            0.0457 |         0.0255 |                0.0202 |                  0.2730 |
| eurosat   | tda      |        0.3853 |            0.1828 |         0.0457 |                0.1372 |                  0.4745 |
| eurosat   | freetta  |        0.3338 |            0.1500 |         0.0369 |                0.1131 |                  0.4493 |
| imagenet  | tda      |        0.0381 |            0.0120 |         0.0066 |                0.0054 |                  0.3150 |
| imagenet  | freetta  |        0.0381 |            0.0117 |         0.0084 |                0.0033 |                  0.3071 |
| pets      | tda      |        0.0395 |            0.0237 |         0.0112 |                0.0125 |                  0.6000 |
| pets      | freetta  |        0.0161 |            0.0095 |         0.0041 |                0.0055 |                  0.5932 |

**Correction Efficiency (CE) = beneficial_flips / total_changes.**  CE > 0.5 means more than half of all changes are beneficial.

## Section 4: Entropy & Confidence

### Mean Confidence (correct vs wrong)

|                         |   correct |   wrong |
|:------------------------|----------:|--------:|
| ('caltech', 'clip')     |    0.9218 |  0.6136 |
| ('caltech', 'freetta')  |    0.9324 |  0.6226 |
| ('caltech', 'tda')      |    0.9706 |  0.767  |
| ('dtd', 'clip')         |    0.5953 |  0.3227 |
| ('dtd', 'freetta')      |    0.726  |  0.4541 |
| ('dtd', 'tda')          |    0.7172 |  0.4421 |
| ('eurosat', 'clip')     |    0.5989 |  0.3348 |
| ('eurosat', 'freetta')  |    0.8221 |  0.6154 |
| ('eurosat', 'tda')      |    0.7751 |  0.5675 |
| ('imagenet', 'clip')    |    0.7579 |  0.4713 |
| ('imagenet', 'freetta') |    0.7807 |  0.5008 |
| ('imagenet', 'tda')     |    0.8276 |  0.5566 |
| ('pets', 'clip')        |    0.875  |  0.5523 |
| ('pets', 'freetta')     |    0.8875 |  0.5704 |
| ('pets', 'tda')         |    0.8977 |  0.5936 |

## Section 7: Disagreement Analysis

| dataset   |   disagreement_rate |   n_disagree |   tda_acc_on_disagree |   freetta_acc_on_disagree |   clip_acc_on_disagree |   mean_clip_entropy_on_disagree |   tda_wins |   freetta_wins |
|:----------|--------------------:|-------------:|----------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------:|---------------:|
| caltech   |              0.0174 |           43 |                0.3488 |                    0.5349 |                 0.5116 |                          1.2537 |         15 |             23 |
| dtd       |              0.1441 |          271 |                0.2214 |                    0.2140 |                 0.1697 |                          2.8066 |         60 |             58 |
| eurosat   |              0.2335 |         1891 |                0.3945 |                    0.2914 |                 0.2078 |                          1.8033 |        746 |            551 |
| imagenet  |              0.0596 |          596 |                0.2752 |                    0.2399 |                 0.2164 |                          2.5698 |        164 |            143 |
| pets      |              0.0373 |          137 |                0.5255 |                    0.3358 |                 0.3358 |                          1.2322 |         72 |             46 |

## Section 8: Failure Buckets

| dataset   |   all_correct |   all_fail |   clip_correct_both_fail |   only_freetta_correct |   only_tda_correct |
|:----------|--------------:|-----------:|-------------------------:|-----------------------:|-------------------:|
| caltech   |        0.9266 |     0.0564 |                   0      |                 0.0004 |             0.0061 |
| dtd       |        0.4064 |     0.4995 |                   0.009  |                 0.0234 |             0.0154 |
| eurosat   |        0.4221 |     0.2901 |                   0.0204 |                 0.0427 |             0.0756 |
| imagenet  |        0.6096 |     0.3557 |                   0.0011 |                 0.0088 |             0.0091 |
| pets      |        0.87   |     0.0897 |                   0.0014 |                 0.0027 |             0.0169 |

## Section 10: Novel Metrics

### Correction Efficiency (CE)

| dataset   | method    |   n_beneficial |   n_harmful |   correction_efficiency |
|:----------|:----------|---------------:|------------:|------------------------:|
| caltech   | tda       |             19 |          22 |                  0.4130 |
| caltech   | freetta   |              5 |           0 |                  1.0000 |
| caltech   | conf_ftta |              5 |           0 |                  1.0000 |
| caltech   | ent_tda   |             21 |          20 |                  0.4667 |
| caltech   | hybrid    |             26 |          18 |                  0.5200 |
| dtd       | tda       |             71 |          31 |                  0.2983 |
| dtd       | freetta   |             86 |          48 |                  0.2730 |
| dtd       | conf_ftta |             83 |          86 |                  0.2054 |
| dtd       | ent_tda   |             75 |          29 |                  0.3112 |
| dtd       | hybrid    |             63 |          47 |                  0.2360 |
| eurosat   | tda       |           1481 |         370 |                  0.4745 |
| eurosat   | freetta   |           1215 |         299 |                  0.4493 |
| eurosat   | conf_ftta |           1195 |         368 |                  0.3974 |
| eurosat   | ent_tda   |           1489 |         369 |                  0.4748 |
| eurosat   | hybrid    |           1333 |         270 |                  0.4846 |
| imagenet  | tda       |            120 |          66 |                  0.3150 |
| imagenet  | freetta   |            117 |          84 |                  0.3071 |
| imagenet  | conf_ftta |            132 |         113 |                  0.2839 |
| imagenet  | ent_tda   |            108 |          45 |                  0.3985 |
| imagenet  | hybrid    |             99 |          84 |                  0.3163 |
| pets      | tda       |             87 |          41 |                  0.6000 |
| pets      | freetta   |             35 |          15 |                  0.5932 |
| pets      | conf_ftta |             32 |          19 |                  0.5161 |
| pets      | ent_tda   |             78 |          43 |                  0.5909 |
| pets      | hybrid    |             47 |          22 |                  0.6351 |

### Overconfidence Error Rate (OER)

| dataset   | method    |   n_wrong |    oer |
|:----------|:----------|----------:|-------:|
| caltech   | clip      |       159 | 0.0692 |
| caltech   | tda       |       162 | 0.3580 |
| caltech   | freetta   |       154 | 0.0779 |
| caltech   | conf_ftta |       154 | 0.0779 |
| caltech   | ent_tda   |       158 | 0.3797 |
| caltech   | hybrid    |       151 | 0.1921 |
| dtd       | clip      |      1054 | 0.0028 |
| dtd       | tda       |      1014 | 0.0483 |
| dtd       | freetta   |      1016 | 0.0679 |
| dtd       | conf_ftta |      1057 | 0.0766 |
| dtd       | ent_tda   |      1008 | 0.0526 |
| dtd       | hybrid    |      1038 | 0.0472 |
| eurosat   | clip      |      4177 | 0.0053 |
| eurosat   | tda       |      3066 | 0.0639 |
| eurosat   | freetta   |      3261 | 0.1352 |
| eurosat   | conf_ftta |      3350 | 0.1057 |
| eurosat   | ent_tda   |      3057 | 0.0677 |
| eurosat   | hybrid    |      3114 | 0.0665 |
| imagenet  | clip      |      3765 | 0.0428 |
| imagenet  | tda       |      3711 | 0.1124 |
| imagenet  | freetta   |      3732 | 0.0616 |
| imagenet  | conf_ftta |      3746 | 0.0627 |
| imagenet  | ent_tda   |      3702 | 0.1126 |
| imagenet  | hybrid    |      3750 | 0.0560 |
| pets      | clip      |       426 | 0.0352 |
| pets      | tda       |       380 | 0.0763 |
| pets      | freetta   |       406 | 0.0468 |
| pets      | conf_ftta |       413 | 0.0436 |
| pets      | ent_tda   |       391 | 0.0818 |
| pets      | hybrid    |       401 | 0.0524 |

### Logit Movement Magnitude (LMM)

| dataset   | method    |   lmm_mean_all |   lmm_std_all |   lmm_mean_beneficial |   lmm_mean_harmful |   lmm_mean_unchanged |
|:----------|:----------|---------------:|--------------:|----------------------:|-------------------:|---------------------:|
| dtd       | tda       |         0.1639 |        0.1146 |                0.1697 |             0.1406 |               0.1657 |
| dtd       | freetta   |         0.1865 |        0.1123 |                0.2298 |             0.2617 |               0.1807 |
| dtd       | conf_ftta |         0.2081 |        0.1345 |                0.2609 |             0.3301 |               0.1926 |
| dtd       | ent_tda   |         0.1664 |        0.1146 |                0.1702 |             0.1485 |               0.1683 |
| dtd       | hybrid    |         0.1574 |        0.0889 |                0.1870 |             0.1901 |               0.1549 |
| caltech   | tda       |         0.0772 |        0.1237 |                0.3607 |             0.3443 |               0.0721 |
| caltech   | freetta   |         0.0141 |        0.0266 |                0.0622 |           nan      |               0.0140 |
| caltech   | conf_ftta |         0.0141 |        0.0265 |                0.0625 |           nan      |               0.0140 |
| caltech   | ent_tda   |         0.0769 |        0.1238 |                0.3508 |             0.3581 |               0.0719 |
| caltech   | hybrid    |         0.0567 |        0.0997 |                0.3266 |             0.5119 |               0.0502 |
| eurosat   | tda       |         0.3632 |        0.2105 |                0.5345 |             0.3920 |               0.2869 |
| eurosat   | freetta   |         0.3936 |        0.2850 |                0.7473 |             0.4813 |               0.2592 |
| eurosat   | conf_ftta |         0.3804 |        0.2499 |                0.6366 |             0.4064 |               0.2791 |
| eurosat   | ent_tda   |         0.3630 |        0.2118 |                0.5361 |             0.3876 |               0.2860 |
| eurosat   | hybrid    |         0.3692 |        0.2507 |                0.6640 |             0.3859 |               0.2604 |
| pets      | tda       |         0.0588 |        0.0784 |                0.1992 |             0.1867 |               0.0535 |
| pets      | freetta   |         0.0260 |        0.0408 |                0.0783 |             0.2493 |               0.0244 |
| pets      | conf_ftta |         0.0267 |        0.0425 |                0.0763 |             0.2227 |               0.0249 |
| pets      | ent_tda   |         0.0559 |        0.0757 |                0.1951 |             0.1681 |               0.0512 |
| pets      | hybrid    |         0.0340 |        0.0419 |                0.1097 |             0.1345 |               0.0323 |
| imagenet  | tda       |         0.1080 |        0.0912 |                0.1429 |             0.1397 |               0.1073 |
| imagenet  | freetta   |         0.0449 |        0.0411 |                0.0921 |             0.0818 |               0.0433 |
| imagenet  | conf_ftta |         0.0492 |        0.0449 |                0.1024 |             0.1011 |               0.0469 |
| imagenet  | ent_tda   |         0.0996 |        0.0925 |                0.0971 |             0.0810 |               0.1002 |
| imagenet  | hybrid    |         0.0340 |        0.0293 |                0.0744 |             0.0712 |               0.0330 |

*LMM = ||prob_method - prob_clip||₂.  Beneficial flips should have moderate LMM; very large LMM with harmful flips suggests the method is moving predictions aggressively in the wrong direction.*

### Stability Score (SS)

| dataset   |   clip_stability |   tda_stability |   freetta_stability |   conf_ftta_stability |   ent_tda_stability |   hybrid_stability |
|:----------|-----------------:|----------------:|--------------------:|----------------------:|--------------------:|-------------------:|
| caltech   |           0.9681 |          0.9681 |              0.9690 |                0.9690 |              0.9688 |             0.9683 |
| dtd       |           0.9340 |          0.9296 |              0.9215 |                0.9211 |              0.9296 |             0.9343 |
| eurosat   |           0.9365 |          0.9288 |              0.9266 |                0.9284 |              0.9301 |             0.9306 |
| imagenet  |           0.9392 |          0.9386 |              0.9398 |                0.9398 |              0.9392 |             0.9398 |
| pets      |           0.9540 |          0.9591 |              0.9541 |                0.9532 |              0.9583 |             0.9564 |

*SS = 1 / (1 + σ(rolling_accuracy)).  Values close to 1.0 indicate a smooth, stable adaptation trajectory.*


## Section 12: Improvement Attempts

| dataset   |   clip_acc |   tda_acc |   freetta_acc |   conf_ftta_acc |   ent_tda_acc |   hybrid_acc |   conf_ftta_gain |   ent_tda_gain |   hybrid_gain |
|:----------|-----------:|----------:|--------------:|----------------:|--------------:|-------------:|-----------------:|---------------:|--------------:|
| caltech   |     0.9355 |    0.9343 |        0.9375 |          0.9375 |        0.9359 |       0.9387 |           0.0020 |         0.0004 |        0.0032 |
| dtd       |     0.4394 |    0.4606 |        0.4596 |          0.4378 |        0.4638 |       0.4479 |          -0.0016 |         0.0245 |        0.0085 |
| eurosat   |     0.4843 |    0.6215 |        0.5974 |          0.5864 |        0.6226 |       0.6156 |           0.1021 |         0.1383 |        0.1312 |
| imagenet  |     0.6235 |    0.6289 |        0.6268 |          0.6254 |        0.6298 |       0.6250 |           0.0019 |         0.0063 |        0.0015 |
| pets      |     0.8839 |    0.8964 |        0.8893 |          0.8874 |        0.8934 |       0.8907 |           0.0035 |         0.0095 |        0.0068 |

### Method 1: Confidence-Gated FreeTTA (ConfGatedFreeTTA)
- **Idea**: Skip the M-step (class mean update) when max CLIP confidence < threshold.
- **Rationale**: Prevents uncertain pseudo-labels from corrupting class means.
- **Expected win case**: Datasets with many uncertain CLIP predictions (DTD, EuroSAT).
- **Expected fail case**: Datasets where CLIP is mostly confident — gating starves adaptation.

### Method 2: Entropy-Gated TDA (EntropyGatedTDA)
- **Idea**: Adaptively tighten the entropy threshold for cache insertion using the running p-th percentile.
- **Rationale**: The original fixed threshold may be too permissive — only the most confident samples should enter the positive cache.
- **Expected win case**: Noisy datasets where standard TDA fills cache with borderline samples.
- **Expected fail case**: Any dataset where the cache needs volume to find similar samples.

### Method 3: Hybrid (TDA + FreeTTA)
- **Idea**: Combine local cache adjustment (TDA) + global mean adjustment (FreeTTA) via weighted sum.
- **Rationale**: Local memory captures short-range stream consistency; global stats capture domain shift.
- **Expected win case**: High disagreement between TDA and FreeTTA = complementary signals.
- **Expected fail case**: When both methods are partially wrong, hybrid inherits both errors.


## Section 11: Synthesis & Deep Insights

# Section 11 — Synthesis: Deep Insights

This section derives evidence-based observations from the computed metrics.
Conclusions are grounded in numbers from the data, not assumptions.

## 1. Accuracy Insights

### CALTECH
- CLIP baseline: 0.9355  TDA: 0.9343 (-0.0012)  FreeTTA: 0.9375 (+0.0020)  Hybrid: 0.9387
- Best adapter: **hybrid**
- TDA and FreeTTA are roughly on par for this dataset.

### DTD
- CLIP baseline: 0.4394  TDA: 0.4606 (+0.0213)  FreeTTA: 0.4596 (+0.0202)  Hybrid: 0.4479
- Best adapter: **ent_tda**
- TDA and FreeTTA are roughly on par for this dataset.

### EUROSAT
- CLIP baseline: 0.4843  TDA: 0.6215 (+0.1372)  FreeTTA: 0.5974 (+0.1131)  Hybrid: 0.6156
- Best adapter: **ent_tda**
- TDA outperforms FreeTTA by 0.0241.  Local cache memory is more effective, suggesting the test stream has within-class consistency that the cache can exploit.

### IMAGENET
- CLIP baseline: 0.6235  TDA: 0.6289 (+0.0054)  FreeTTA: 0.6268 (+0.0033)  Hybrid: 0.6250
- Best adapter: **ent_tda**
- TDA and FreeTTA are roughly on par for this dataset.

### PETS
- CLIP baseline: 0.8839  TDA: 0.8964 (+0.0125)  FreeTTA: 0.8893 (+0.0055)  Hybrid: 0.8907
- Best adapter: **tda**
- TDA and FreeTTA are roughly on par for this dataset.
## 2. Flip / Correction Efficiency Insights

### CALTECH
- **tda**: change_rate=0.019 CE=0.413  net_correction=-0.0012
- **freetta**: change_rate=0.002 CE=1.000  net_correction=+0.0020
  → High CE (1.000): when freetta changes a prediction it is usually right.
- **conf_ftta**: change_rate=0.002 CE=1.000  net_correction=+0.0020
  → High CE (1.000): when conf_ftta changes a prediction it is usually right.
- **ent_tda**: change_rate=0.018 CE=0.467  net_correction=+0.0004
- **hybrid**: change_rate=0.020 CE=0.520  net_correction=+0.0032

### DTD
- **tda**: change_rate=0.127 CE=0.298  net_correction=+0.0213
  → Low CE (0.298): tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **freetta**: change_rate=0.168 CE=0.273  net_correction=+0.0202
  → Low CE (0.273): freetta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **conf_ftta**: change_rate=0.215 CE=0.205  net_correction=-0.0016
  → Low CE (0.205): conf_ftta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **ent_tda**: change_rate=0.128 CE=0.311  net_correction=+0.0245
  → Low CE (0.311): ent_tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **hybrid**: change_rate=0.142 CE=0.236  net_correction=+0.0085
  → Low CE (0.236): hybrid makes many harmful changes — the cache/statistics are corrupting more than they help.

### EUROSAT
- **tda**: change_rate=0.385 CE=0.475  net_correction=+0.1372
- **freetta**: change_rate=0.334 CE=0.449  net_correction=+0.1131
- **conf_ftta**: change_rate=0.371 CE=0.397  net_correction=+0.1021
  → Low CE (0.397): conf_ftta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **ent_tda**: change_rate=0.387 CE=0.475  net_correction=+0.1383
- **hybrid**: change_rate=0.340 CE=0.485  net_correction=+0.1312

### IMAGENET
- **tda**: change_rate=0.038 CE=0.315  net_correction=+0.0054
  → Low CE (0.315): tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **freetta**: change_rate=0.038 CE=0.307  net_correction=+0.0033
  → Low CE (0.307): freetta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **conf_ftta**: change_rate=0.046 CE=0.284  net_correction=+0.0019
  → Low CE (0.284): conf_ftta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **ent_tda**: change_rate=0.027 CE=0.399  net_correction=+0.0063
  → Low CE (0.399): ent_tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **hybrid**: change_rate=0.031 CE=0.316  net_correction=+0.0015
  → Low CE (0.316): hybrid makes many harmful changes — the cache/statistics are corrupting more than they help.

### PETS
- **tda**: change_rate=0.040 CE=0.600  net_correction=+0.0125
- **freetta**: change_rate=0.016 CE=0.593  net_correction=+0.0055
- **conf_ftta**: change_rate=0.017 CE=0.516  net_correction=+0.0035
- **ent_tda**: change_rate=0.036 CE=0.591  net_correction=+0.0095
- **hybrid**: change_rate=0.020 CE=0.635  net_correction=+0.0068
## 3. Entropy / Confidence Insights

### CALTECH
- CLIP: confidence(correct)=0.922  confidence(wrong)=0.614  entropy(wrong)=1.089
- TDA: confidence(correct)=0.971  confidence(wrong)=0.767  entropy(wrong)=0.622
  → TDA produces overconfident wrong predictions (conf=0.767).  OER risk is elevated.
- FREETTA: confidence(correct)=0.932  confidence(wrong)=0.623  entropy(wrong)=1.033

### DTD
- CLIP: confidence(correct)=0.595  confidence(wrong)=0.323  entropy(wrong)=2.555
- TDA: confidence(correct)=0.717  confidence(wrong)=0.442  entropy(wrong)=2.087
- FREETTA: confidence(correct)=0.726  confidence(wrong)=0.454  entropy(wrong)=1.943

### EUROSAT
- CLIP: confidence(correct)=0.599  confidence(wrong)=0.335  entropy(wrong)=1.809
- TDA: confidence(correct)=0.775  confidence(wrong)=0.568  entropy(wrong)=1.196
- FREETTA: confidence(correct)=0.822  confidence(wrong)=0.615  entropy(wrong)=0.969

### IMAGENET
- CLIP: confidence(correct)=0.758  confidence(wrong)=0.471  entropy(wrong)=2.050
- TDA: confidence(correct)=0.828  confidence(wrong)=0.557  entropy(wrong)=1.828
- FREETTA: confidence(correct)=0.781  confidence(wrong)=0.501  entropy(wrong)=1.844

### PETS
- CLIP: confidence(correct)=0.875  confidence(wrong)=0.552  entropy(wrong)=1.133
- TDA: confidence(correct)=0.898  confidence(wrong)=0.594  entropy(wrong)=1.024
- FREETTA: confidence(correct)=0.887  confidence(wrong)=0.570  entropy(wrong)=1.066
## 4. Disagreement Analysis Insights

### CALTECH
- Disagreement rate: 0.017.  When they disagree: TDA wins 15 times, FreeTTA wins 23 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.

### DTD
- Disagreement rate: 0.144.  When they disagree: TDA wins 60 times, FreeTTA wins 58 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.

### EUROSAT
- Disagreement rate: 0.233.  When they disagree: TDA wins 746 times, FreeTTA wins 551 times.
  → High disagreement rate: the two methods respond very differently to the same inputs — their mechanisms are genuinely complementary.

### IMAGENET
- Disagreement rate: 0.060.  When they disagree: TDA wins 164 times, FreeTTA wins 143 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.

### PETS
- Disagreement rate: 0.037.  When they disagree: TDA wins 72 times, FreeTTA wins 46 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.
## 5. Improvement Method Insights

### CALTECH
- ConfGatedFreeTTA vs FreeTTA: +0.0000 → **NEUTRAL**
- EntropyGatedTDA vs TDA: +0.0016 → **IMPROVED**
- Hybrid vs best(TDA, FreeTTA): +0.0012 → **IMPROVED**
  → Hybrid gains from combining local cache (short-range) and global statistics (domain shift).  The two mechanisms are complementary here.

### DTD
- ConfGatedFreeTTA vs FreeTTA: -0.0218 → **DEGRADED**
  → Gating the M-step hurt FreeTTA here.  Discarding uncertain updates removed too much signal, starving the class means of the adaptation they need.
- EntropyGatedTDA vs TDA: +0.0032 → **IMPROVED**
- Hybrid vs best(TDA, FreeTTA): -0.0128 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

### EUROSAT
- ConfGatedFreeTTA vs FreeTTA: -0.0110 → **DEGRADED**
  → Gating the M-step hurt FreeTTA here.  Discarding uncertain updates removed too much signal, starving the class means of the adaptation they need.
- EntropyGatedTDA vs TDA: +0.0011 → **IMPROVED**
- Hybrid vs best(TDA, FreeTTA): -0.0059 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

### IMAGENET
- ConfGatedFreeTTA vs FreeTTA: -0.0014 → **DEGRADED**
- EntropyGatedTDA vs TDA: +0.0009 → **NEUTRAL**
- Hybrid vs best(TDA, FreeTTA): -0.0039 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

### PETS
- ConfGatedFreeTTA vs FreeTTA: -0.0019 → **DEGRADED**
- EntropyGatedTDA vs TDA: -0.0030 → **DEGRADED**
  → Adaptive entropy gating shrank the cache too aggressively.  The original fixed threshold already selects good samples.
- Hybrid vs best(TDA, FreeTTA): -0.0057 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

## 6. Novel Metric Commentary (Section 10)

### Stability Scores
| dataset   |   clip_stability |   tda_stability |   freetta_stability |   conf_ftta_stability |   ent_tda_stability |   hybrid_stability |
|:----------|-----------------:|----------------:|--------------------:|----------------------:|--------------------:|-------------------:|
| caltech   |         0.96809  |        0.96814  |            0.968993 |              0.968993 |            0.9688   |           0.968341 |
| dtd       |         0.93398  |        0.929607 |            0.921535 |              0.921109 |            0.929559 |           0.934345 |
| eurosat   |         0.93648  |        0.928755 |            0.926563 |              0.928368 |            0.930053 |           0.930566 |
| imagenet  |         0.939155 |        0.938639 |            0.939833 |              0.939843 |            0.939236 |           0.939787 |
| pets      |         0.954016 |        0.959068 |            0.954085 |              0.953205 |            0.958266 |           0.956402 |
- caltech: most stable method = **freetta**
- dtd: most stable method = **clip**
- eurosat: most stable method = **clip**
- imagenet: most stable method = **freetta**
- pets: most stable method = **tda**

### Overconfidence Error Rate (OER > 0.1 highlighted)
| dataset   | method    |      oer |
|:----------|:----------|---------:|
| caltech   | tda       | 0.358025 |
| caltech   | ent_tda   | 0.379747 |
| caltech   | hybrid    | 0.192053 |
| eurosat   | freetta   | 0.135235 |
| eurosat   | conf_ftta | 0.105672 |
| imagenet  | tda       | 0.112369 |
| imagenet  | ent_tda   | 0.112642 |
High OER entries indicate methods that produce confident wrong predictions.  This is a calibration failure and is dangerous in deployment.

## 7. Summary Hypotheses
Based on the above analysis, the following hypotheses are proposed:

1. **FreeTTA effectiveness scales with domain shift magnitude**: datasets with larger domain gap from ImageNet training benefit more from global statistics adaptation (EuroSAT > DTD > Caltech).

2. **TDA cache saturation limits long-stream gains**: the fixed capacity (pos_shot_capacity * C slots) caps how much the cache can represent.  On large-class datasets (ImageNet, 1000 classes) the cache is too sparse per class to help much.

3. **Confidence gating is a double-edged sword**: on datasets where CLIP is uncertain frequently (DTD textures), gating the M-step removes most of the adaptation signal and hurts FreeTTA.  On datasets where CLIP is mostly confident (Caltech), gating is nearly neutral.

4. **Hybrid gains require the two methods to be complementary**: when TDA and FreeTTA disagree frequently the hybrid can benefit from both.  When they agree, combining them adds noise without signal.

## Generated Plots

- `accuracy_heatmap.png`
- `caltech/conf_ftta_skip_rate.png`
- `caltech/entropy_confidence.png`
- `caltech/freetta_drift_comparison.png`
- `caltech/freetta_internals.png`
- `caltech/lmm_analysis.png`
- `caltech/oer_comparison.png`
- `caltech/pca_logit_visualization.png`
- `caltech/prediction_change.png`
- `caltech/tda_internals.png`
- `caltech/trajectory_all_methods.png`
- `caltech/trajectory_core.png`
- `correction_efficiency.png`
- `difficulty_split.png`
- `disagreement_summary.png`
- `dtd/conf_ftta_skip_rate.png`
- `dtd/entropy_confidence.png`
- `dtd/freetta_drift_comparison.png`
- `dtd/freetta_internals.png`
- `dtd/lmm_analysis.png`
- `dtd/oer_comparison.png`
- `dtd/pca_logit_visualization.png`
- `dtd/prediction_change.png`
- `dtd/tda_internals.png`
- `dtd/trajectory_all_methods.png`
- `dtd/trajectory_core.png`
- `eurosat/conf_ftta_skip_rate.png`
- `eurosat/entropy_confidence.png`
- `eurosat/freetta_drift_comparison.png`
- `eurosat/freetta_internals.png`
- `eurosat/lmm_analysis.png`
- `eurosat/oer_comparison.png`
- `eurosat/pca_logit_visualization.png`
- `eurosat/prediction_change.png`
- `eurosat/tda_internals.png`
- `eurosat/trajectory_all_methods.png`
- `eurosat/trajectory_core.png`
- `failure_buckets.png`
- `imagenet/conf_ftta_skip_rate.png`
- `imagenet/entropy_confidence.png`
- `imagenet/freetta_drift_comparison.png`
- `imagenet/freetta_internals.png`
- `imagenet/lmm_analysis.png`
- `imagenet/oer_comparison.png`
- `imagenet/pca_logit_visualization.png`
- `imagenet/prediction_change.png`
- `imagenet/tda_internals.png`
- `imagenet/trajectory_all_methods.png`
- `imagenet/trajectory_core.png`
- `improvement_comparison.png`
- `improvement_deltas.png`
- `pets/conf_ftta_skip_rate.png`
- `pets/entropy_confidence.png`
- `pets/freetta_drift_comparison.png`
- `pets/freetta_internals.png`
- `pets/lmm_analysis.png`
- `pets/oer_comparison.png`
- `pets/pca_logit_visualization.png`
- `pets/prediction_change.png`
- `pets/tda_internals.png`
- `pets/trajectory_all_methods.png`
- `pets/trajectory_core.png`
- `stability_scores.png`

## Run Metadata

```json
{
  "datasets": [
    "dtd",
    "caltech",
    "eurosat",
    "pets",
    "imagenet"
  ],
  "device": "cpu",
  "seed": 42,
  "rolling_window": 50,
  "total_samples": 26114,
  "elapsed_seconds": 201.5,
  "features_dir": "data/processed",
  "output_dir": "outputs/research_pipeline"
}
```