# Dataset Report: CALTECH

## Accuracy
|   clip |    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|-------:|----------:|------------:|----------:|---------:|
| 0.9355 | 0.9343 |    0.9375 |      0.9375 |    0.9359 |   0.9387 |

### Gains over CLIP
|     tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|--------:|----------:|------------:|----------:|---------:|
| -0.0012 |     0.002 |       0.002 |    0.0004 |   0.0032 |

## Prediction Change Metrics (Section 3)
| dataset   | method    |   n_samples |   n_unchanged_correct |   n_unchanged_wrong |   n_beneficial |   n_harmful |   n_other_changed_wrong |   change_rate |   beneficial_rate |   harmful_rate |   net_correction_score |   net_correction_rate |   correction_efficiency |   harm_rate_on_clip_correct |
|:----------|:----------|------------:|----------------------:|--------------------:|---------------:|------------:|------------------------:|--------------:|------------------:|---------------:|-----------------------:|----------------------:|------------------------:|----------------------------:|
| caltech   | tda       |        2465 |                  2284 |                 135 |             19 |          22 |                       5 |        0.0187 |            0.0077 |         0.0089 |                     -3 |               -0.0012 |                  0.4130 |                      0.0095 |
| caltech   | freetta   |        2465 |                  2306 |                 154 |              5 |           0 |                       0 |        0.0020 |            0.0020 |         0.0000 |                      5 |                0.0020 |                  1.0000 |                      0.0000 |
| caltech   | conf_ftta |        2465 |                  2306 |                 154 |              5 |           0 |                       0 |        0.0020 |            0.0020 |         0.0000 |                      5 |                0.0020 |                  1.0000 |                      0.0000 |
| caltech   | ent_tda   |        2465 |                  2286 |                 134 |             21 |          20 |                       4 |        0.0183 |            0.0085 |         0.0081 |                      1 |                0.0004 |                  0.4667 |                      0.0087 |
| caltech   | hybrid    |        2465 |                  2288 |                 127 |             26 |          18 |                       6 |        0.0203 |            0.0105 |         0.0073 |                      8 |                0.0032 |                  0.5200 |                      0.0078 |

## Entropy / Confidence (Section 4)
| method    |   ('mean_confidence', 'all') |   ('mean_confidence', 'correct') |   ('mean_confidence', 'wrong') |   ('mean_entropy', 'all') |   ('mean_entropy', 'correct') |   ('mean_entropy', 'wrong') |
|:----------|-----------------------------:|---------------------------------:|-------------------------------:|--------------------------:|------------------------------:|----------------------------:|
| clip      |                     0.901949 |                         0.921829 |                       0.613631 |                  0.37388  |                      0.324579 |                    1.08891  |
| conf_ftta |                     0.913028 |                         0.932387 |                       0.622519 |                  0.321126 |                      0.27367  |                    1.03327  |
| ent_tda   |                     0.957543 |                         0.970494 |                       0.768445 |                  0.150183 |                      0.118301 |                    0.6157   |
| freetta   |                     0.913085 |                         0.932442 |                       0.622603 |                  0.320805 |                      0.273346 |                    1.033    |
| hybrid    |                     0.936462 |                         0.952477 |                       0.691032 |                  0.22685  |                      0.186415 |                    0.846503 |
| tda       |                     0.957234 |                         0.970614 |                       0.767021 |                  0.151907 |                      0.118823 |                    0.622221 |

## Disagreement Analysis (Section 7)
| dataset   |   disagreement_rate |   n_disagree |   tda_acc_on_disagree |   freetta_acc_on_disagree |   clip_acc_on_disagree |   mean_clip_entropy_on_disagree |   tda_wins |   freetta_wins |
|:----------|--------------------:|-------------:|----------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------:|---------------:|
| caltech   |           0.0174442 |           43 |              0.348837 |                  0.534884 |               0.511628 |                         1.25374 |         15 |             23 |

## TDA Internal (Section 6)
| dataset   |   final_pos_cache |   final_neg_cache |   mean_gate_rate |   max_pos_cache |
|:----------|------------------:|------------------:|-----------------:|----------------:|
| caltech   |               300 |                86 |        0.0896552 |             300 |

## FreeTTA Internal (Section 6)
| dataset   |   mean_em_weight |   final_mu_drift |   final_prior_entropy |   final_sigma_trace |   mean_mu_update_norm |
|:----------|-----------------:|-----------------:|----------------------:|--------------------:|----------------------:|
| caltech   |         0.825633 |          1.13519 |               3.67739 |            0.355548 |             0.0012351 |

## Novel Metrics (Section 10)
### Correction Efficiency
| method    |   n_beneficial |   n_harmful |   correction_efficiency |
|:----------|---------------:|------------:|------------------------:|
| tda       |             19 |          22 |                  0.4130 |
| freetta   |              5 |           0 |                  1.0000 |
| conf_ftta |              5 |           0 |                  1.0000 |
| ent_tda   |             21 |          20 |                  0.4667 |
| hybrid    |             26 |          18 |                  0.5200 |

### Overconfidence Error Rate
| method    |   n_wrong |    oer |
|:----------|----------:|-------:|
| clip      |       159 | 0.0692 |
| tda       |       162 | 0.3580 |
| freetta   |       154 | 0.0779 |
| conf_ftta |       154 | 0.0779 |
| ent_tda   |       158 | 0.3797 |
| hybrid    |       151 | 0.1921 |

## Generated Plots
- `conf_ftta_skip_rate.png`
- `entropy_confidence.png`
- `freetta_drift_comparison.png`
- `freetta_internals.png`
- `lmm_analysis.png`
- `oer_comparison.png`
- `pca_logit_visualization.png`
- `prediction_change.png`
- `tda_internals.png`
- `trajectory_all_methods.png`
- `trajectory_core.png`