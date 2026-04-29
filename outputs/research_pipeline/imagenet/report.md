# Dataset Report: IMAGENET

## Accuracy
|   clip |    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|-------:|----------:|------------:|----------:|---------:|
| 0.6235 | 0.6289 |    0.6268 |      0.6254 |    0.6298 |    0.625 |

### Gains over CLIP
|    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|----------:|------------:|----------:|---------:|
| 0.0054 |    0.0033 |      0.0019 |    0.0063 |   0.0015 |

## Prediction Change Metrics (Section 3)
| dataset   | method    |   n_samples |   n_unchanged_correct |   n_unchanged_wrong |   n_beneficial |   n_harmful |   n_other_changed_wrong |   change_rate |   beneficial_rate |   harmful_rate |   net_correction_score |   net_correction_rate |   correction_efficiency |   harm_rate_on_clip_correct |
|:----------|:----------|------------:|----------------------:|--------------------:|---------------:|------------:|------------------------:|--------------:|------------------:|---------------:|-----------------------:|----------------------:|------------------------:|----------------------------:|
| imagenet  | tda       |       10000 |                  6169 |                3450 |            120 |          66 |                     195 |        0.0381 |            0.0120 |         0.0066 |                     54 |                0.0054 |                  0.3150 |                      0.0106 |
| imagenet  | freetta   |       10000 |                  6151 |                3468 |            117 |          84 |                     180 |        0.0381 |            0.0117 |         0.0084 |                     33 |                0.0033 |                  0.3071 |                      0.0135 |
| imagenet  | conf_ftta |       10000 |                  6122 |                3413 |            132 |         113 |                     220 |        0.0465 |            0.0132 |         0.0113 |                     19 |                0.0019 |                  0.2839 |                      0.0181 |
| imagenet  | ent_tda   |       10000 |                  6190 |                3539 |            108 |          45 |                     118 |        0.0271 |            0.0108 |         0.0045 |                     63 |                0.0063 |                  0.3985 |                      0.0072 |
| imagenet  | hybrid    |       10000 |                  6151 |                3536 |             99 |          84 |                     130 |        0.0313 |            0.0099 |         0.0084 |                     15 |                0.0015 |                  0.3163 |                      0.0135 |

## Entropy / Confidence (Section 4)
| method    |   ('mean_confidence', 'all') |   ('mean_confidence', 'correct') |   ('mean_confidence', 'wrong') |   ('mean_entropy', 'all') |   ('mean_entropy', 'correct') |   ('mean_entropy', 'wrong') |
|:----------|-----------------------------:|---------------------------------:|-------------------------------:|--------------------------:|------------------------------:|----------------------------:|
| clip      |                     0.649993 |                         0.757885 |                       0.471321 |                   1.38067 |                      0.97651  |                     2.04998 |
| conf_ftta |                     0.676679 |                         0.781757 |                       0.501251 |                   1.21247 |                      0.834503 |                     1.8435  |
| ent_tda   |                     0.727114 |                         0.82506  |                       0.560485 |                   1.13433 |                      0.7467   |                     1.79379 |
| freetta   |                     0.676228 |                         0.780661 |                       0.500828 |                   1.21382 |                      0.838511 |                     1.84417 |
| hybrid    |                     0.669128 |                         0.775623 |                       0.491635 |                   1.26781 |                      0.878653 |                     1.91642 |
| tda       |                     0.726996 |                         0.827567 |                       0.556558 |                   1.14331 |                      0.739417 |                     1.82778 |

## Disagreement Analysis (Section 7)
| dataset   |   disagreement_rate |   n_disagree |   tda_acc_on_disagree |   freetta_acc_on_disagree |   clip_acc_on_disagree |   mean_clip_entropy_on_disagree |   tda_wins |   freetta_wins |
|:----------|--------------------:|-------------:|----------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------:|---------------:|
| imagenet  |              0.0596 |          596 |              0.275168 |                  0.239933 |               0.216443 |                         2.56982 |        164 |            143 |

## TDA Internal (Section 6)
| dataset   |   final_pos_cache |   final_neg_cache |   mean_gate_rate |   max_pos_cache |
|:----------|------------------:|------------------:|-----------------:|----------------:|
| imagenet  |              2968 |              1675 |           0.3594 |            2968 |

## FreeTTA Internal (Section 6)
| dataset   |   mean_em_weight |   final_mu_drift |   final_prior_entropy |   final_sigma_trace |   mean_mu_update_norm |
|:----------|-----------------:|-----------------:|----------------------:|--------------------:|----------------------:|
| imagenet  |         0.532558 |          1.12797 |               3.96526 |            0.363754 |           0.000416698 |

## Novel Metrics (Section 10)
### Correction Efficiency
| method    |   n_beneficial |   n_harmful |   correction_efficiency |
|:----------|---------------:|------------:|------------------------:|
| tda       |            120 |          66 |                  0.3150 |
| freetta   |            117 |          84 |                  0.3071 |
| conf_ftta |            132 |         113 |                  0.2839 |
| ent_tda   |            108 |          45 |                  0.3985 |
| hybrid    |             99 |          84 |                  0.3163 |

### Overconfidence Error Rate
| method    |   n_wrong |    oer |
|:----------|----------:|-------:|
| clip      |      3765 | 0.0428 |
| tda       |      3711 | 0.1124 |
| freetta   |      3732 | 0.0616 |
| conf_ftta |      3746 | 0.0627 |
| ent_tda   |      3702 | 0.1126 |
| hybrid    |      3750 | 0.0560 |

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