# Dataset Report: EUROSAT

## Accuracy
|   clip |    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|-------:|----------:|------------:|----------:|---------:|
| 0.4843 | 0.6215 |    0.5974 |      0.5864 |    0.6226 |   0.6156 |

### Gains over CLIP
|    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|----------:|------------:|----------:|---------:|
| 0.1372 |    0.1131 |      0.1021 |    0.1383 |   0.1312 |

## Prediction Change Metrics (Section 3)
| dataset   | method    |   n_samples |   n_unchanged_correct |   n_unchanged_wrong |   n_beneficial |   n_harmful |   n_other_changed_wrong |   change_rate |   beneficial_rate |   harmful_rate |   net_correction_score |   net_correction_rate |   correction_efficiency |   harm_rate_on_clip_correct |
|:----------|:----------|------------:|----------------------:|--------------------:|---------------:|------------:|------------------------:|--------------:|------------------:|---------------:|-----------------------:|----------------------:|------------------------:|----------------------------:|
| eurosat   | tda       |        8100 |                  3553 |                1426 |           1481 |         370 |                    1270 |        0.3853 |            0.1828 |         0.0457 |                   1111 |                0.1372 |                  0.4745 |                      0.0943 |
| eurosat   | freetta   |        8100 |                  3624 |                1772 |           1215 |         299 |                    1190 |        0.3338 |            0.1500 |         0.0369 |                    916 |                0.1131 |                  0.4493 |                      0.0762 |
| eurosat   | conf_ftta |        8100 |                  3555 |                1538 |           1195 |         368 |                    1444 |        0.3712 |            0.1475 |         0.0454 |                    827 |                0.1021 |                  0.3974 |                      0.0938 |
| eurosat   | ent_tda   |        8100 |                  3554 |                1410 |           1489 |         369 |                    1278 |        0.3872 |            0.1838 |         0.0456 |                   1120 |                0.1383 |                  0.4748 |                      0.0941 |
| eurosat   | hybrid    |        8100 |                  3653 |                1696 |           1333 |         270 |                    1148 |        0.3396 |            0.1646 |         0.0333 |                   1063 |                0.1312 |                  0.4846 |                      0.0688 |

## Entropy / Confidence (Section 4)
| method    |   ('mean_confidence', 'all') |   ('mean_confidence', 'correct') |   ('mean_confidence', 'wrong') |   ('mean_entropy', 'all') |   ('mean_entropy', 'correct') |   ('mean_entropy', 'wrong') |
|:----------|-----------------------------:|---------------------------------:|-------------------------------:|--------------------------:|------------------------------:|----------------------------:|
| clip      |                     0.462752 |                         0.598945 |                       0.33484  |                  1.49607  |                      1.16326  |                    1.80864  |
| conf_ftta |                     0.718278 |                         0.805511 |                       0.594589 |                  0.743778 |                      0.531886 |                    1.04422  |
| ent_tda   |                     0.697633 |                         0.775685 |                       0.568874 |                  0.884157 |                      0.69643  |                    1.19384  |
| freetta   |                     0.73889  |                         0.822095 |                       0.61542  |                  0.673866 |                      0.475092 |                    0.968825 |
| hybrid    |                     0.717727 |                         0.800673 |                       0.584918 |                  0.786396 |                      0.587452 |                    1.10494  |
| tda       |                     0.696565 |                         0.775146 |                       0.567543 |                  0.88622  |                      0.697686 |                    1.19577  |

## Disagreement Analysis (Section 7)
| dataset   |   disagreement_rate |   n_disagree |   tda_acc_on_disagree |   freetta_acc_on_disagree |   clip_acc_on_disagree |   mean_clip_entropy_on_disagree |   tda_wins |   freetta_wins |
|:----------|--------------------:|-------------:|----------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------:|---------------:|
| eurosat   |            0.233457 |         1891 |                0.3945 |                   0.29138 |               0.207827 |                          1.8033 |        746 |            551 |

## TDA Internal (Section 6)
| dataset   |   final_pos_cache |   final_neg_cache |   mean_gate_rate |   max_pos_cache |
|:----------|------------------:|------------------:|-----------------:|----------------:|
| eurosat   |                30 |                18 |         0.206667 |              30 |

## FreeTTA Internal (Section 6)
| dataset   |   mean_em_weight |   final_mu_drift |   final_prior_entropy |   final_sigma_trace |   mean_mu_update_norm |
|:----------|-----------------:|-----------------:|----------------------:|--------------------:|----------------------:|
| eurosat   |         0.197767 |          1.14137 |              0.716778 |            0.348555 |           0.000400408 |

## Novel Metrics (Section 10)
### Correction Efficiency
| method    |   n_beneficial |   n_harmful |   correction_efficiency |
|:----------|---------------:|------------:|------------------------:|
| tda       |           1481 |         370 |                  0.4745 |
| freetta   |           1215 |         299 |                  0.4493 |
| conf_ftta |           1195 |         368 |                  0.3974 |
| ent_tda   |           1489 |         369 |                  0.4748 |
| hybrid    |           1333 |         270 |                  0.4846 |

### Overconfidence Error Rate
| method    |   n_wrong |    oer |
|:----------|----------:|-------:|
| clip      |      4177 | 0.0053 |
| tda       |      3066 | 0.0639 |
| freetta   |      3261 | 0.1352 |
| conf_ftta |      3350 | 0.1057 |
| ent_tda   |      3057 | 0.0677 |
| hybrid    |      3114 | 0.0665 |

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