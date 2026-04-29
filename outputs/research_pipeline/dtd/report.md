# Dataset Report: DTD

## Accuracy
|   clip |    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|-------:|----------:|------------:|----------:|---------:|
| 0.4394 | 0.4606 |    0.4596 |      0.4378 |    0.4638 |   0.4479 |

### Gains over CLIP
|    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|----------:|------------:|----------:|---------:|
| 0.0213 |    0.0202 |     -0.0016 |    0.0245 |   0.0085 |

## Prediction Change Metrics (Section 3)
| dataset   | method    |   n_samples |   n_unchanged_correct |   n_unchanged_wrong |   n_beneficial |   n_harmful |   n_other_changed_wrong |   change_rate |   beneficial_rate |   harmful_rate |   net_correction_score |   net_correction_rate |   correction_efficiency |   harm_rate_on_clip_correct |
|:----------|:----------|------------:|----------------------:|--------------------:|---------------:|------------:|------------------------:|--------------:|------------------:|---------------:|-----------------------:|----------------------:|------------------------:|----------------------------:|
| dtd       | tda       |        1880 |                   795 |                 847 |             71 |          31 |                     136 |        0.1266 |            0.0378 |         0.0165 |                     40 |                0.0213 |                  0.2983 |                      0.0375 |
| dtd       | freetta   |        1880 |                   778 |                 787 |             86 |          48 |                     181 |        0.1676 |            0.0457 |         0.0255 |                     38 |                0.0202 |                  0.2730 |                      0.0581 |
| dtd       | conf_ftta |        1880 |                   740 |                 736 |             83 |          86 |                     235 |        0.2149 |            0.0441 |         0.0457 |                     -3 |               -0.0016 |                  0.2054 |                      0.1041 |
| dtd       | ent_tda   |        1880 |                   797 |                 842 |             75 |          29 |                     137 |        0.1282 |            0.0399 |         0.0154 |                     46 |                0.0245 |                  0.3112 |                      0.0351 |
| dtd       | hybrid    |        1880 |                   779 |                 834 |             63 |          47 |                     157 |        0.1420 |            0.0335 |         0.0250 |                     16 |                0.0085 |                  0.2360 |                      0.0569 |

## Entropy / Confidence (Section 4)
| method    |   ('mean_confidence', 'all') |   ('mean_confidence', 'correct') |   ('mean_confidence', 'wrong') |   ('mean_entropy', 'all') |   ('mean_entropy', 'correct') |   ('mean_entropy', 'wrong') |
|:----------|-----------------------------:|---------------------------------:|-------------------------------:|--------------------------:|------------------------------:|----------------------------:|
| clip      |                     0.442454 |                         0.59529  |                       0.32268  |                   2.15335 |                      1.6402   |                     2.5555  |
| conf_ftta |                     0.578816 |                         0.739642 |                       0.453594 |                   1.50885 |                      0.975504 |                     1.92413 |
| ent_tda   |                     0.571214 |                         0.713192 |                       0.448391 |                   1.64742 |                      1.15981  |                     2.06924 |
| freetta   |                     0.579039 |                         0.725969 |                       0.454091 |                   1.52807 |                      1.03993  |                     1.94318 |
| hybrid    |                     0.560956 |                         0.714995 |                       0.436003 |                   1.62896 |                      1.11129  |                     2.04888 |
| tda       |                     0.568805 |                         0.717214 |                       0.442057 |                   1.65399 |                      1.14668  |                     2.08725 |

## Disagreement Analysis (Section 7)
| dataset   |   disagreement_rate |   n_disagree |   tda_acc_on_disagree |   freetta_acc_on_disagree |   clip_acc_on_disagree |   mean_clip_entropy_on_disagree |   tda_wins |   freetta_wins |
|:----------|--------------------:|-------------:|----------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------:|---------------:|
| dtd       |            0.144149 |          271 |              0.221402 |                  0.214022 |               0.169742 |                         2.80657 |         60 |             58 |

## TDA Internal (Section 6)
| dataset   |   final_pos_cache |   final_neg_cache |   mean_gate_rate |   max_pos_cache |
|:----------|------------------:|------------------:|-----------------:|----------------:|
| dtd       |               135 |                57 |         0.226064 |             135 |

## FreeTTA Internal (Section 6)
| dataset   |   mean_em_weight |   final_mu_drift |   final_prior_entropy |   final_sigma_trace |   mean_mu_update_norm |
|:----------|-----------------:|-----------------:|----------------------:|--------------------:|----------------------:|
| dtd       |         0.464729 |          1.16037 |               2.08321 |            0.326716 |            0.00205376 |

## Novel Metrics (Section 10)
### Correction Efficiency
| method    |   n_beneficial |   n_harmful |   correction_efficiency |
|:----------|---------------:|------------:|------------------------:|
| tda       |             71 |          31 |                  0.2983 |
| freetta   |             86 |          48 |                  0.2730 |
| conf_ftta |             83 |          86 |                  0.2054 |
| ent_tda   |             75 |          29 |                  0.3112 |
| hybrid    |             63 |          47 |                  0.2360 |

### Overconfidence Error Rate
| method    |   n_wrong |    oer |
|:----------|----------:|-------:|
| clip      |      1054 | 0.0028 |
| tda       |      1014 | 0.0483 |
| freetta   |      1016 | 0.0679 |
| conf_ftta |      1057 | 0.0766 |
| ent_tda   |      1008 | 0.0526 |
| hybrid    |      1038 | 0.0472 |

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