# Dataset Report: PETS

## Accuracy
|   clip |    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|-------:|----------:|------------:|----------:|---------:|
| 0.8839 | 0.8964 |    0.8893 |      0.8874 |    0.8934 |   0.8907 |

### Gains over CLIP
|    tda |   freetta |   conf_ftta |   ent_tda |   hybrid |
|-------:|----------:|------------:|----------:|---------:|
| 0.0125 |    0.0055 |      0.0035 |    0.0095 |   0.0068 |

## Prediction Change Metrics (Section 3)
| dataset   | method    |   n_samples |   n_unchanged_correct |   n_unchanged_wrong |   n_beneficial |   n_harmful |   n_other_changed_wrong |   change_rate |   beneficial_rate |   harmful_rate |   net_correction_score |   net_correction_rate |   correction_efficiency |   harm_rate_on_clip_correct |
|:----------|:----------|------------:|----------------------:|--------------------:|---------------:|------------:|------------------------:|--------------:|------------------:|---------------:|-----------------------:|----------------------:|------------------------:|----------------------------:|
| pets      | tda       |        3669 |                  3202 |                 322 |             87 |          41 |                      17 |        0.0395 |            0.0237 |         0.0112 |                     46 |                0.0125 |                  0.6000 |                      0.0126 |
| pets      | freetta   |        3669 |                  3228 |                 382 |             35 |          15 |                       9 |        0.0161 |            0.0095 |         0.0041 |                     20 |                0.0055 |                  0.5932 |                      0.0046 |
| pets      | conf_ftta |        3669 |                  3224 |                 383 |             32 |          19 |                      11 |        0.0169 |            0.0087 |         0.0052 |                     13 |                0.0035 |                  0.5161 |                      0.0059 |
| pets      | ent_tda   |        3669 |                  3200 |                 337 |             78 |          43 |                      11 |        0.0360 |            0.0213 |         0.0117 |                     35 |                0.0095 |                  0.5909 |                      0.0133 |
| pets      | hybrid    |        3669 |                  3221 |                 374 |             47 |          22 |                       5 |        0.0202 |            0.0128 |         0.0060 |                     25 |                0.0068 |                  0.6351 |                      0.0068 |

## Entropy / Confidence (Section 4)
| method    |   ('mean_confidence', 'all') |   ('mean_confidence', 'correct') |   ('mean_confidence', 'wrong') |   ('mean_entropy', 'all') |   ('mean_entropy', 'correct') |   ('mean_entropy', 'wrong') |
|:----------|-----------------------------:|---------------------------------:|-------------------------------:|--------------------------:|------------------------------:|----------------------------:|
| clip      |                     0.837536 |                         0.875005 |                       0.552298 |                  0.488659 |                      0.404048 |                     1.13278 |
| conf_ftta |                     0.852514 |                         0.888825 |                       0.566245 |                  0.433158 |                      0.35129  |                     1.07858 |
| ent_tda   |                     0.864512 |                         0.897446 |                       0.588402 |                  0.402155 |                      0.327473 |                     1.02826 |
| freetta   |                     0.852402 |                         0.887486 |                       0.570437 |                  0.433395 |                      0.354633 |                     1.0664  |
| hybrid    |                     0.855136 |                         0.888946 |                       0.579594 |                  0.424788 |                      0.347517 |                     1.05452 |
| tda       |                     0.866176 |                         0.897671 |                       0.593584 |                  0.399888 |                      0.327726 |                     1.02447 |

## Disagreement Analysis (Section 7)
| dataset   |   disagreement_rate |   n_disagree |   tda_acc_on_disagree |   freetta_acc_on_disagree |   clip_acc_on_disagree |   mean_clip_entropy_on_disagree |   tda_wins |   freetta_wins |
|:----------|--------------------:|-------------:|----------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------:|---------------:|
| pets      |           0.0373399 |          137 |              0.525547 |                  0.335766 |               0.335766 |                          1.2322 |         72 |             46 |

## TDA Internal (Section 6)
| dataset   |   final_pos_cache |   final_neg_cache |   mean_gate_rate |   max_pos_cache |
|:----------|------------------:|------------------:|-----------------:|----------------:|
| pets      |               111 |                70 |         0.260016 |             111 |

## FreeTTA Internal (Section 6)
| dataset   |   mean_em_weight |   final_mu_drift |   final_prior_entropy |   final_sigma_trace |   mean_mu_update_norm |
|:----------|-----------------:|-----------------:|----------------------:|--------------------:|----------------------:|
| pets      |         0.650374 |          1.12215 |               2.58808 |            0.370294 |           0.000872927 |

## Novel Metrics (Section 10)
### Correction Efficiency
| method    |   n_beneficial |   n_harmful |   correction_efficiency |
|:----------|---------------:|------------:|------------------------:|
| tda       |             87 |          41 |                  0.6000 |
| freetta   |             35 |          15 |                  0.5932 |
| conf_ftta |             32 |          19 |                  0.5161 |
| ent_tda   |             78 |          43 |                  0.5909 |
| hybrid    |             47 |          22 |                  0.6351 |

### Overconfidence Error Rate
| method    |   n_wrong |    oer |
|:----------|----------:|-------:|
| clip      |       426 | 0.0352 |
| tda       |       380 | 0.0763 |
| freetta   |       406 | 0.0468 |
| conf_ftta |       413 | 0.0436 |
| ent_tda   |       391 | 0.0818 |
| hybrid    |       401 | 0.0524 |

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