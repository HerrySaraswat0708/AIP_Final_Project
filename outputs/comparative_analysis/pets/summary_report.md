# Dataset Report: pets

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.883892 |  0.884165 |      0.889616 |        0.000272554 |             0.00572363 |          0.00545108 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| pets      |              0.914963 |          0.83456 |                  0.0804034 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| pets      | tda      |      3669 |                      3192 |                     354 |                      52 |                   51 |                          20 |                 0.869992 |              0.0964841 |     0.0335241 |                    0.422764 |                           0.0157262 |                      1 |           0.000272554 |                             1.08937 |                          1.06186 |
| pets      | freetta  |      3669 |                      3208 |                     348 |                      56 |                   35 |                          22 |                 0.874353 |              0.0948487 |     0.0307986 |                    0.495575 |                           0.0107925 |                     21 |           0.00572363  |                             1.0737  |                          1.12585 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| pets      | clip     | all      |      3669 |       3.60982  |         3.6099   |   0.000411194 |         0.0304082 |           0.0303879 |      0.000700282 |
| pets      | clip     | correct  |      3243 |       3.60983  |         3.60991  |   0.000411695 |         0.0304729 |           0.0304406 |      0.000678452 |
| pets      | clip     | wrong    |       426 |       3.60979  |         3.60987  |   0.000406133 |         0.0299154 |           0.0298972 |      0.000666934 |
| pets      | tda      | all      |      3669 |       0.363868 |         0.170958 |   0.424431    |         0.878743  |           0.967931  |      0.170331    |
| pets      | tda      | correct  |      3244 |       0.288528 |         0.130774 |   0.356745    |         0.910948  |           0.977199  |      0.137803    |
| pets      | tda      | wrong    |       425 |       0.938935 |         0.905549 |   0.457933    |         0.632922  |           0.606162  |      0.192804    |
| pets      | freetta  | all      |      3669 |       0.379217 |         0.189102 |   0.421708    |         0.866074  |           0.961682  |      0.179425    |
| pets      | freetta  | correct  |      3264 |       0.301969 |         0.144179 |   0.354198    |         0.900347  |           0.973106  |      0.148016    |
| pets      | freetta  | wrong    |       405 |       1.00178  |         0.967645 |   0.405367    |         0.589852  |           0.571022  |      0.171216    |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| pets      |            0.032979 |                  0.355372 |                      0.520661 |                            3.60983 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| pets      |       50 |                      229 |                            5 |                           5 |              0.0624148 |                 0.00136277 |                        0.00136277 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| pets      |                        76.0619 |                        49.6329 |                        0 |                    19.8324 |                 0.650374 |                   0.000989191 |                    1.122 |                       2.58808 |                    0.370464 |                  0.0804034 |              0.914963 |          0.83456 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`