# Dataset Report: eurosat

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.484321 |  0.533333 |      0.593457 |          0.0490123 |               0.109136 |           0.0601235 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| eurosat   |              0.782593 |         0.898025 |                  -0.115432 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| eurosat   | tda      |      8100 |                      3195 |                    1510 |                    1125 |                  728 |                        1542 |                 0.394444 |               0.18642  |      0.419136 |                    0.33137  |                           0.185572  |                    397 |             0.0490123 |                            1.40253  |                          1.191   |
| eurosat   | freetta  |      8100 |                      3549 |                    2044 |                    1258 |                  374 |                         875 |                 0.438148 |               0.252346 |      0.309506 |                    0.501795 |                           0.0953352 |                    884 |             0.109136  |                            0.556335 |                          1.39815 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| eurosat   | clip     | all      |      8100 |       2.30241  |         2.30244  |   0.000136927 |          0.102754 |            0.102399 |       0.00155294 |
| eurosat   | clip     | correct  |      3923 |       2.30236  |         2.30235  |   0.000128656 |          0.103611 |            0.103567 |       0.00152875 |
| eurosat   | clip     | wrong    |      4177 |       2.30247  |         2.30252  |   0.00012328  |          0.101948 |            0.101605 |       0.00106878 |
| eurosat   | tda      | all      |      8100 |       0.976941 |         0.982734 |   0.594225    |          0.651706 |            0.64511  |       0.239761   |
| eurosat   | tda      | correct  |      4320 |       0.775628 |         0.675342 |   0.599426    |          0.731953 |            0.79334  |       0.237325   |
| eurosat   | tda      | wrong    |      3780 |       1.20701  |         1.24917  |   0.496747    |          0.559995 |            0.53478  |       0.207467   |
| eurosat   | freetta  | all      |      8100 |       0.80229  |         0.843122 |   0.585724    |          0.692988 |            0.680023 |       0.246045   |
| eurosat   | freetta  | correct  |      4807 |       0.511585 |         0.297338 |   0.513093    |          0.811858 |            0.934593 |       0.217789   |
| eurosat   | freetta  | wrong    |      3293 |       1.22665  |         1.28446  |   0.395132    |          0.519466 |            0.49379  |       0.170098   |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| eurosat   |            0.309012 |                  0.228126 |                      0.422693 |                            2.30244 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| eurosat   |       50 |                      941 |                            8 |                           8 |               0.116173 |                0.000987654 |                       0.000987654 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| eurosat   |                        29.1616 |                        11.9036 |                        0 |                        162 |                 0.197767 |                   0.000382625 |                  1.14164 |                      0.716778 |                    0.348244 |                  -0.115432 |              0.782593 |         0.898025 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`