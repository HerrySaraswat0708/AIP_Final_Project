# Dataset Report: eurosat

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.484321 |   0.53037 |      0.510617 |          0.0460494 |              0.0262963 |          -0.0197531 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| eurosat   |              0.782593 |         0.898025 |                  -0.115432 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| eurosat   | tda      |      8100 |                      3176 |                    1544 |                    1120 |                  747 |                        1513 |                 0.392099 |               0.190617 |      0.417284 |                    0.331361 |                           0.190415  |                    373 |             0.0460494 |                             1.42067 |                          1.17947 |
| eurosat   | freetta  |      8100 |                      3651 |                    2272 |                     485 |                  272 |                        1420 |                 0.450741 |               0.280494 |      0.268765 |                    0.222784 |                           0.0693347 |                    213 |             0.0262963 |                             2.30228 |                          2.30223 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| eurosat   | clip     | all      |      8100 |       2.30241  |         2.30244  |   0.000136929 |          0.102754 |            0.102399 |       0.00155294 |
| eurosat   | clip     | correct  |      3923 |       2.30236  |         2.30235  |   0.000128655 |          0.103611 |            0.103567 |       0.00152875 |
| eurosat   | clip     | wrong    |      4177 |       2.30247  |         2.30252  |   0.000123283 |          0.101948 |            0.101605 |       0.00106878 |
| eurosat   | tda      | all      |      8100 |       0.964308 |         0.965372 |   0.594294    |          0.658134 |            0.656333 |       0.238904   |
| eurosat   | tda      | correct  |      4296 |       0.771472 |         0.665567 |   0.604709    |          0.734557 |            0.800167 |       0.237454   |
| eurosat   | tda      | wrong    |      3804 |       1.18209  |         1.22576  |   0.499661    |          0.571826 |            0.546427 |       0.20931    |
| eurosat   | freetta  | all      |      8100 |       2.30224  |         2.3023   |   0.000262681 |          0.104025 |            0.103666 |       0.00208708 |
| eurosat   | freetta  | correct  |      4136 |       2.30215  |         2.30216  |   0.000254968 |          0.105061 |            0.104975 |       0.00201564 |
| eurosat   | freetta  | wrong    |      3964 |       2.30233  |         2.30243  |   0.000238542 |          0.102943 |            0.102458 |       0.00153989 |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| eurosat   |            0.393457 |                  0.298714 |                       0.24851 |                            2.30247 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| eurosat   |       50 |                      943 |                            5 |                           5 |                0.11642 |                0.000617284 |                       0.000617284 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| eurosat   |                        29.1616 |                         17.191 |                        0 |                        162 |              1.00086e-05 |                   9.11446e-06 |                0.0654586 |                       2.30254 |                     529.465 |                  -0.115432 |              0.782593 |         0.898025 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`