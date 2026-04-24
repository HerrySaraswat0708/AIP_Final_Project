# Dataset Report: pets

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.883892 |  0.883347 |      0.881167 |       -0.000545108 |            -0.00272554 |         -0.00218043 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| pets      |              0.914963 |          0.83456 |                  0.0804034 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| pets      | tda      |      3669 |                      3190 |                     354 |                      51 |                   53 |                          21 |                 0.869447 |              0.0964841 |     0.0340692 |                    0.408    |                           0.0163429 |                     -2 |          -0.000545108 |                             1.03716 |                          1.06381 |
| pets      | freetta  |      3669 |                      3183 |                     340 |                      50 |                   60 |                          36 |                 0.867539 |              0.0926683 |     0.0397929 |                    0.342466 |                           0.0185014 |                    -10 |          -0.00272554  |                             3.60949 |                          3.60957 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| pets      | clip     | all      |      3669 |       3.60982  |         3.6099   |   0.000411194 |         0.0304082 |           0.0303879 |      0.000700282 |
| pets      | clip     | correct  |      3243 |       3.60983  |         3.60991  |   0.000411695 |         0.0304729 |           0.0304406 |      0.000678452 |
| pets      | clip     | wrong    |       426 |       3.60979  |         3.60987  |   0.000406133 |         0.0299154 |           0.0298972 |      0.000666934 |
| pets      | tda      | all      |      3669 |       0.356814 |         0.163716 |   0.420828    |         0.880441  |           0.968865  |      0.168598    |
| pets      | tda      | correct  |      3241 |       0.280252 |         0.124791 |   0.34913     |         0.912862  |           0.978251  |      0.135529    |
| pets      | tda      | wrong    |       428 |       0.93657  |         0.870117 |   0.463271    |         0.63494   |           0.616959  |      0.190662    |
| pets      | freetta  | all      |      3669 |       3.60945  |         3.60951  |   0.00049195  |         0.0309949 |           0.0309966 |      0.000801405 |
| pets      | freetta  | correct  |      3233 |       3.60944  |         3.60951  |   0.000489748 |         0.031085  |           0.0310687 |      0.000758835 |
| pets      | freetta  | wrong    |       436 |       3.60947  |         3.60955  |   0.000507536 |         0.0303268 |           0.030363  |      0.000792591 |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| pets      |           0.0534206 |                  0.413265 |                      0.372449 |                            3.60989 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| pets      |       50 |                      231 |                          126 |                         116 |              0.0629599 |                  0.0343418 |                         0.0316162 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| pets      |                        76.0619 |                        46.1717 |                        0 |                    19.8324 |              8.81516e-08 |                   1.06073e-07 |              0.000275688 |                       3.61044 |                     523.822 |                  0.0804034 |              0.914963 |          0.83456 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`