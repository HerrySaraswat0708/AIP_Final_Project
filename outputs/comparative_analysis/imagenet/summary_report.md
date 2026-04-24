# Dataset Report: imagenet

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|     0.6235 |     0.627 |        0.6235 |             0.0035 |                      0 |             -0.0035 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| imagenet  |                0.8272 |           0.4189 |                     0.4083 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| imagenet  | tda      |     10000 |                      6164 |                    3470 |                     106 |                   71 |                         189 |                   0.6164 |                 0.347  |        0.0366 |                    0.289617 |                           0.0113873 |                     35 |                0.0035 |                             2.09285 |                          2.28016 |
| imagenet  | freetta  |     10000 |                      6235 |                    3765 |                       0 |                    0 |                           0 |                   0.6235 |                 0.3765 |        0      |                    0        |                           0         |                      0 |                0      |                           nan       |                        nan       |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| imagenet  | clip     | all      |     10000 |       6.90732  |         6.90734  |   0.000144232 |        0.00114748 |          0.00114709 |      3.37564e-05 |
| imagenet  | clip     | correct  |      6235 |       6.90731  |         6.90733  |   0.000147412 |        0.00115696 |          0.00115677 |      3.18571e-05 |
| imagenet  | clip     | wrong    |      3765 |       6.90734  |         6.90735  |   0.000137589 |        0.00113178 |          0.00113049 |      3.08273e-05 |
| imagenet  | tda      | all      |     10000 |       1.1295   |         0.770337 |   1.11348     |        0.731909   |          0.811735   |      0.25722     |
| imagenet  | tda      | correct  |      6270 |       0.732665 |         0.404738 |   0.862604    |        0.829907   |          0.924729   |      0.205091    |
| imagenet  | tda      | wrong    |      3730 |       1.79657  |         1.55108  |   1.16769     |        0.567178   |          0.578052   |      0.251781    |
| imagenet  | freetta  | all      |     10000 |       6.90713  |         6.90716  |   0.000207647 |        0.00117934 |          0.00117871 |      4.16186e-05 |
| imagenet  | freetta  | correct  |      6235 |       6.90712  |         6.90715  |   0.000212092 |        0.00119102 |          0.00119071 |      3.93364e-05 |
| imagenet  | freetta  | wrong    |      3765 |       6.90715  |         6.90718  |   0.000198332 |        0.00116    |          0.00115833 |      3.792e-05   |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| imagenet  |              0.0366 |                  0.289617 |                      0.193989 |                            6.90734 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip | freetta_break_even_vs_clip   |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|:-----------------------------|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| imagenet  |       50 |                      178 |                              |                         440 |                 0.0178 |                        nan |                             0.044 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| imagenet  |                        1685.01 |                        714.167 |                        0 |                          2 |              3.16843e-14 |                   8.01175e-11 |              4.26838e-07 |                       6.90775 |                     516.883 |                     0.4083 |                0.8272 |           0.4189 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`