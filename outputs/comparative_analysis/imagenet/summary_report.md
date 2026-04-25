# Dataset Report: imagenet

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|     0.6235 |    0.6272 |        0.6272 |             0.0037 |                 0.0037 |                   0 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| imagenet  |                0.8272 |           0.4189 |                     0.4083 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| imagenet  | tda      |     10000 |                      6157 |                    3455 |                     115 |                   78 |                         195 |                   0.6157 |                 0.3455 |        0.0388 |                    0.296392 |                           0.01251   |                     37 |                0.0037 |                             2.07507 |                          2.23493 |
| imagenet  | freetta  |     10000 |                      6156 |                    3482 |                     116 |                   79 |                         167 |                   0.6156 |                 0.3482 |        0.0362 |                    0.320442 |                           0.0126704 |                     37 |                0.0037 |                             2.07391 |                          2.02242 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| imagenet  | clip     | all      |     10000 |       6.90732  |         6.90734  |   0.000144218 |        0.00114748 |          0.00114709 |      3.37564e-05 |
| imagenet  | clip     | correct  |      6235 |       6.90731  |         6.90733  |   0.000147393 |        0.00115696 |          0.00115677 |      3.18571e-05 |
| imagenet  | clip     | wrong    |      3765 |       6.90734  |         6.90735  |   0.000137582 |        0.00113178 |          0.00113049 |      3.08273e-05 |
| imagenet  | tda      | all      |     10000 |       1.12784  |         0.775713 |   1.09907     |        0.732179   |          0.809609   |      0.256633    |
| imagenet  | tda      | correct  |      6272 |       0.733193 |         0.398789 |   0.856363    |        0.830209   |          0.926315   |      0.204836    |
| imagenet  | tda      | wrong    |      3728 |       1.7918   |         1.59305  |   1.14173     |        0.567253   |          0.574268   |      0.250413    |
| imagenet  | freetta  | all      |     10000 |       1.22731  |         0.987004 |   1.03506     |        0.674089   |          0.711021   |      0.263861    |
| imagenet  | freetta  | correct  |      6272 |       0.850453 |         0.618419 |   0.826171    |        0.778124   |          0.863755   |      0.222357    |
| imagenet  | freetta  | wrong    |      3728 |       1.86133  |         1.69189  |   1.04141     |        0.499061   |          0.478729   |      0.233943    |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| imagenet  |               0.056 |                  0.251786 |                      0.251786 |                            6.90733 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| imagenet  |       50 |                      363 |                            2 |                           2 |                 0.0363 |                     0.0002 |                            0.0002 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| imagenet  |                        1685.01 |                        912.301 |                        0 |                          2 |                 0.532558 |                   0.000473977 |                  1.12765 |                       3.96526 |                    0.364114 |                     0.4083 |                0.8272 |           0.4189 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`