# Dataset Report: pets

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.883892 |   0.88689 |      0.886345 |         0.00299809 |             0.00245298 |        -0.000545108 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| pets      |              0.914963 |          0.83456 |                  0.0804034 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| pets      | tda      |      3669 |                      3233 |                     393 |                      21 |                   10 |                          12 |                 0.881167 |               0.107114 |     0.0117198 |                    0.488372 |                          0.00308356 |                     11 |            0.00299809 |                             1.09188 |                          1.36329 |
| pets      | freetta  |      3669 |                      3229 |                     386 |                      23 |                   14 |                          17 |                 0.880076 |               0.105206 |     0.0147179 |                    0.425926 |                          0.00431699 |                      9 |            0.00245298 |                             1.13817 |                          1.25943 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| pets      | clip     | all      |      3669 |       3.60982  |         3.6099   |   0.000411193 |         0.0304082 |           0.0303879 |      0.000700282 |
| pets      | clip     | correct  |      3243 |       3.60983  |         3.60991  |   0.000411691 |         0.0304729 |           0.0304406 |      0.000678452 |
| pets      | clip     | wrong    |       426 |       3.60979  |         3.60987  |   0.000406148 |         0.0299154 |           0.0298972 |      0.000666934 |
| pets      | tda      | all      |      3669 |       0.445549 |         0.264855 |   0.453328    |         0.851182  |           0.943176  |      0.184435    |
| pets      | tda      | correct  |      3254 |       0.36266  |         0.211097 |   0.382618    |         0.88756   |           0.960421  |      0.151197    |
| pets      | tda      | wrong    |       415 |       1.09547  |         1.02867  |   0.438993    |         0.565939  |           0.550608  |      0.172474    |
| pets      | freetta  | all      |      3669 |       0.433582 |         0.259241 |   0.445526    |         0.85135   |           0.945433  |      0.185961    |
| pets      | freetta  | correct  |      3252 |       0.351707 |         0.197185 |   0.376492    |         0.887989  |           0.961934  |      0.152788    |
| pets      | freetta  | wrong    |       417 |       1.07208  |         1.01708  |   0.425529    |         0.565622  |           0.547355  |      0.173521    |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| pets      |           0.0106296 |                  0.435897 |                      0.384615 |                            3.60995 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| pets      |       50 |                        5 |                            5 |                         101 |             0.00136277 |                 0.00136277 |                         0.0275279 |

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