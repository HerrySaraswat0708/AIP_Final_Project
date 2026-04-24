# Comparative Analysis Outputs

## Accuracy Summary
| dataset   |   samples |   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |   tda_final_positive_cache_size |   tda_final_negative_cache_size |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |
|:----------|----------:|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|--------------------------------:|--------------------------------:|-------------------------:|------------------------------:|----------------------------:|
| caltech   |      2465 |   0.935497 |  0.935903 |      0.935903 |         0.00040568 |             0.00040568 |                   0 |                             300 |                              62 |                  1.22459 |                       4.60503 |                     517.429 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| caltech   |              0.974442 |         0.918458 |                  0.0559838 |

## Prediction Change Analysis
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| caltech   | tda      |      2465 |                      2297 |                     148 |                      10 |                    9 |                           1 |                 0.931846 |              0.0600406 |    0.00811359 |                    0.5      |                         0.00390286  |                      1 |            0.00040568 |                            0.929317 |                         0.832297 |
| caltech   | freetta  |      2465 |                      2305 |                     157 |                       2 |                    1 |                           0 |                 0.935091 |              0.0636917 |    0.00121704 |                    0.666667 |                         0.000433651 |                      1 |            0.00040568 |                            4.60495  |                         4.60481  |

## Entropy / Confidence Summary
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| caltech   | clip     | all      |      2465 |       4.6048   |        4.60482   |   0.000124148 |         0.0112927 |           0.0113273 |      0.000308858 |
| caltech   | clip     | correct  |      2306 |       4.60481  |        4.60483   |   0.000122992 |         0.0113025 |           0.0113355 |      0.000304652 |
| caltech   | clip     | wrong    |       159 |       4.60477  |        4.60478   |   0.000135919 |         0.0111506 |           0.0112177 |      0.000333535 |
| caltech   | tda      | all      |      2465 |       0.167737 |        0.0170917 |   0.357252    |         0.954786  |           0.9982    |      0.105733    |
| caltech   | tda      | correct  |      2307 |       0.129636 |        0.0135858 |   0.313953    |         0.968979  |           0.998515  |      0.085051    |
| caltech   | tda      | wrong    |       158 |       0.724061 |        0.661197  |   0.470411    |         0.747564  |           0.753977  |      0.151367    |
| caltech   | freetta  | all      |      2465 |       4.6048   |        4.60482   |   0.00012418  |         0.0112934 |           0.0113286 |      0.000307361 |
| caltech   | freetta  | correct  |      2307 |       4.6048   |        4.60483   |   0.000123067 |         0.0113027 |           0.0113358 |      0.000303757 |
| caltech   | freetta  | wrong    |       158 |       4.60477  |        4.60477   |   0.000135123 |         0.0111581 |           0.0112254 |      0.000327234 |

## Adaptation Latency
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| caltech   |       50 |                     1374 |                           13 |                          13 |               0.557404 |                 0.00527383 |                        0.00527383 |

## Difficulty-Conditioned Comparison
| dataset   | difficulty_bin   |   samples |   clip_acc |   tda_acc |   freetta_acc |   freetta_minus_tda |
|:----------|:-----------------|----------:|-----------:|----------:|--------------:|--------------------:|
| caltech   | easy             |       822 |   0.907543 |  0.906326 |      0.907543 |          0.00121655 |
| caltech   | medium           |       821 |   0.937881 |  0.937881 |      0.936663 |         -0.00121803 |
| caltech   | hard             |       822 |   0.961071 |  0.963504 |      0.963504 |          0          |

## Disagreement Comparison
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| caltech   |          0.00851927 |                   0.47619 |                       0.47619 |                            4.60477 |

## Internal Mechanism Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| caltech   |                        136.109 |                        34.9716 |                        0 |                       4.93 |                  0.63098 |                    0.00202908 |                  1.22459 |                       4.60503 |                     517.429 |                  0.0559838 |              0.974442 |         0.918458 |

## Failure Bucket Summary
| dataset   | bucket                                 |   count |       rate |
|:----------|:---------------------------------------|--------:|-----------:|
| caltech   | clip_wrong_tda_wrong_freetta_correct   |       1 | 0.00040568 |
| caltech   | clip_wrong_tda_correct_freetta_wrong   |       9 | 0.00365112 |
| caltech   | clip_correct_tda_wrong_freetta_correct |       9 | 0.00365112 |
| caltech   | clip_correct_tda_correct_freetta_wrong |       1 | 0.00040568 |
| caltech   | all_wrong                              |     148 | 0.0600406  |