# Comparative Analysis Outputs

## Accuracy Summary
| dataset   |   samples |   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |   tda_final_positive_cache_size |   tda_final_negative_cache_size |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |
|:----------|----------:|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|--------------------------------:|--------------------------------:|-------------------------:|------------------------------:|----------------------------:|
| dtd       |      1880 |   0.439362 |  0.458511 |      0.454787 |        0.0191489   |             0.0154255  |         -0.0037234  |                             221 |                              50 |                  1.15455 |                       1.25138 |                    0.33347  |
| pets      |      3669 |   0.883892 |  0.884165 |      0.889616 |        0.000272554 |             0.00572363 |          0.00545108 |                             111 |                              70 |                  1.122   |                       2.58808 |                    0.370464 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| dtd       |              0.731915 |          0.63617 |                  0.0957447 |
| pets      |              0.914963 |          0.83456 |                  0.0804034 |

## Prediction Change Analysis
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| dtd       | tda      |      1880 |                       775 |                     757 |                      87 |                   51 |                         210 |                 0.412234 |              0.40266   |     0.185106  |                    0.25     |                           0.0617433 |                     36 |           0.0191489   |                             1.57028 |                          1.72599 |
| dtd       | freetta  |      1880 |                       811 |                     917 |                      44 |                   15 |                          93 |                 0.431383 |              0.487766  |     0.0808511 |                    0.289474 |                           0.0181598 |                     29 |           0.0154255   |                             2.67556 |                          2.72562 |
| pets      | tda      |      3669 |                      3192 |                     354 |                      52 |                   51 |                          20 |                 0.869992 |              0.0964841 |     0.0335241 |                    0.422764 |                           0.0157262 |                      1 |           0.000272554 |                             1.08937 |                          1.06186 |
| pets      | freetta  |      3669 |                      3208 |                     348 |                      56 |                   35 |                          22 |                 0.874353 |              0.0948487 |     0.0307986 |                    0.495575 |                           0.0107925 |                     21 |           0.00572363  |                             1.0737  |                          1.12585 |

## Entropy / Confidence Summary
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| dtd       | clip     | all      |      1880 |       3.84998  |         3.84999  |   6.22754e-05 |         0.0223227 |           0.0222356 |      0.000402887 |
| dtd       | clip     | correct  |       826 |       3.84996  |         3.84997  |   6.38869e-05 |         0.0225377 |           0.0224843 |      0.000436958 |
| dtd       | clip     | wrong    |      1054 |       3.84999  |         3.85     |   5.72649e-05 |         0.0221541 |           0.0221057 |      0.000274312 |
| dtd       | tda      | all      |      1880 |       0.854972 |         0.569715 |   0.859935    |         0.756905  |           0.877797  |      0.262563    |
| dtd       | tda      | correct  |       862 |       0.506233 |         0.118719 |   0.704012    |         0.861383  |           0.98297   |      0.212361    |
| dtd       | tda      | wrong    |      1018 |       1.15027  |         1.05693  |   0.869363    |         0.668437  |           0.686475  |      0.268437    |
| dtd       | freetta  | all      |      1880 |       1.89652  |         2.11211  |   0.97399     |         0.49814   |           0.435998  |      0.283342    |
| dtd       | freetta  | correct  |       855 |       1.40651  |         1.31092  |   0.993966    |         0.644288  |           0.68457   |      0.281191    |
| dtd       | freetta  | wrong    |      1025 |       2.30526  |         2.52114  |   0.740615    |         0.37623   |           0.304712  |      0.220493    |
| pets      | clip     | all      |      3669 |       3.60982  |         3.6099   |   0.000411194 |         0.0304082 |           0.0303879 |      0.000700282 |
| pets      | clip     | correct  |      3243 |       3.60983  |         3.60991  |   0.000411695 |         0.0304729 |           0.0304406 |      0.000678452 |
| pets      | clip     | wrong    |       426 |       3.60979  |         3.60987  |   0.000406133 |         0.0299154 |           0.0298972 |      0.000666934 |
| pets      | tda      | all      |      3669 |       0.363868 |         0.170958 |   0.424431    |         0.878743  |           0.967931  |      0.170331    |
| pets      | tda      | correct  |      3244 |       0.288528 |         0.130774 |   0.356745    |         0.910948  |           0.977199  |      0.137803    |
| pets      | tda      | wrong    |       425 |       0.938935 |         0.905549 |   0.457933    |         0.632922  |           0.606162  |      0.192804    |
| pets      | freetta  | all      |      3669 |       0.379217 |         0.189102 |   0.421708    |         0.866074  |           0.961682  |      0.179425    |
| pets      | freetta  | correct  |      3264 |       0.301969 |         0.144179 |   0.354198    |         0.900347  |           0.973106  |      0.148016    |
| pets      | freetta  | wrong    |       405 |       1.00178  |         0.967645 |   0.405367    |         0.589852  |           0.571022  |      0.171216    |

## Adaptation Latency
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| dtd       |       50 |                      118 |                            9 |                           9 |              0.062766  |                 0.00478723 |                        0.00478723 |
| pets      |       50 |                      229 |                            5 |                           5 |              0.0624148 |                 0.00136277 |                        0.00136277 |

## Difficulty-Conditioned Comparison
| dataset   | difficulty_bin   |   samples |   clip_acc |   tda_acc |   freetta_acc |   freetta_minus_tda |
|:----------|:-----------------|----------:|-----------:|----------:|--------------:|--------------------:|
| dtd       | medium           |        30 |   0.666667 |  0.666667 |      0.666667 |          0          |
| dtd       | hard             |      1850 |   0.435676 |  0.455135 |      0.451351 |         -0.00378378 |
| pets      | easy             |      1850 |   0.877297 |  0.877838 |      0.885405 |          0.00756757 |
| pets      | medium           |      1819 |   0.890599 |  0.890599 |      0.893898 |          0.00329852 |

## Disagreement Comparison
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| dtd       |            0.17234  |                  0.212963 |                      0.191358 |                            3.85    |
| pets      |            0.032979 |                  0.355372 |                      0.520661 |                            3.60983 |

## Internal Mechanism Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| dtd       |                       178.089  |                        32.1995 |                        0 |                    5.71429 |                 0.253014 |                   0.00185544  |                  1.15455 |                       1.25138 |                    0.33347  |                  0.0957447 |              0.731915 |          0.63617 |
| pets      |                        76.0619 |                        49.6329 |                        0 |                   19.8324  |                 0.650374 |                   0.000989191 |                  1.122   |                       2.58808 |                    0.370464 |                  0.0804034 |              0.914963 |          0.83456 |

## Failure Bucket Summary
| dataset   | bucket                                 |   count |       rate |
|:----------|:---------------------------------------|--------:|-----------:|
| dtd       | clip_wrong_tda_wrong_freetta_correct   |      22 | 0.0117021  |
| dtd       | clip_wrong_tda_correct_freetta_wrong   |      65 | 0.0345745  |
| dtd       | clip_correct_tda_wrong_freetta_correct |      40 | 0.0212766  |
| dtd       | clip_correct_tda_correct_freetta_wrong |       4 | 0.00212766 |
| dtd       | all_wrong                              |     945 | 0.50266    |
| pets      | clip_wrong_tda_wrong_freetta_correct   |      32 | 0.00872172 |
| pets      | clip_wrong_tda_correct_freetta_wrong   |      28 | 0.00763151 |
| pets      | clip_correct_tda_wrong_freetta_correct |      31 | 0.00844917 |
| pets      | clip_correct_tda_correct_freetta_wrong |      15 | 0.00408831 |
| pets      | all_wrong                              |     342 | 0.0932134  |