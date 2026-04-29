# Dataset Report: dtd

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.439362 |  0.458511 |      0.454787 |          0.0191489 |              0.0154255 |          -0.0037234 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| dtd       |              0.731915 |          0.63617 |                  0.0957447 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| dtd       | tda      |      1880 |                       775 |                     757 |                      87 |                   51 |                         210 |                 0.412234 |               0.40266  |     0.185106  |                    0.25     |                           0.0617433 |                     36 |             0.0191489 |                             1.57028 |                          1.72599 |
| dtd       | freetta  |      1880 |                       811 |                     917 |                      44 |                   15 |                          93 |                 0.431383 |               0.487766 |     0.0808511 |                    0.289474 |                           0.0181598 |                     29 |             0.0154255 |                             2.67556 |                          2.72562 |

## Entropy / Confidence Metrics
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

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| dtd       |             0.17234 |                  0.212963 |                      0.191358 |                               3.85 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| dtd       |       50 |                      118 |                            9 |                           9 |               0.062766 |                 0.00478723 |                        0.00478723 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| dtd       |                        178.089 |                        32.1995 |                        0 |                    5.71429 |                 0.253014 |                    0.00185544 |                  1.15455 |                       1.25138 |                     0.33347 |                  0.0957447 |              0.731915 |          0.63617 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`