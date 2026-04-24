# Dataset Report: pets

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.883892 |   0.88798 |      0.884165 |         0.00408831 |            0.000272554 |         -0.00381575 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| pets      |              0.914963 |          0.83456 |                  0.0804034 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| pets      | tda      |      3669 |                      3231 |                     389 |                      27 |                   12 |                          10 |                 0.880621 |               0.106023 |    0.0133551  |                    0.55102  |                          0.00370028 |                     15 |           0.00408831  |                             1.10381 |                          1.33862 |
| pets      | freetta  |      3669 |                      3237 |                     415 |                       7 |                    6 |                           4 |                 0.882257 |               0.11311  |    0.00463342 |                    0.411765 |                          0.00185014 |                      1 |           0.000272554 |                             3.60984 |                          3.60977 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| pets      | clip     | all      |      3669 |       3.60982  |         3.6099   |   0.000411194 |         0.0304082 |           0.0303879 |      0.000700282 |
| pets      | clip     | correct  |      3243 |       3.60983  |         3.60991  |   0.000411695 |         0.0304729 |           0.0304406 |      0.000678452 |
| pets      | clip     | wrong    |       426 |       3.60979  |         3.60987  |   0.000406133 |         0.0299154 |           0.0298972 |      0.000666934 |
| pets      | tda      | all      |      3669 |       0.439995 |         0.26377  |   0.451827    |         0.852671  |           0.944454  |      0.183531    |
| pets      | tda      | correct  |      3258 |       0.358065 |         0.203599 |   0.381105    |         0.888342  |           0.960912  |      0.151145    |
| pets      | tda      | wrong    |       411 |       1.08946  |         1.02298  |   0.44282     |         0.5699    |           0.557369  |      0.171922    |
| pets      | freetta  | all      |      3669 |       3.6098   |         3.60988  |   0.000415399 |         0.0304464 |           0.0304244 |      0.000704037 |
| pets      | freetta  | correct  |      3244 |       3.6098   |         3.60988  |   0.000415652 |         0.0305118 |           0.0304821 |      0.000681074 |
| pets      | freetta  | wrong    |       425 |       3.60977  |         3.60985  |   0.00041224  |         0.0299471 |           0.0299279 |      0.000675675 |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| pets      |           0.0144454 |                  0.509434 |                      0.245283 |                            3.60989 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| pets      |       50 |                        5 |                          413 |                          87 |             0.00136277 |                   0.112565 |                         0.0237122 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| pets      |                        76.0619 |                        46.1717 |                        0 |                    19.8324 |               0.00073206 |                   0.000290067 |                 0.795472 |                       3.61066 |                     525.616 |                  0.0804034 |              0.914963 |          0.83456 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`