# Dataset Report: dtd

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.439362 |  0.451596 |      0.465426 |           0.012234 |              0.0260638 |           0.0138298 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| dtd       |              0.731915 |          0.63617 |                  0.0957447 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| dtd       | tda      |      1880 |                       792 |                     865 |                      57 |                   34 |                         132 |                 0.421277 |               0.460106 |      0.118617 |                    0.255605 |                           0.0411622 |                     23 |             0.012234  |                             2.41213 |                          2.48342 |
| dtd       | freetta  |      1880 |                       783 |                     791 |                      92 |                   43 |                         171 |                 0.416489 |               0.420745 |      0.162766 |                    0.300654 |                           0.0520581 |                     49 |             0.0260638 |                             2.28327 |                          2.18668 |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| dtd       | clip     | all      |      1880 |        3.84998 |         3.84999  |   6.22791e-05 |         0.0223227 |           0.0222356 |      0.000402886 |
| dtd       | clip     | correct  |       826 |        3.84996 |         3.84997  |   6.38978e-05 |         0.0225377 |           0.0224843 |      0.000436958 |
| dtd       | clip     | wrong    |      1054 |        3.84999 |         3.85     |   5.72608e-05 |         0.0221541 |           0.0221057 |      0.000274312 |
| dtd       | tda      | all      |      1880 |        1.55481 |         1.61805  |   0.984717    |         0.592474  |           0.582342  |      0.287186    |
| dtd       | tda      | correct  |       849 |        1.08573 |         0.861222 |   0.926507    |         0.730729  |           0.823773  |      0.259439    |
| dtd       | tda      | wrong    |      1031 |        1.94109 |         2.09625  |   0.854913    |         0.478624  |           0.419171  |      0.257418    |
| dtd       | freetta  | all      |      1880 |        1.53753 |         1.63477  |   0.969575    |         0.578782  |           0.545698  |      0.287817    |
| dtd       | freetta  | correct  |       875 |        1.0814  |         0.872197 |   0.943328    |         0.71575   |           0.810036  |      0.273009    |
| dtd       | freetta  | wrong    |      1005 |        1.93465 |         2.13437  |   0.803084    |         0.459531  |           0.399863  |      0.243957    |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| dtd       |            0.142021 |                  0.164794 |                      0.262172 |                               3.85 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| dtd       |       50 |                      358 |                           30 |                          30 |               0.190426 |                  0.0159574 |                         0.0159574 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| dtd       |                          112.9 |                        38.7059 |                        0 |                          8 |                 0.464729 |                    0.00206804 |                  1.16053 |                       2.08321 |                    0.326536 |                  0.0957447 |              0.731915 |          0.63617 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`