# Dataset Report: caltech

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.935497 |  0.933063 |      0.935497 |        -0.00243408 |                      0 |          0.00243408 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| caltech   |              0.974442 |         0.918458 |                  0.0559838 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| caltech   | tda      |      2465 |                      2289 |                     147 |                      11 |                   17 |                           1 |                 0.9286   |              0.0596349 |     0.0117647 |                     0.37931 |                          0.00737207 |                     -6 |           -0.00243408 |                            0.799891 |                         0.795635 |
| caltech   | freetta  |      2465 |                      2306 |                     159 |                       0 |                    0 |                           0 |                 0.935497 |              0.064503  |     0         |                     0       |                          0          |                      0 |            0          |                          nan        |                       nan        |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| caltech   | clip     | all      |      2465 |      4.6048    |      4.60482     |   0.000124148 |         0.0112927 |           0.0113273 |      0.000308858 |
| caltech   | clip     | correct  |      2306 |      4.60481   |      4.60483     |   0.000122992 |         0.0113025 |           0.0113355 |      0.000304652 |
| caltech   | clip     | wrong    |       159 |      4.60477   |      4.60478     |   0.000135919 |         0.0111506 |           0.0112177 |      0.000333535 |
| caltech   | tda      | all      |      2465 |      0.100751  |      0.00130176  |   0.282399    |         0.971252  |           0.999891  |      0.0889502   |
| caltech   | tda      | correct  |      2300 |      0.0775181 |      0.000992375 |   0.256999    |         0.980341  |           0.999919  |      0.0726755   |
| caltech   | tda      | wrong    |       165 |      0.424606  |      0.323348    |   0.397895    |         0.844557  |           0.926455  |      0.165456    |
| caltech   | freetta  | all      |      2465 |      4.60464   |      4.60467     |   0.00018016  |         0.0115703 |           0.0116117 |      0.000379049 |
| caltech   | freetta  | correct  |      2306 |      4.60464   |      4.60467     |   0.000178461 |         0.0115823 |           0.0116216 |      0.000373969 |
| caltech   | freetta  | wrong    |       159 |      4.60459   |      4.6046      |   0.000197475 |         0.0113959 |           0.0114772 |      0.000408254 |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| caltech   |           0.0117647 |                   0.37931 |                      0.586207 |                            4.60481 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip | freetta_break_even_vs_clip   |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|:-----------------------------|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| caltech   |       50 |                     1374 |                              |                          41 |               0.557404 |                        nan |                         0.0166329 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| caltech   |                        136.109 |                        34.9716 |                        0 |                       4.93 |              1.00165e-09 |                   1.06972e-09 |              2.57599e-07 |                       4.60517 |                     513.204 |                  0.0559838 |              0.974442 |         0.918458 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`