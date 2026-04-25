# Dataset Report: caltech

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.935497 |  0.935903 |      0.936308 |         0.00040568 |            0.000811359 |          0.00040568 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| caltech   |              0.974442 |         0.918458 |                  0.0559838 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| caltech   | tda      |      2465 |                      2297 |                     148 |                      10 |                    9 |                           1 |                 0.931846 |              0.0600406 |    0.00811359 |                         0.5 |                          0.00390286 |                      1 |           0.00040568  |                            0.929316 |                         0.832297 |
| caltech   | freetta  |      2465 |                      2302 |                     153 |                       6 |                    4 |                           0 |                 0.933874 |              0.062069  |    0.0040568  |                         0.6 |                          0.00173461 |                      2 |           0.000811359 |                            1.06011  |                         1.56688  |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| caltech   | clip     | all      |      2465 |       4.6048   |        4.60482   |   0.000124142 |         0.0112927 |           0.0113273 |      0.000308858 |
| caltech   | clip     | correct  |      2306 |       4.60481  |        4.60483   |   0.000122988 |         0.0113025 |           0.0113355 |      0.000304652 |
| caltech   | clip     | wrong    |       159 |       4.60477  |        4.60478   |   0.000135888 |         0.0111506 |           0.0112177 |      0.000333535 |
| caltech   | tda      | all      |      2465 |       0.167737 |        0.0170918 |   0.357252    |         0.954786  |           0.9982    |      0.105733    |
| caltech   | tda      | correct  |      2307 |       0.129636 |        0.0135858 |   0.313953    |         0.968979  |           0.998516  |      0.0850509   |
| caltech   | tda      | wrong    |       158 |       0.724062 |        0.661197  |   0.470411    |         0.747564  |           0.753977  |      0.151367    |
| caltech   | freetta  | all      |      2465 |       0.342616 |        0.10086   |   0.542614    |         0.907329  |           0.98564   |      0.156616    |
| caltech   | freetta  | correct  |      2308 |       0.293085 |        0.0814832 |   0.493139    |         0.927092  |           0.988738  |      0.133602    |
| caltech   | freetta  | wrong    |       157 |       1.07076  |        0.75031   |   0.693898    |         0.616802  |           0.626798  |      0.180464    |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| caltech   |          0.00811359 |                      0.45 |                           0.5 |                            4.60471 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip |   freetta_break_even_vs_clip |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|-----------------------------:|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| caltech   |       50 |                     1374 |                         1050 |                         119 |               0.557404 |                   0.425963 |                         0.0482759 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| caltech   |                        136.109 |                        42.6166 |                        0 |                       4.93 |                 0.825633 |                    0.00168554 |                  1.13427 |                       3.67739 |                    0.356591 |                  0.0559838 |              0.974442 |         0.918458 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`