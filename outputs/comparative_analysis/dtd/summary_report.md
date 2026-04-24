# Dataset Report: dtd

## Accuracy Table
|   clip_acc |   tda_acc |   freetta_acc |   tda_gain_vs_clip |   freetta_gain_vs_clip |   freetta_minus_tda |
|-----------:|----------:|--------------:|-------------------:|-----------------------:|--------------------:|
|   0.439362 |  0.451596 |      0.439362 |           0.012234 |                      0 |           -0.012234 |

## Geometry Probe
| dataset   |   oracle_centroid_acc |   oracle_1nn_acc |   geometry_alignment_score |
|:----------|----------------------:|-----------------:|---------------------------:|
| dtd       |              0.731915 |          0.63617 |                  0.0957447 |

## Prediction Change Metrics
| dataset   | method   |   samples |   unchanged_correct_count |   unchanged_wrong_count |   beneficial_flip_count |   harmful_flip_count |   other_changed_wrong_count |   unchanged_correct_rate |   unchanged_wrong_rate |   change_rate |   beneficial_flip_precision |   harmful_flip_rate_on_clip_correct |   net_correction_score |   net_correction_rate |   avg_entropy_after_beneficial_flip |   avg_entropy_after_harmful_flip |
|:----------|:---------|----------:|--------------------------:|------------------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|--------------:|----------------------------:|------------------------------------:|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|
| dtd       | tda      |      1880 |                       795 |                     852 |                      54 |                   31 |                         148 |                 0.422872 |               0.453191 |      0.123936 |                     0.23176 |                           0.0375303 |                     23 |              0.012234 |                             2.44656 |                          2.49788 |
| dtd       | freetta  |      1880 |                       826 |                    1054 |                       0 |                    0 |                           0 |                 0.439362 |               0.560638 |      0        |                     0       |                           0         |                      0 |              0        |                           nan       |                        nan       |

## Entropy / Confidence Metrics
| dataset   | method   | subset   |   samples |   mean_entropy |   median_entropy |   std_entropy |   mean_confidence |   median_confidence |   std_confidence |
|:----------|:---------|:---------|----------:|---------------:|-----------------:|--------------:|------------------:|--------------------:|-----------------:|
| dtd       | clip     | all      |      1880 |        3.84998 |         3.84999  |   6.22754e-05 |         0.0223227 |           0.0222356 |      0.000402887 |
| dtd       | clip     | correct  |       826 |        3.84996 |         3.84997  |   6.38869e-05 |         0.0225377 |           0.0224843 |      0.000436958 |
| dtd       | clip     | wrong    |      1054 |        3.84999 |         3.85     |   5.72649e-05 |         0.0221541 |           0.0221057 |      0.000274312 |
| dtd       | tda      | all      |      1880 |        1.54908 |         1.60278  |   1.00098     |         0.596867  |           0.595407  |      0.289171    |
| dtd       | tda      | correct  |       849 |        1.07366 |         0.846504 |   0.939473    |         0.735287  |           0.836626  |      0.261013    |
| dtd       | tda      | wrong    |      1031 |        1.94057 |         2.09184  |   0.872267    |         0.482882  |           0.428703  |      0.260013    |
| dtd       | freetta  | all      |      1880 |        3.8499  |         3.84992  |   8.9762e-05  |         0.0225377 |           0.0224316 |      0.000488686 |
| dtd       | freetta  | correct  |       826 |        3.84988 |         3.84989  |   9.21436e-05 |         0.0227985 |           0.0227329 |      0.000530614 |
| dtd       | freetta  | wrong    |      1054 |        3.84992 |         3.84994  |   8.24211e-05 |         0.0223333 |           0.0222745 |      0.00033204  |

## Disagreement Metrics
| dataset   |   disagreement_rate |   tda_acc_on_disagreement |   freetta_acc_on_disagreement |   avg_clip_entropy_on_disagreement |
|:----------|--------------------:|--------------------------:|------------------------------:|-----------------------------------:|
| dtd       |            0.123936 |                   0.23176 |                      0.133047 |                               3.85 |

## Latency Metrics
| dataset   |   window |   tda_break_even_vs_clip | freetta_break_even_vs_clip   |   freetta_break_even_vs_tda |   tda_break_even_ratio |   freetta_break_even_ratio |   freetta_vs_tda_break_even_ratio |
|:----------|---------:|-------------------------:|:-----------------------------|----------------------------:|-----------------------:|---------------------------:|----------------------------------:|
| dtd       |       50 |                      287 |                              |                         228 |                0.15266 |                        nan |                          0.121277 |

## Internal Metrics
| dataset   |   tda_mean_positive_cache_size |   tda_mean_negative_cache_size |   tda_negative_gate_rate |   tda_cache_pressure_ratio |   freetta_mean_em_weight |   freetta_mean_mu_update_norm |   freetta_final_mu_drift |   freetta_final_prior_entropy |   freetta_final_sigma_trace |   geometry_alignment_score |   oracle_centroid_acc |   oracle_1nn_acc |
|:----------|-------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------------:|-----------------:|
| dtd       |                          112.9 |                        62.0697 |                        0 |                          8 |              2.99152e-08 |                   3.79029e-08 |               3.7633e-05 |                       3.85015 |                     512.918 |                  0.0957447 |              0.731915 |          0.63617 |

## Generated PNG Outputs
- `prediction_change_analysis.png`
- `entropy_confidence_analysis.png`
- `trajectory_analysis.png`
- `freetta_internal_analysis.png`
- `tda_internal_analysis.png`
- `pca_logit_visualization.png`