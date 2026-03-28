# Architecture/Loss/Internal Analysis - eurosat

## Method Mechanics Linked to Architecture and Objectives
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.172 (early) to ~1.000 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = -0.2244.
- Prediction churn = 0.5009, recovery latency = 2.4727.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.2079.
- Corr(mu_update_norm, correctness) = 0.0804.
- Mean prototype drift from init moved from ~0.1475 (early) to ~0.2250 (late).
- Prediction churn = 0.0877, recovery latency = 5.9341.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 69.06%.
- On disagreement samples: FreeTTA win rate = 0.52%, TDA win rate = 34.69%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.