# Architecture/Loss/Internal Analysis - dtd

## Method Mechanics Linked to Architecture and Objectives
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.026 (early) to ~0.458 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = -0.1199.
- Prediction churn = 0.5217, recovery latency = 2.3957.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.3167.
- Corr(mu_update_norm, correctness) = 0.2301.
- Mean prototype drift from init moved from ~0.0273 (early) to ~0.1469 (late).
- Prediction churn = 0.5435, recovery latency = 3.2646.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 43.40%.
- On disagreement samples: FreeTTA win rate = 7.11%, TDA win rate = 26.35%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.