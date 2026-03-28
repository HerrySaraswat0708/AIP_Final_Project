# Architecture/Loss/Internal Analysis - dtd

## Method Mechanics Linked to Architecture and Objectives
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.002 (early) to ~0.009 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = 0.3240.
- Prediction churn = 0.0000, recovery latency = 1.4444.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.1857.
- Corr(mu_update_norm, correctness) = 0.1857.
- Mean prototype drift from init moved from ~0.0038 (early) to ~0.0113 (late).
- Prediction churn = 0.0000, recovery latency = 1.4444.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 0.00%.
- On disagreement samples: FreeTTA win rate = 0.00%, TDA win rate = 0.00%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.