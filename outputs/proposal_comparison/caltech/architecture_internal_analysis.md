# Architecture/Loss/Internal Analysis - caltech

## Method Mechanics Linked to Architecture and Objectives
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.309 (early) to ~1.000 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = -0.3874.
- Prediction churn = 0.3450, recovery latency = 1.2798.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.4520.
- Corr(mu_update_norm, correctness) = 0.2073.
- Mean prototype drift from init moved from ~0.3357 (early) to ~0.8176 (late).
- Prediction churn = 0.2326, recovery latency = 1.5173.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 27.15%.
- On disagreement samples: FreeTTA win rate = 3.65%, TDA win rate = 53.01%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.