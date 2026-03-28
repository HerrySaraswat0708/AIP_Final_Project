# Architecture/Loss/Internal Analysis - pets

## Method Mechanics Linked to Architecture and Objectives
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.126 (early) to ~1.000 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = 0.2268.
- Prediction churn = 0.1633, recovery latency = 1.2988.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.5082.
- Corr(mu_update_norm, correctness) = 0.2743.
- Mean prototype drift from init moved from ~0.2304 (early) to ~0.5110 (late).
- Prediction churn = 0.2041, recovery latency = 2.4360.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 50.04%.
- On disagreement samples: FreeTTA win rate = 1.31%, TDA win rate = 76.09%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.