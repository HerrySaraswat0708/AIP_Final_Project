# Architecture/Loss/Internal Analysis - dtd

## Method Mechanics Linked to Architecture and Objectives
- CLIP baseline uses contrastive pretraining and cosine-similarity logits (no test-time updates).
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.001 (early) to ~0.010 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = 0.2445.
- Prediction churn = 0.0000, recovery latency = 2.0000.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.2683.
- Corr(mu_update_norm, correctness) = 0.2665.
- Mean prototype drift from init moved from ~0.0002 (early) to ~0.0121 (late).
- Prediction churn = 0.0000, recovery latency = 1.3333.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 80.00%.
- On disagreement samples: FreeTTA win rate = 62.50%, TDA win rate = 0.00%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.