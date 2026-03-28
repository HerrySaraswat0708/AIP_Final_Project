# Architecture/Loss/Internal Analysis - caltech

## Method Mechanics Linked to Architecture and Objectives
- CLIP baseline uses contrastive pretraining and cosine-similarity logits (no test-time updates).
- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.
- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.

## Empirical Link: TDA Internal Signals
- Cache fill ratio rose from ~0.001 (early) to ~0.020 (late), showing growing memory reliance.
- Corr(jsd_clip_fused, correctness) = 0.1846.
- Prediction churn = 0.0000, recovery latency = 1.2500.

## Empirical Link: FreeTTA Internal Signals
- Mean EM confidence-weight vs correctness corr = 0.3112.
- Corr(mu_update_norm, correctness) = 0.3175.
- Mean prototype drift from init moved from ~0.0035 (early) to ~0.0463 (late).
- Prediction churn = 0.0000, recovery latency = 1.2500.

## TDA vs FreeTTA Decision Regions
- Disagreement rate = 0.00%.
- On disagreement samples: FreeTTA win rate = 0.00%, TDA win rate = 0.00%.
- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.