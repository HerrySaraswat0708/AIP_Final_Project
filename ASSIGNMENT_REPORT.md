# Comparative Analysis of TDA and FreeTTA for Test-Time Adaptation

## 1. Chosen Papers and Framing

For this assignment I compare:

1. **Baseline paper:** *Training-Free Test-Time Adaptation for Vision-Language Models (TDA)*.
2. **Later extension-style paper:** *Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM (FreeTTA)*.

Important clarification: FreeTTA is **not** a literal follow-up module added on top of TDA. The "extension" framing here is **my project-level interpretation** based on both methods solving the same problem with the same frozen CLIP-style backbone while moving from a local cache correction rule (TDA) to an explicit online distribution model (FreeTTA). That makes them a good baseline-versus-more-expressive comparison for this assignment.

## 2. Brief Description of Both Algorithms in My Own Words

### TDA

TDA keeps the vision-language backbone frozen and performs adaptation by building a **small online cache** of confident pseudo-labeled test features. It has:

- A **positive cache** that reinforces classes supported by similar stored features.
- A **negative cache** that suppresses classes when the model is uncertain.
- Entropy gates that decide which predictions are trusted enough to store.

So TDA is basically a **memory-based local correction mechanism**. It does not explicitly estimate the target data distribution. Instead, it reweights CLIP logits using similarities to a few cached test examples.

### FreeTTA

FreeTTA also keeps the backbone frozen, but instead of storing a few examples, it assumes each class can be represented by a **Gaussian-like distribution** in feature space. It initializes class means from text embeddings, then performs an **online EM-style update**:

- The **E-step** computes class responsibilities for the incoming sample.
- The **M-step** updates class priors, means, and covariance.
- The update is scaled by an entropy-based weight, so uncertain samples affect the model less.

So FreeTTA is a **global distribution-modeling method** rather than a local memory method.

## 3. Why the Later Method Should Give Better Results

FreeTTA should outperform TDA when the domain shift is **global and coherent**, because it uses the whole stream to estimate class statistics instead of relying on a few cached exemplars. That gives it three advantages:

1. **It uses more evidence.** TDA keeps only a few examples per class, while FreeTTA keeps running sufficient statistics for the entire stream.
2. **It models priors and class drift.** If the target distribution changes class frequencies or class centroids, FreeTTA can track that directly.
3. **It separates confidence weighting from prediction fusion.** The CLIP branch stays as a stable discriminative anchor, while the generative branch adds a correction term.

However, this superiority is not unconditional. FreeTTA can lose when:

1. A class is not well described by a single Gaussian-like cluster.
2. Early pseudo-label errors distort the running statistics.
3. The test stream order changes the online EM trajectory.

That already suggests the main hypothesis for the experiments:

- **TDA should be stronger when local neighbors matter more than global class means.**
- **FreeTTA should be stronger when long-range target distribution information matters.**

## 4. Experimental Setup

- Backbone setting: frozen CLIP-style image and text features already reproduced in this repository.
- Datasets used: `DTD`, `Caltech101`, `EuroSAT`, and `Oxford Pets`.
- Headline reproduced results use the existing repo outputs:
  - `outputs/tuning/best_tda_run_results.json`
  - `outputs/freetta_best_results.json`
- I also ran a **controlled natural-order comparison** with the new analysis pipeline:
  - `experiments/run_assignment_analysis.py`
  - consolidated outputs in `outputs/assignment_submission/`

All numbers below are top-1 accuracy.

## 5. Main Quantitative Results

### 5.1 Reproduced Results from the Existing Runs

| Dataset | CLIP | TDA | FreeTTA | FreeTTA - TDA |
|---|---:|---:|---:|---:|
| Caltech | 94.08 | 94.24 | 94.00 | -0.24 |
| DTD | 47.07 | 47.39 | 46.76 | -0.64 |
| EuroSAT | 59.54 | 58.04 | 61.56 | +3.52 |
| Pets | 89.94 | 88.63 | 90.16 | +1.53 |

Observation: in the reproduced setting, FreeTTA wins on **EuroSAT** and **Pets**, while TDA is slightly better on **Caltech** and **DTD**.

### 5.2 Controlled Natural-Order Comparison

| Dataset | CLIP | TDA | FreeTTA | FreeTTA - TDA |
|---|---:|---:|---:|---:|
| Caltech | 94.08 | 94.24 | 94.32 | +0.08 |
| DTD | 47.07 | 47.39 | 48.19 | +0.80 |
| EuroSAT | 59.54 | 58.04 | 62.19 | +4.15 |
| Pets | 89.94 | 88.63 | 89.02 | +0.38 |

Observation: in this controlled run, FreeTTA wins on **all four datasets**, but the size of the win is very uneven. The gain is tiny on Caltech and Pets, modest on DTD, and large on EuroSAT.

### 5.3 What the Difference Between the Two Tables Means

The reproduced FreeTTA runs and the controlled natural-order runs are not identical, and that difference is informative. Only the **FreeTTA stream order** changed here, which produced the following accuracy deltas:

| Dataset | Natural Order - Reproduced FreeTTA |
|---|---:|
| Caltech | +0.32 |
| DTD | +1.44 |
| EuroSAT | +0.63 |
| Pets | -1.14 |

This shows that FreeTTA is **order-sensitive**, which makes sense because online EM updates depend on the arrival sequence of samples.

## 6. When the Improvements Occur

### 6.1 Difficulty-Conditioned Analysis

I divided each dataset into `easy`, `medium`, and `hard` thirds using the shared zero-shot CLIP entropy as a method-agnostic difficulty score.

Key findings:

- **DTD:** FreeTTA is tied on easy samples, but better on medium (`+1.28`) and hard (`+1.12`) bins.
- **Caltech:** FreeTTA helps only on medium samples (`+0.85`), and is slightly worse on easy and hard bins.
- **EuroSAT:** the largest improvement occurs on hard samples (`+12.15`), which is the clearest evidence that distribution modeling helps when CLIP is unsure.
- **Pets:** FreeTTA is slightly worse on easy samples, but better on medium (`+0.49`) and hard (`+0.98`) samples.

**Conclusion:** FreeTTA does **not** win because it makes already-easy predictions easier. It mainly helps when the baseline is uncertain.

### 6.2 Stream-Phase Analysis

I also split the stream into `early`, `middle`, and `late` stages.

Key findings:

- **EuroSAT:** FreeTTA gains grow as more samples arrive, reaching `+12.79` points over TDA in the late phase.
- **Pets:** FreeTTA is worse early (`-2.40`), then turns positive in the middle and strongly positive late (`+2.94`).
- **DTD:** FreeTTA helps early and middle, then ties late.
- **Caltech:** gains are concentrated only in the middle; the early and late phases are already very easy.

**Conclusion:** FreeTTA tends to help more after it has accumulated enough evidence. That is exactly what we would expect from an online distribution estimator.

### 6.3 Disagreement Analysis

I checked only the samples where TDA and FreeTTA predict different classes.

| Dataset | Disagreement Rate | TDA Acc on Disagreements | FreeTTA Acc on Disagreements |
|---|---:|---:|---:|
| Caltech | 1.34% | 39.39 | 45.45 |
| DTD | 15.00% | 17.38 | 22.70 |
| EuroSAT | 11.73% | 16.74 | 52.11 |
| Pets | 7.60% | 39.07 | 44.09 |

This is a strong result: when the two methods actually disagree, **FreeTTA is usually the more reliable correction**, especially on EuroSAT.

## 7. Architecture, Losses, and Internal Workings

This section addresses the extra requirement beyond pure accuracy.

### 7.1 Shared Backbone

Both methods rely on the same frozen CLIP representation space, which was originally learned with a **contrastive image-text loss** during pretraining. That matters because both adaptation methods inherit the geometry created by that contrastive objective.

### 7.2 TDA Internal Mechanics

TDA has **no test-time backpropagation loss**. Its adaptation signal comes from:

- entropy-based sample selection,
- similarity-weighted cache retrieval,
- additive positive cache logits,
- subtractive negative cache logits,
- an optional fallback to the original CLIP prediction if the fused prediction looks less confident.

So TDA is best understood as a **non-parametric retrieval-and-reweighting scheme**, not as a learned optimizer.

### 7.3 FreeTTA Internal Mechanics

FreeTTA also avoids test-time backpropagation, but it still has a real probabilistic objective: it behaves like an **online EM update for a Gaussian mixture model**. That means:

- the generative branch is tied to **distribution likelihood** rather than entropy minimization,
- the CLIP branch provides the anchor prior,
- the entropy term is used as an **update weight**, not as a target to optimize directly.

This is important. Many TTA methods over-trust entropy minimization, which can create overconfident mistakes. FreeTTA instead uses confidence only to decide **how much** to update, while the actual correction comes from a generative model.

### 7.4 Internal Metric Evidence from the Runs

The internal summaries explain a lot of the behavior:

1. **EuroSAT is the clearest FreeTTA case.**
   - TDA final cache sizes are only `30` positive and `30` negative entries total.
   - EuroSAT has `8100` samples and only `10` classes, so TDA is effectively capped at about `3` positive and `3` negative exemplars per class.
   - FreeTTA, in contrast, updates class means, priors, and covariance using the whole stream.
   - This compression mismatch is a strong mechanistic reason for FreeTTA's `+4.15` controlled gain.

2. **Pets shows why stronger updates are not always enough.**
   - FreeTTA's average EM weight is very high (`0.697`) and the final mean drift is large (`1.225`).
   - That makes it adaptive, but it also explains the early-phase drop before late-phase recovery.
   - TDA is simpler and more conservative here, so the final gain is only modest.

3. **Caltech is already close to saturated.**
   - CLIP is already above `94%`.
   - TDA's negative gate opens only about `4.9%` of the time, and FreeTTA's gain appears only in the medium-difficulty region.
   - This is a low-headroom regime, so neither method can improve much.

4. **DTD is the borderline case.**
   - TDA is very active: the negative gate opens about `49.7%` of the time.
   - FreeTTA still improves medium/hard samples, but only slightly.
   - This is consistent with texture classes being harder to summarize with one global centroid-and-covariance model.

### 7.5 New Observation Not Explicitly Stated in Either Paper

My clearest new observation is:

> **The relative advantage of FreeTTA over TDA is controlled by the mismatch between how much target information is available in the stream and how much target information each method can actually retain.**

TDA can only preserve a very small, class-capped memory. FreeTTA compresses the whole stream into running distribution statistics. That is why FreeTTA benefits most on long streams with coherent class structure, while TDA remains competitive when local exemplars matter more or when the stream is too easy for global modeling to matter.

## 8. Additional New Analysis: Oracle Geometry Probe

I added a post-hoc diagnostic that is not used by either method at test time:

- an **oracle class-centroid classifier**, which is conceptually closer to FreeTTA's class-mean modeling,
- an **oracle leave-one-out 1-NN classifier**, which is conceptually closer to TDA's exemplar reasoning.

This probe is only for explanation, because it uses ground-truth labels after the fact.

| Dataset | Oracle Centroid | Oracle 1-NN |
|---|---:|---:|
| Caltech | 97.44 | 91.89 |
| DTD | 73.19 | 63.62 |
| EuroSAT | 78.21 | 89.84 |
| Pets | 91.47 | 83.46 |

Interpretation:

- The CLIP feature space already contains much more structure than either online method fully exploits.
- On **EuroSAT**, the 1-NN oracle is especially strong, but TDA still underperforms badly because its tiny capped cache cannot realize that potential.
- On **Caltech** and **Pets**, the centroid oracle is very strong, which supports FreeTTA's modeling assumption.
- On **DTD**, both online methods are far below the oracle probes, which suggests the main limitation is not raw feature separability but online pseudo-label estimation.

## 9. Final Answer to the Assignment Questions

### (a) Why should the later method be superior?

Because FreeTTA upgrades the adaptation mechanism from a small local cache to an online distribution model. That should be superior when the target shift is class-level and the stream is long enough to estimate stable means, priors, and covariance. My new analysis shows that the gain is concentrated in medium/hard regions and late phases, which is exactly where accumulated target statistics should matter most.

### (b) When do the improvements occur?

The improvements occur most consistently:

- on **harder samples**,
- in the **middle/late stream** after enough evidence has accumulated,
- on datasets where a tiny class-capped cache cannot retain enough information from a long stream.

They do **not** always occur on already-easy samples or in low-headroom datasets like Caltech, and they are sensitive to stream order.

### (c) Brief description, results, and reasoning

TDA is a retrieval-style cache method; FreeTTA is an online EM distribution-modeling method. The reproduced results show a mixed outcome, but the controlled natural-order analysis shows FreeTTA ahead on all four datasets, with the biggest win on EuroSAT. The internal metrics and difficulty/phase analyses explain why: FreeTTA helps mainly when uncertainty is high and enough target evidence has accumulated, while TDA is limited by its tiny retained memory and pseudo-label gating.

## 10. Files Produced for Submission

- Main report: `ASSIGNMENT_REPORT.md`
- Controlled summary tables: `outputs/assignment_submission/`
- New analysis pipeline: `experiments/run_assignment_analysis.py`

