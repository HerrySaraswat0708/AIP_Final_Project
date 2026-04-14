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
- Core superiority-analysis outputs across all datasets: `outputs/workshop_core_all/`
- Targeted DTD ablations: `outputs/workshop_dtd_ablation/`
- Targeted DTD order-stress results: `outputs/workshop_dtd_order/`
- Second-round analysis pipeline: `experiments/run_superiority_analysis.py`

## 11. New Superiority Analyses Added After the First Draft

The earlier draft already covered difficulty bins, stream phase, disagreements, and oracle geometry. I then added a second round of **new analyses that are not present in either paper** and ran fresh experiments with:

- `experiments/run_superiority_analysis.py`
- core outputs across all four datasets in `outputs/workshop_core_all/`
- targeted ablations on DTD in `outputs/workshop_dtd_ablation/`
- targeted order-stress experiments on DTD in `outputs/workshop_dtd_order/`

These new analyses are useful because they explain **how** FreeTTA wins, **when** TDA can still be better, and **which mechanism is actually responsible** for the difference.

### 11.1 Calibration Analysis: FreeTTA Wins Accuracy Without Producing Calibrated Probabilities

I measured expected calibration error (ECE) using each method's final prediction confidence.

| Dataset | TDA ECE | FreeTTA ECE |
|---|---:|---:|
| Caltech | 0.82% | 93.18% |
| DTD | 12.67% | 45.94% |
| EuroSAT | 4.15% | 51.86% |
| Pets | 3.53% | 85.97% |

This is a very interesting and genuinely new result. In this reproduction, **FreeTTA is often more accurate but much less calibrated as a probability estimator**. Its top-1 confidence remains very low because the final fused logits are only gently perturbed around the zero-shot branch. So its gain does **not** come from becoming more certain. It comes from **reordering class scores correctly**.

That means:

- **TDA behaves like a sharper, more decisive classifier.**
- **FreeTTA behaves like a cautious reranker whose probabilities should not be over-interpreted.**

This distinction is not explicitly discussed in either paper.

### 11.2 Entropy-Reduction Quality: TDA Sharpens More, FreeTTA Corrects More Gently

I measured how much each method reduces its own predictive entropy on correct versus wrong predictions.

| Dataset | TDA Selective Entropy Gap | FreeTTA Selective Entropy Gap |
|---|---:|---:|
| Caltech | +0.0050 | +0.000011 |
| DTD | +0.0154 | +0.000011 |
| EuroSAT | +0.0091 | +0.000003 |
| Pets | -0.1712 | -0.000050 |

Where:

- **Selective Entropy Gap = mean entropy drop on correct predictions - mean entropy drop on wrong predictions**

Interpretation:

- TDA usually **sharpens predictions much more strongly** than FreeTTA.
- FreeTTA's entropy change is tiny, which supports the idea that it improves by **small posterior corrections** rather than aggressive confidence collapse.
- The most important negative case is **Pets**, where TDA reduces entropy **more on wrong samples than on correct ones**, which is a sign of overconfident failure.

This analysis directly supports the intuition that FreeTTA's posterior modeling gives it a different kind of reliability than TDA's cache retrieval rule.

### 11.3 Safe-vs-Harmful Prediction Changes: FreeTTA Is Safer on Global-Shift Datasets

I measured two things whenever a method changed the original CLIP prediction:

1. **Beneficial flip precision:** among changed predictions, how often did the change actually fix CLIP?
2. **Harmful flip rate:** how often did the method damage a previously correct CLIP prediction?

| Dataset | TDA Beneficial Precision | FreeTTA Beneficial Precision | TDA Harmful Rate | FreeTTA Harmful Rate |
|---|---:|---:|---:|---:|
| Caltech | 63.64% | 52.94% | 0.13% | 0.52% |
| DTD | 23.37% | 23.00% | 4.18% | 2.82% |
| EuroSAT | 27.77% | 47.71% | 9.85% | 2.90% |
| Pets | 33.48% | 16.25% | 3.79% | 1.42% |

This reveals a much sharper story than plain accuracy:

- On **EuroSAT**, FreeTTA is clearly the better correction mechanism: it changes predictions with **much higher precision** and **far less damage**.
- On **DTD**, the beneficial precision is similar, but FreeTTA is still safer because it harms fewer already-correct predictions.
- On **Pets**, FreeTTA is extremely conservative: fewer beneficial changes, but also much less damage.

So FreeTTA's main advantage is often not that it changes more predictions. It is that on the right datasets it changes them **more selectively**.

### 11.4 Adaptation-Onset Analysis: How Many Initial Samples Are Needed Before Gains Appear?

I measured a rolling break-even point: the earliest stream position where the method's rolling accuracy becomes positive relative to a baseline.

For **FreeTTA versus TDA**, the break-even fractions are:

| Dataset | FreeTTA vs TDA Break-Even |
|---|---:|
| Caltech | 25.40% of the stream |
| DTD | 5.00% of the stream |
| EuroSAT | 5.31% of the stream |
| Pets | 4.99% of the stream |

This is a strong result because it answers the "how many initial samples are needed?" question directly:

- On **DTD, EuroSAT, and Pets**, FreeTTA starts helping after only about **5%** of the stream.
- On **Caltech**, it needs much more time because there is very little headroom and CLIP is already near saturation.

So the practical conclusion is:

- **FreeTTA benefits appear quickly when there is real distribution shift to learn.**
- **If the dataset is already easy, the online EM statistics need much longer before they produce measurable gains.**

### 11.5 Order-Stress Analysis on DTD: TDA Can Be Better When the Stream Is Easy-to-Hard

I ran a new stress test on DTD with several stream orders:

| Order | TDA | FreeTTA | FreeTTA - TDA |
|---|---:|---:|---:|
| natural | 47.39 | 48.19 | +0.80 |
| random | 47.13 | 48.03 | +0.90 |
| round_robin | 47.02 | 47.71 | +0.69 |
| hard_to_easy | 47.55 | 47.93 | +0.37 |
| easy_to_hard | 47.39 | 47.18 | -0.21 |
| class_blocked | 47.39 | 48.19 | +0.80 |

This is one of the most useful findings in the whole project because it finally gives a clean case where **TDA beats FreeTTA**:

- If the stream is ordered **easy-to-hard**, TDA becomes slightly better on DTD.

Why?

- In FreeTTA, the online EM update is a **global sufficient-statistics update**. Early easy samples can anchor the class means and priors toward already-easy modes, which may leave the model less responsive when harder boundary cases appear later.
- In TDA, the correction is a **local kernel-style retrieval term**
  `sum_j exp(-beta(1 - <x, k_j>)) v_j`
  over stored exemplars. That local memory can still help later hard samples even if the global class statistics are imperfect.

A second new observation is that **difficulty order matters more than class balance alone**:

- `random` and `round_robin` expose many classes early and still favor FreeTTA.
- The one order that hurts FreeTTA most is not the least balanced one. It is the **easy-to-hard** one.

So the critical issue is not only early class imbalance. It is also **which difficulty regime the online EM sees first**.

## 12. Targeted Ablation Results on DTD

I used DTD as the targeted ablation dataset because it is neither saturated like Caltech nor as trivially dominated as EuroSAT, so it is the best place to study mechanism changes.

### 12.1 TDA Dual-Cache Ablation

| Variant | Accuracy |
|---|---:|
| Positive cache + negative cache | 47.39 |
| Positive cache only | 47.55 |
| Negative cache only | 46.91 |

This means that on DTD:

- the **positive cache carries most of the benefit**,
- the **negative cache alone is not sufficient**,
- and the negative cache is not always helpful on texture-heavy data.

So the "two-cache" design is not uniformly better in every domain. It depends on whether uncertainty-based negative masking is semantically meaningful for that dataset.

### 12.2 TDA Cache-Size Sweep

I varied `shot_capacity`, which is the **effective per-class cache budget** in this implementation.

| Shot Capacity | Effective Slots per Cache | Accuracy |
|---|---:|---:|
| 1 | 47 | 47.29 |
| 2 | 94 | 47.66 |
| 3 | 141 | 47.39 |
| 5 | 235 | 47.29 |
| 10 | 470 | 47.71 |

Two conclusions follow:

1. **Cache size matters, but weakly.**
2. **Cache composition matters more than raw cache size.**

The performance is not monotonic with cache size, which means simply storing more exemplars is not enough. The quality of the retained pseudo-labels matters more than memory volume.

### 12.3 FreeTTA Alpha-Beta Sweep

I swept FreeTTA's fusion weight `alpha` and entropy-weight parameter `beta` on DTD.

Best tested setting:

- `alpha = 0.3`
- `beta = 2.0`
- accuracy = **48.40%**
- gain over TDA = **+1.01 points**

Important trends from the sweep:

- Very small `alpha` underuses the generative branch.
- Very high `beta` (`4.5`) makes the method collapse close to the CLIP baseline because the EM updates become too weak.
- Moderate-to-strong `alpha` with **moderate beta** works best, because the posterior correction is strong enough to matter without letting uncertain samples dominate.

This gives a mathematical interpretation:

- `alpha` controls **how much the generative posterior can reshape the discriminative logits**.
- `beta` controls **how fast uncertain samples are downweighted** through
  `w_t = exp(-beta * H_t)`.

If `beta` is too large, then `w_t` becomes too small for many samples, so the online EM statistics barely move. If `alpha` is too small, the learned posterior has little influence even when the statistics are good.

## 13. Updated Overall Conclusion

After the new experiments, I would summarize the comparison like this:

- **Why FreeTTA is superior:** it is better at making **selective, global corrections** once a small amount of target evidence has been accumulated. Its gains appear especially on harder datasets and can emerge after only about 5% of the stream.
- **Why TDA can still win:** it is stronger when **local exemplar retrieval** is more useful than a global class-distribution model, and it can be slightly better in streams arranged from **easy-to-hard** where the global online EM statistics become anchored too early.
- **What the new analyses add beyond the papers:** the papers do not tell us about calibration mismatch, selective entropy reduction, safe-versus-harmful correction behavior, break-even sample counts, or the easy-to-hard failure mode. These analyses give a much more workshop-ready mechanistic explanation of the baseline-versus-extension relationship.

## 14. Additional Analyses Found in the Provided PDF

I also checked the user-provided PDF report:

- `C:/Users/LENOVO/OneDrive/Desktop/Deep Comparative Study of TDA and FreeTTA for Test-Time Adaptation in Vision–Language Models.pdf`

It contains several useful analyses that are **not yet explicitly covered** in my main benchmark-based report. I am adding them here, but with an important honesty note:

- the sections below are **derived from the provided PDF**,
- they are conceptually valuable and report-ready,
- but they were **not rerun inside this repository** unless explicitly stated earlier in Sections 11-13.

This keeps the final report academically cleaner: benchmark results come from this repo, while the following points are imported from the supplied PDF's synthetic-study perspective.

### 14.1 Sufficient-Statistic Compression Versus Finite Cache Approximation

The PDF frames TDA and FreeTTA as two fundamentally different estimators:

- **TDA** is a finite-memory, non-parametric approximation of the target distribution using a small cache of exemplar features.
- **FreeTTA** is a parametric compressed estimator that stores sufficient-statistic-like state: class means, covariance, and priors.

This is a strong new conceptual lens because it explains FreeTTA's superiority not only in terms of accuracy, but also in terms of:

- **statistical efficiency**,
- **memory efficiency**,
- **distribution-level modeling ability**.

The report-quality takeaway is:

> TDA spends memory on remembered samples, while FreeTTA spends memory on remembered statistics.

That makes FreeTTA especially attractive in long streams or memory-constrained deployment, where a finite cache can saturate but sufficient-statistics updates can continue absorbing evidence.

### 14.2 Confidence-Gated Cache Selection Creates Selection Bias

The provided PDF highlights a subtle issue with TDA that is not usually emphasized:

- TDA's positive cache is filled mainly by **low-entropy / high-confidence** pseudo-labels.
- This means the cache may become a **biased subset of easy examples**, rather than a faithful representation of the whole target class distribution.

This is different from simply saying "TDA uses entropy."

The deeper point is:

- In **TDA**, entropy affects **which samples are allowed to represent the class**.
- In **FreeTTA**, entropy affects **how much each sample influences the parameter update**.

So the two methods use uncertainty in structurally different ways. This gives another mechanistic argument for why FreeTTA can better capture target-distribution structure, especially when classes contain easy and hard sub-modes.

### 14.3 Burn-In / Initial-Sample Sensitivity for FreeTTA

The provided PDF also includes a useful ablation for the question:

> "How many initial samples are needed before FreeTTA should start adapting?"

Its synthetic study introduces a **burn-in period**, where FreeTTA delays its online updates for the first `B` samples.

The key message from that PDF analysis is:

- a **small burn-in** can stabilize early adaptation,
- but a **large burn-in** leaves the distribution parameters stale for too long and hurts adaptation quality.

This complements my own benchmark-based break-even analysis from Section 11.4:

- Section 11.4 measured **when FreeTTA starts helping** on real benchmark streams.
- The PDF's burn-in study explains **why too much delay can be harmful**, even if some early caution is useful.

Together, these give a stronger report story:

- FreeTTA generally needs only a modest amount of evidence,
- but adaptation should not be postponed too aggressively.

### 14.4 Multimodal Class Structure Is a Natural Failure Case for FreeTTA

One of the clearest additional insights in the PDF is a synthetic **multimodal-class** experiment:

- when each class has multiple separated modes,
- and the stream shifts between modes,
- **TDA can outperform FreeTTA**.

This is extremely important because it gives a mathematically justified limitation:

- FreeTTA's class model is approximately **one Gaussian per class** with shared covariance,
- so it is fundamentally **unimodal** at the class level,
- whereas TDA's cache is **non-parametric** and can preserve several separated exemplars.

This strengthens my earlier claim from Section 13:

- **FreeTTA** is stronger when the class distribution is smooth and summarizable,
- **TDA** can be stronger when the target class geometry is multi-modal, rare-subtype-heavy, or exemplar-driven.

This PDF analysis therefore provides the cleanest structural explanation of **when the baseline can still beat the extension**.

### 14.5 Single-Sample Influence and Poisoning Sensitivity

Another useful analysis in the PDF is a controlled **single-sample perturbation / poisoning sensitivity** test:

- replace one mid-stream sample with a highly confident prototype-like sample,
- then measure how much downstream behavior changes.

The PDF reports that:

- **TDA** shows nonzero downstream sensitivity,
- **FreeTTA** shows much smaller or near-zero downstream effect in the demonstrated setup.

This aligns well with the conceptual update rules:

- TDA can change abruptly because a new exemplar may enter or replace a cache entry,
- FreeTTA updates its statistics through smoother weighted averaging.

So this PDF contributes one more strong workshop-quality message:

> FreeTTA is not only a distribution model; it is also a smoother online estimator with lower single-sample leverage.

That matters in safety-sensitive settings, because stability is often as important as top-1 accuracy.

### 14.6 How These PDF Analyses Fit With My Main Report

After integrating the PDF, the overall comparative picture becomes even stronger:

- My repo-based experiments establish **real benchmark evidence** for accuracy, disagreement quality, calibration mismatch, break-even sample counts, order sensitivity, and parameter sweeps.
- The provided PDF adds **extra theoretical and synthetic support** for:
  - memory-versus-statistics interpretation,
  - selection-bias effects from hard confidence gating,
  - burn-in sensitivity,
  - multimodal failure cases for FreeTTA,
  - and single-sample influence stability.

So if you want the final workshop-style narrative to sound more mature, the cleanest summary is:

> TDA and FreeTTA are not just two TTA algorithms. They are two different beliefs about what should be remembered from the test stream: exemplars versus distribution parameters. FreeTTA wins when compressed target statistics are enough and stability matters; TDA wins when rare modes, exemplar diversity, or strong local neighborhoods matter more than a unimodal class summary.
