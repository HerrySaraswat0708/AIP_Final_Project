# Comparative Analysis of TDA and FreeTTA for Test-Time Adaptation

## 1. Papers and Framing

This report compares:

1. **TDA**: *Training-Free Test-Time Adaptation for Vision-Language Models*.
2. **FreeTTA**: *Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM*.

FreeTTA is not literally built on top of TDA, but the two methods are directly comparable because they solve the same problem under the same frozen CLIP-style backbone setting. The clean comparison is therefore not "baseline vs add-on module" but **memory-based local adaptation vs online global distribution modeling**.

## 2. Shared Backbone, Losses, and Why the Comparison is Meaningful

Both methods inherit the same pretrained CLIP representation space. That space is created by CLIP's **contrastive image-text pretraining loss**, which aligns image features and text features by maximizing similarity of matched pairs and minimizing similarity of mismatched pairs.

That matters for the comparison because neither TDA nor FreeTTA retrains the backbone at test time:

- The **architecture** is the same frozen image encoder plus frozen text encoder.
- The **adaptation target** is the same: improve test-time classification under shift without source retraining.
- The main difference is **how they exploit the geometry of the CLIP feature space**.

So the comparison reduces to this question:

> Given the same frozen CLIP geometry, is it better to adapt by storing a few local exemplars, or by estimating global class statistics online?

## 3. What Each Method is Actually Doing

### 3.1 TDA

TDA does not optimize a test-time loss with backpropagation. Instead, it performs **retrieval-style logit correction** using online caches.

Mechanically it has:

- a **positive cache** that adds support for classes similar to stored confident examples,
- a **negative cache** that suppresses classes when entropy falls in an uncertainty band,
- entropy gates that decide whether a test sample is informative enough to be stored.

So TDA is best understood as a **non-parametric local memory method**. It is strong when nearby target exemplars are informative and a few stored examples per class are enough.

### 3.2 FreeTTA

FreeTTA treats each class as a Gaussian-like component in feature space and performs an **online EM-style update**.

Mechanically it has:

- class means initialized from text embeddings,
- class priors updated online,
- a shared covariance matrix,
- an entropy-derived weight that scales how much each sample influences the update,
- final prediction from combining CLIP logits with a generative logit term.

So FreeTTA is best understood as a **global distribution-modeling method**. It is strong when the target shift is class-level and enough stream evidence exists to estimate meaningful class statistics.

## 4. Paper-Level Quantitative Comparison

To compare the two papers fairly, I use the numbers reported in their tables for the shared ViT-B/16 datasets available in this repository.

| Dataset | TDA | FreeTTA | FreeTTA - TDA |
|---|---:|---:|---:|
| Caltech101 | 94.24 | 94.63 | +0.39 |
| DTD | 47.40 | 46.96 | -0.44 |
| EuroSAT | 58.00 | 62.93 | +4.93 |
| OxfordPets | 88.63 | 90.11 | +1.48 |
| ImageNetV2-style evaluation used in this repo | 64.67 | 64.92 | +0.25 |

### Immediate quantitative takeaway

FreeTTA is **not uniformly superior**. It is better on most shared datasets, but the gain is highly uneven:

- **large** on `EuroSAT`,
- **moderate** on `OxfordPets`,
- **small** on `Caltech101` and `ImageNet`,
- **negative** on `DTD`.

That pattern already suggests that the real distinction is not model size or optimization power. It is the **match between dataset geometry and adaptation mechanism**.

## 5. New Analyses Not Presented in Either Paper

I replaced the old multi-file analysis setup with one consolidated pipeline:

- [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)

It is designed to support paper comparison plus new mechanism-level analysis under one script.

The main new analyses are:

1. **Geometry Alignment Probe**
2. **Flip Efficiency Analysis**
3. **Adaptation Latency / Break-Even Analysis**
4. **Cache Pressure Analysis**
5. **Disagreement Reliability Analysis**

These are described mathematically in [queries.md](/home/herrys/projects/AIP-Final-project/queries.md).

## 6. New Analysis 1: Geometry Alignment Probe

This is the strongest new analysis.

I compare two post-hoc oracle probes in the same CLIP feature space:

- an **oracle centroid classifier**,
- an **oracle leave-one-out 1-NN classifier**.

Why this is useful:

- The centroid probe matches FreeTTA's modeling assumption: classes are well represented by global means.
- The 1-NN probe matches TDA's operating bias: decisions can be improved using local exemplar neighborhoods.

### Interpretation rule

- If **centroid oracle > 1-NN oracle**, the dataset geometry is more favorable to FreeTTA.
- If **1-NN oracle > centroid oracle**, the dataset geometry is more favorable to TDA.

### What this says about the shared datasets

- **EuroSAT**: a class-level remote-sensing problem with relatively coherent class clusters and long streams. This is exactly the sort of setting where global distribution modeling should dominate sparse memory.
- **DTD**: texture categories are often multi-modal and locally structured. A single centroid or shared-covariance Gaussian is a weaker description here, so TDA remains competitive or superior.
- **Caltech101 / OxfordPets**: both are relatively well-structured categories, so centroid modeling is plausible, but the headroom is smaller because CLIP is already strong.

This directly explains why FreeTTA's biggest paper gain is on EuroSAT and why DTD is the main failure case.

## 7. New Analysis 2: Flip Efficiency

Instead of only measuring final accuracy, I analyze **what happens when a method changes CLIP's decision**.

A prediction change can be:

- **beneficial**: CLIP was wrong, method becomes correct,
- **harmful**: CLIP was correct, method becomes wrong.

This is a better mechanism-level comparison than plain accuracy because the two methods fail for different reasons:

- TDA flips are caused by **cache retrieval plus entropy-gated suppression**.
- FreeTTA flips are caused by **drift in the estimated generative model**.

### Why this matters

A good adaptation method should not just change predictions often. It should change them **selectively**.

The useful quantity is not just change rate, but:

- beneficial flip precision,
- harmful flip rate on CLIP-correct samples,
- confidence/entropy after beneficial vs harmful flips.

### Expected pattern

- **TDA** should be better when the useful correction is highly local and only a few samples need to flip.
- **FreeTTA** should be better when many medium-confidence samples benefit from consistent global drift in class statistics.

## 8. New Analysis 3: Adaptation Latency

This measures **how many test samples each method needs before it starts helping**.

I compute a rolling-window break-even point versus CLIP and versus the competing method.

This is a direct reflection of internal mechanics:

- TDA can improve quickly because one strong stored exemplar can immediately help the next similar sample.
- FreeTTA often needs more samples before its estimated means and priors become useful.

### Interpretation

- **short break-even latency** favors TDA,
- **longer latency but larger late-stage gain** favors FreeTTA.

This separates **fast local adaptation** from **slow but global distribution estimation**.

## 9. New Analysis 4: Cache Pressure

TDA has a hard memory bottleneck. With class count `C`, positive shot capacity `S+`, and negative shot capacity `S-`, the total retained memory is approximately:

`MemorySlots = C * (S+ + S-)`

A useful derived quantity is:

`CachePressure = N / MemorySlots`

where `N` is the number of test samples.

### Interpretation

- **High cache pressure** means the test stream contains much more evidence than TDA can retain.
- In that case, FreeTTA has an advantage because it compresses evidence into sufficient statistics instead of discarding almost all of it.

This is especially relevant for `EuroSAT`, where the stream is long relative to the tiny TDA cache.

## 10. New Analysis 5: Disagreement Reliability

I evaluate only the samples where TDA and FreeTTA disagree.

This isolates the regime where the methods are making genuinely different corrections rather than both inheriting CLIP's answer.

### Interpretation

- If FreeTTA is consistently better on disagreement samples, its global model is correcting errors that sparse local memory misses.
- If TDA is better on disagreements, local exemplar reasoning is capturing structure that a class-mean model washes out.

This analysis is especially valuable because it avoids the misleading comfort of overall accuracy, where both methods can look similar simply because CLIP is already strong.

## 11. When FreeTTA Should Be Superior

FreeTTA should be superior when most of the following hold:

1. **The class geometry is globally coherent.**
   - Evidence: centroid oracle stronger than 1-NN oracle.

2. **The stream is long relative to TDA's memory budget.**
   - Evidence: high cache pressure.

3. **The main target shift is class-level drift, not just local neighborhood irregularity.**
   - Evidence: global statistics continue improving late in the stream.

4. **Late-stage gains matter more than early adaptation speed.**
   - Evidence: longer break-even latency but stronger late rolling advantage.

5. **Prediction changes are globally consistent rather than sparse and local.**
   - Evidence: high beneficial-flip precision on medium/hard samples.

### Datasets where this logic points toward FreeTTA

- **EuroSAT**: strongest case.
- **OxfordPets**: moderate case.
- **Caltech101**: plausible, but only modest gains because CLIP is already near saturation.
- **ImageNet**: only slight expected advantage because the class space is large and the shared-covariance assumption is demanding.

## 12. When TDA Should Be Superior

TDA should be superior when most of the following hold:

1. **Useful evidence is local rather than globally centroid-like.**
   - Evidence: 1-NN oracle stronger than centroid oracle.

2. **The dataset has multi-modal classes or strong local texture variation.**
   - A single Gaussian per class is a poor fit.

3. **Fast early corrections matter more than long-term statistical estimation.**
   - Evidence: short break-even latency.

4. **The method must avoid harmful model drift.**
   - TDA is more conservative because it never tries to estimate a full target distribution.

5. **Only a small number of local corrections are needed.**
   - Then a sparse cache is enough.

### Datasets where this logic points toward TDA

- **DTD** is the clearest case.
- It can also stay competitive in easy, low-headroom settings where there is little to gain from fitting global statistics.

## 13. Final Comparative Judgment

The best way to understand these papers is not "which one wins more datasets?" but:

> **TDA and FreeTTA are exploiting different kinds of structure in the same CLIP feature space.**

- TDA exploits **local neighbor evidence**.
- FreeTTA exploits **global class-statistic evidence**.

That is why the winner changes by dataset.

### My main new conclusion

The strongest new conclusion from this comparison is:

> **The decisive factor is not simply how much target information exists, but whether that information is better compressed as a few useful exemplars or as global class statistics.**

This conclusion is not stated directly in either paper, and it is what the new analyses are designed to test.

## 14. Deliverables

- New single analysis script: [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)
- Methodology and mathematical background: [queries.md](/home/herrys/projects/AIP-Final-project/queries.md)
- This report: [ASSIGNMENT_REPORT.md](/home/herrys/projects/AIP-Final-project/ASSIGNMENT_REPORT.md)

