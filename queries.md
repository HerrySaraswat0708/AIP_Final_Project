# Queries and Comparison Design for TDA vs FreeTTA

## 1. Goal

This document explains the comparison questions used to analyze **TDA** and **FreeTTA**, the mathematical motivation behind each comparison, and how each comparison helps decide:

- when **FreeTTA** should be better,
- when **TDA** should be better,
- what those outcomes reveal about the architecture, losses, and internal workings of both methods.

The implementation for these comparisons is consolidated in:

- [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)

## 2. Shared Premise: Same Backbone, Different Adaptation Geometry

Both methods operate on the same pretrained CLIP feature space.

Let:

- `f(x)` be the image feature,
- `g(t_y)` be the text feature for class `y`.

CLIP itself predicts using similarity logits of the form:

`z_y = f(x)^T g(t_y)`

The backbone is frozen. Therefore the comparison is entirely about **how each method modifies the decision rule at test time**.

### Why the CLIP loss matters

CLIP is trained by a contrastive loss. That means the feature space already contains semantic alignment information, but it does **not** guarantee that the target domain will be well described by:

- a few local exemplars,
- or a single Gaussian-like class cluster.

That ambiguity is exactly why TDA and FreeTTA can behave differently on the same dataset.

## 3. Query 1: Which Geometry Fits the Dataset Better?

### Question

Is the dataset better described by **local neighbors** or by **global class centers**?

### Comparison used

I use two oracle probes in the frozen CLIP feature space:

1. **Centroid oracle**
2. **Leave-one-out 1-NN oracle**

### Mathematical background

#### Oracle centroid classifier

For class `y`, define its oracle centroid:

`c_y = mean_{i: y_i = y} f(x_i)`

Prediction:

`y* = argmax_y f(x)^T c_y`

This is conceptually aligned with FreeTTA, because FreeTTA models each class with a mean vector and global statistics.

#### Oracle leave-one-out 1-NN classifier

For sample `x_i`, predict using the nearest other sample in feature space:

`j* = argmax_{j != i} f(x_i)^T f(x_j)`

Then assign label `y_{j*}`.

This is conceptually aligned with TDA, because TDA relies on exemplar similarity and retrieval-like correction.

### New metric

**Geometry Alignment Score**:

`GAS = Acc_centroid_oracle - Acc_1NN_oracle`

### Interpretation

- `GAS > 0`: geometry favors FreeTTA-like reasoning.
- `GAS < 0`: geometry favors TDA-like reasoning.

### What this tells us

- If `GAS` is positive and large, FreeTTA's global modeling assumption is plausible.
- If `GAS` is negative, local neighborhood structure matters more, so TDA should be favored.

### Cases

- **FreeTTA superior**: datasets with coherent class clusters, such as EuroSAT-like scenarios.
- **TDA superior**: datasets with multi-modal or texture-heavy local structure, such as DTD-like scenarios.

## 4. Query 2: Are Prediction Changes Helpful or Harmful?

### Question

When a method changes CLIP's prediction, does it usually fix a mistake or create one?

### Comparison used

For each method, every sample belongs to one of four groups:

1. unchanged correct,
2. unchanged wrong,
3. beneficial flip,
4. harmful flip.

### Mathematical background

Let:

- `c_clip(i)` be CLIP correctness on sample `i`,
- `c_m(i)` be correctness of method `m`,
- `p_clip(i)` and `p_m(i)` be predictions.

Then:

**Beneficial flip**:

`BF_m(i) = 1[p_clip(i) != p_m(i)] * 1[c_clip(i)=0] * 1[c_m(i)=1]`

**Harmful flip**:

`HF_m(i) = 1[p_clip(i) != p_m(i)] * 1[c_clip(i)=1] * 1[c_m(i)=0]`

### New metrics

1. **Beneficial Flip Precision**

`BFP_m = sum BF_m / max(sum Changed_m, 1)`

2. **Harmful Flip Rate on CLIP-Correct Samples**

`HFR_m = sum HF_m / max(sum 1[c_clip=1], 1)`

### Interpretation

- High `BFP` means the method changes predictions selectively and well.
- High `HFR` means the method overcorrects and damages already-correct CLIP predictions.

### What this tells us internally

- TDA errors come from **bad retrieval/gating**.
- FreeTTA errors come from **bad distribution drift**.

### Cases

- **FreeTTA superior** when many medium-confidence CLIP mistakes can be corrected consistently by global class statistics.
- **TDA superior** when only sparse local mistakes need correction and over-modeling the whole stream is risky.

## 5. Query 3: How Long Until Adaptation Starts Helping?

### Question

Does the method help immediately, or only after accumulating enough target evidence?

### Comparison used

I compute rolling accuracy gains against CLIP and against the competing method, then find the first time the rolling gain becomes positive.

### Mathematical background

For method `m`, define rolling accuracy over window `w`:

`A_m(t; w) = mean_{k=t-w+1}^t 1[c_m(k)=1]`

Then define break-even against CLIP as the smallest `t` such that:

`A_m(t; w) - A_clip(t; w) > 0`

### New metric

**Break-Even Latency**:

`BEL_m = min { t : A_m(t; w) > A_clip(t; w) }`

and similarly for FreeTTA versus TDA.

### Interpretation

- Small `BEL`: method adapts quickly.
- Large `BEL`: method needs more evidence before its internal state becomes useful.

### Cases

- **TDA superior** when early local corrections matter most.
- **FreeTTA superior** when late-stage gains are worth waiting for.

## 6. Query 4: Is TDA Memory Under Compression Pressure?

### Question

Is the target stream much larger than the amount of evidence TDA can retain?

### Comparison used

I compare the number of stream samples to the total number of cache slots TDA can keep.

### Mathematical background

If the number of classes is `C`, positive shot capacity is `S+`, and negative shot capacity is `S-`, then TDA can retain at most:

`Slots = C * (S+ + S-)`

For test stream length `N`, define:

`CachePressure = N / Slots`

### Interpretation

- Low pressure: TDA memory is less constrained.
- High pressure: TDA must aggressively compress and discard target evidence.

### Why this matters

FreeTTA does not store a sparse set of exemplars. It stores sufficient statistics. Therefore high cache pressure is a structural advantage for FreeTTA.

### Cases

- **FreeTTA superior** when `CachePressure` is high.
- **TDA superior** when `CachePressure` is low and local exemplars are enough.

## 7. Query 5: Which Method Is More Reliable Exactly When They Disagree?

### Question

When TDA and FreeTTA produce different answers, which one should we trust more?

### Mathematical background

Define disagreement set:

`D = { i : p_tda(i) != p_freetta(i) }`

Then compare:

`Acc_tda(D)` and `Acc_freetta(D)`

### Interpretation

This isolates the truly informative region of the comparison. If both methods agree, the difference is not interesting. The key question is what happens when their inductive biases actually diverge.

### Cases

- **FreeTTA superior** if disagreement accuracy is higher on high-entropy samples and in late stream phases.
- **TDA superior** if disagreement accuracy is higher on locally structured or multi-modal samples.

## 8. Query 6: What Do the Internal Signals Say?

### Question

Can we relate performance differences to the internal signals each method computes?

### TDA signals used

- positive cache size,
- negative cache size,
- negative-gate-open rate.

### FreeTTA signals used

- EM weight,
- mean update norm,
- total mean drift,
- prior entropy,
- covariance trace.

### Why these matter

These are direct summaries of each method's internal operating regime.

#### TDA

- large positive cache usage means retrieval has enough support,
- large negative-gate activity means entropy-gated suppression is doing real work,
- small caches plus high stream length indicate information bottleneck.

#### FreeTTA

- high EM weight means the method trusts CLIP and updates aggressively,
- large mean drift means the target distribution estimate is moving far from the original text anchor,
- prior entropy shows whether the estimated class distribution is collapsing or diversifying,
- covariance trace shows how much global spread the model is estimating.

## 9. Which Cases Favor FreeTTA?

FreeTTA should be superior when most of these signals align:

1. `GAS > 0`.
2. Cache pressure is high.
3. Break-even latency is acceptable and late-stage gains are strong.
4. Beneficial flip precision is high on medium/hard samples.
5. Disagreement accuracy favors FreeTTA.
6. Internal drift is meaningful but not chaotic.

This usually describes datasets like:

- **EuroSAT**: coherent class-level shift, long stream, low class count, high compression pressure on TDA.
- **OxfordPets**: moderate centroid structure and enough stream evidence to help distribution modeling.

## 10. Which Cases Favor TDA?

TDA should be superior when most of these signals align:

1. `GAS < 0`.
2. Break-even latency is short and early gains matter.
3. Beneficial local flips occur without large harmful drift.
4. Cache pressure is not too high.
5. Disagreement accuracy favors TDA.
6. The dataset is locally structured, multi-modal, or texture-heavy.

This usually describes datasets like:

- **DTD**: a poor match for one-global-centroid-per-class reasoning, but a better match for local exemplar correction.

## 11. Why This Comparison Is New Relative to the Papers

Neither paper gives a unified answer to:

- whether the dataset geometry itself favors local or global adaptation,
- how quickly each method begins to help,
- whether each method's prediction changes are mostly helpful or harmful,
- how memory bottlenecks change the comparison,
- which method is more trustworthy exactly when they disagree.

That is why these queries are useful. They turn the comparison from a single accuracy table into a mechanism-level explanation.

## 12. Final Practical Rule

A compact rule for deciding between the two methods is:

- choose **FreeTTA** when target evidence is abundant and globally coherent,
- choose **TDA** when the useful structure is local, sparse, or multi-modal.

Equivalently:

- **FreeTTA wins when distribution compression is the right abstraction.**
- **TDA wins when exemplar retrieval is the right abstraction.**

