# Comparative Analysis of TDA and FreeTTA for Test-Time Adaptation

This report is written for **proposal option 2** of the course project:

> choose one recent paper and a stronger later method for the same problem, explain why the later method should improve over the baseline, add new analysis not present in either paper, and experimentally analyze when the improvement occurs.

## 1. Problem Motivation

Large vision-language models such as CLIP are strong zero-shot classifiers, but their accuracy drops under test-time distribution shift. The practical question is whether we can improve a frozen CLIP model at test time without retraining on source data.

This project studies two recent answers to that question:

1. **TDA**: a training-free cache-based adaptation method.
2. **FreeTTA**: a later online EM-style adaptation method.

The key motivation is not only to compare final accuracies, but to understand:

- what each method is doing to CLIP predictions,
- what internal statistics each method changes,
- when one method should theoretically beat the other,
- why that expected ordering may or may not appear in our own benchmark.

## 2. Papers Chosen and Why the Comparison Is Meaningful

This project compares:

1. **TDA**: *Training-Free Test-Time Adaptation for Vision-Language Models*.
2. **FreeTTA**: *Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM*.

`FreeTTA` is not a literal code extension of `TDA`, but it is a later method for the **same frozen-CLIP test-time adaptation problem**. That makes the comparison meaningful:

- both methods use the same pretrained CLIP representation space,
- both avoid backbone retraining at test time,
- both try to improve classification under target-domain shift,
- they differ mainly in **how they modify CLIP's decision rule**.

So the real comparison is:

> given the same frozen CLIP feature space, is it better to adapt using local exemplar memory or global class-statistic modeling?

## 3. What Each Method Does in Simple Words

### 3.1 CLIP Baseline

CLIP predicts using image-text similarity in a shared embedding space:

`z_y = f(x)^T g(t_y)`

where `f(x)` is the image feature and `g(t_y)` is the text feature for class `y`.

CLIP itself does not adapt at test time. It is the frozen baseline.

### 3.2 TDA

TDA is best understood as **memory-based local correction**.

It maintains:

- a positive cache of confident target examples,
- a negative cache used for suppression under uncertainty,
- entropy gates that decide when a sample is reliable enough to store.

At test time, TDA does not change the CLIP backbone. Instead, it changes the output logits by adding retrieval-style support from the cache.

Intuition:

- if a new test sample is close to previously seen confident examples,
- TDA can push CLIP toward the locally supported class.

So TDA helps when **local neighborhoods** in feature space are reliable.

### 3.3 FreeTTA

FreeTTA is best understood as **online global distribution modeling**.

It maintains:

- class means,
- class priors,
- a shared covariance estimate,
- an entropy-based weight for how much each sample should influence the online update.

FreeTTA also leaves the CLIP backbone frozen. But instead of using exemplar retrieval, it changes the output by combining CLIP logits with a generative score based on the evolving class statistics.

Intuition:

- if the target stream reveals stable class-level drift,
- then estimating global class statistics can correct CLIP systematically.

So FreeTTA should help when the target domain is better described by **class-level global structure** than by sparse local memory.

## 4. Theoretical Expectation From the Papers

From the papers, the broad expectation is:

- `FreeTTA` should be stronger overall on most shared datasets,
- especially when the target stream is long and class-level statistics can be estimated well,
- while `TDA` should remain strong on datasets where local exemplar structure matters more.

Paper-reported numbers for shared datasets show that expectation clearly:

| Dataset | TDA | FreeTTA | FreeTTA - TDA |
| --- | ---: | ---: | ---: |
| Caltech101 | 94.24 | 94.63 | +0.39 |
| DTD | 47.40 | 46.96 | -0.44 |
| EuroSAT | 58.00 | 62.93 | +4.93 |
| OxfordPets | 88.63 | 90.11 | +1.48 |
| ImageNet-style setting reported by the papers | 64.67 | 64.92 | +0.25 |

So the paper-level theory is not that `FreeTTA` wins every dataset. The real theoretical claim is:

- `FreeTTA` should win more often,
- its gains should be largest when global class statistics are the right abstraction,
- `TDA` should remain competitive or better when local retrieval structure is more informative.

## 5. What We Reproduced in This Repository

Our final tuned full-dataset runs in this repository are:

| Dataset | CLIP | TDA | FreeTTA | TDA - CLIP | FreeTTA - CLIP | FreeTTA - TDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Caltech | 0.9355 | 0.9408 | 0.9359 | +0.0053 | +0.0004 | -0.0049 |
| DTD | 0.4394 | 0.4702 | 0.4638 | +0.0309 | +0.0245 | -0.0064 |
| EuroSAT | 0.4843 | 0.5600 | 0.5106 | +0.0757 | +0.0263 | -0.0494 |
| ImageNet | 0.6237 | 0.6282 | 0.6237 | +0.0045 | +0.0000 | -0.0045 |
| Pets | 0.8839 | 0.8861 | 0.8842 | +0.0022 | +0.0003 | -0.0019 |

These results do **not** match the overall paper expectation. In this reproduced benchmark:

- `TDA` beats `CLIP` on all five datasets,
- `FreeTTA` improves over `CLIP` on some datasets but not by much,
- `TDA` remains stronger than `FreeTTA` on all five datasets in the final tuned table.

## 6. Why the Paper Expectation and Our Reproduction Differ

This mismatch is central to the viva story and should be stated clearly.

The comparison in this repository is still meaningful, but the numbers do not perfectly follow the paper ordering. The most likely reasons are:

1. **Protocol mismatch**
   The repository evaluates both methods in a shared extracted-feature benchmark rather than exactly reproducing every paper's full raw-image pipeline and official evaluation environment.

2. **Implementation availability mismatch**
   `TDA` has an official public implementation that we checked against.
   `FreeTTA` does not have the same level of public official code availability in this repo workflow, so its reproduction is more dependent on paper-based reimplementation and tuning.

3. **Benchmark-specific geometry**
   The frozen CLIP features used in this repository may expose local structure differently from the setups used in the papers.

4. **High baseline accuracy on several datasets**
   On datasets such as Caltech and Pets, CLIP is already very strong. In that regime, even a theoretically better adaptive method can fail to show large gains because the room for improvement is tiny.

The important point is:

> the reproduction gap does not invalidate the project. It creates the main research question for the project: why does the expected ordering fail here, and what does that reveal about the methods?

## 7. Our Main Contribution Beyond Both Papers

The main contribution of this project is **not** claiming a new better method for option 2. The main contribution is a unified comparison framework that explains how the two methods behave.

The new analyses added in this repository are:

1. **Geometry Alignment Probe**
2. **Prediction Flip Analysis**
3. **Adaptation Latency / Break-Even Analysis**
4. **Cache Pressure Analysis**
5. **Disagreement Reliability Analysis**
6. **Internal-State Diagnostics**

These analyses are not presented as one unified mechanism-level framework in either paper.

## 8. What We Wanted to Understand

The project is organized around the following questions:

1. How are `TDA` and `FreeTTA` different from base `CLIP`?
2. What changes do they make to CLIP outputs?
3. Are those changes mostly helpful or harmful?
4. How quickly do they start helping?
5. What internal statistics do they update?
6. Where do they fail?
7. When should one method theoretically succeed while the other should struggle?

Those questions are formalized in [queries.md](/home/herrys/projects/AIP-Final-project/queries.md).

## 9. How TDA and FreeTTA Change CLIP Outputs

This is one of the most viva-relevant parts of the project.

### 9.1 How TDA changes CLIP outputs

TDA changes CLIP predictions by adding **retrieval-based support or suppression**:

- if a test sample is similar to stored positive-cache samples from a class, that class logit gets boosted,
- if uncertainty enters the negative-gate regime, some classes can be suppressed using the negative cache.

So TDA changes CLIP output in a **local, sample-specific way**.

It does not try to change the meaning of the whole dataset distribution. It tries to fix CLIP using nearby target evidence.

### 9.2 How FreeTTA changes CLIP outputs

FreeTTA changes CLIP predictions by updating **global class statistics**:

- class means move,
- priors change,
- covariance summarizes the stream,
- the final logit combines the original CLIP similarity with a generative score from the updated statistics.

So FreeTTA changes CLIP output in a **global, distribution-level way**.

It is effectively saying:

- "based on the whole test stream so far, this class center and prior should now look different from the original text anchor."

### 9.3 Why this difference matters

This directly explains the different failure modes:

- `TDA` can fail if local retrieved neighbors are misleading or too sparse.
- `FreeTTA` can fail if the estimated class distribution drifts in the wrong direction or if the Gaussian-style assumption is a poor fit.

## 10. New Comparison 1: Geometry Alignment

We compare:

- an oracle centroid classifier,
- an oracle leave-one-out 1-NN classifier.

Interpretation:

- centroid-friendly geometry supports the logic behind `FreeTTA`,
- local-neighbor-friendly geometry supports the logic behind `TDA`.

This comparison is important because it separates:

- "the method should win in theory"

from:

- "the feature geometry in this benchmark actually supports that theory."

In other words, this is how we test whether the benchmark geometry matches the paper intuition.

## 11. New Comparison 2: Prediction Flips

A method should not be judged only by final accuracy. It should also be judged by **what kind of prediction changes it makes**.

We measure:

- beneficial flips,
- harmful flips,
- beneficial flip precision,
- harmful flip rate on CLIP-correct samples.

This tells us whether the method is:

- selectively correcting CLIP,
- or overcorrecting and damaging already-correct predictions.

This is one of the cleanest ways to explain "what the method is doing to CLIP."

## 12. New Comparison 3: Adaptation Trajectory

We compare how the two methods evolve over the target stream.

The relevant question is not only:

- "who ends higher?"

but also:

- "who starts helping earlier?"
- "who needs more evidence?"
- "who gets stronger late in the stream?"

This is captured by:

- rolling accuracy curves,
- break-even latency,
- dataset-wise adaptation dynamics plots.

Interpretation:

- early gains support `TDA`'s local-memory advantage,
- later gains support `FreeTTA`'s global-statistics advantage.

## 13. New Comparison 4: Internal Statistics and Entropy

We also compare internal quantities that are not visible from final accuracy alone.

### TDA internals

- positive cache size,
- negative cache size,
- gate activity,
- effective cache pressure.

These tell us whether TDA has enough useful local evidence to retrieve from.

### FreeTTA internals

- EM weight,
- mean update norm,
- total mean drift,
- prior entropy,
- covariance trace.

These tell us whether FreeTTA is updating meaningfully, remaining stable, or drifting too far from the original CLIP anchor.

This is the key comparison for explaining:

- how both methods modify statistics relative to base `CLIP`,
- and how those statistic changes relate to success or failure.

## 14. When One Method Should Succeed and the Other Should Struggle

### FreeTTA should succeed when:

- class-level global structure is coherent,
- the stream is long enough to estimate useful statistics,
- cache compression would otherwise discard too much evidence,
- late-stage global correction matters more than immediate local fixes.

### TDA should succeed when:

- local neighborhood information is highly informative,
- classes are multi-modal or not well summarized by one mean,
- a few strong local corrections are enough,
- harmful global drift would be riskier than sparse retrieval.

These are not guarantees in a formal proof sense, but they are the most defensible mechanism-level conditions supported by the design of the methods.

## 15. Dataset-Wise Observations in This Repo

### Caltech

- CLIP is already near saturation.
- `TDA` gets a small gain.
- `FreeTTA` is almost tied with CLIP.
- Main lesson: low headroom makes it hard for global distribution modeling to show a clear advantage.

### DTD

- both methods improve over CLIP,
- `TDA` is slightly stronger in the final tuned table,
- texture-heavy local structure is consistent with TDA staying strong.

### EuroSAT

- both methods improve over CLIP,
- `TDA` gains much more than `FreeTTA` in this reproduction,
- this is the biggest mismatch with paper intuition and therefore the most interesting analysis case in the project.

### ImageNet

- gains are small for both methods,
- `TDA` gives a modest positive lift,
- `FreeTTA` is essentially tied with CLIP,
- this suggests weak practical headroom under the current benchmark.

### Pets

- CLIP is already strong,
- `TDA` gives a small improvement,
- `FreeTTA` gives only a marginal gain,
- again, limited headroom reduces the chance of seeing the paper-style ordering clearly.

## 16. Main Conclusion

The core conclusion of the project is:

> `TDA` and `FreeTTA` are not simply two methods where one always dominates the other. They exploit different structure in the same CLIP feature space.

- `TDA` exploits local target evidence through retrieval-like cache correction.
- `FreeTTA` exploits global target evidence through online distribution modeling.

The paper-level expectation says `FreeTTA` should often be stronger. Our reproduction does not fully match that expectation. That mismatch becomes the main scientific contribution of the project, because it motivates a deeper comparison of:

- geometry,
- trajectory,
- prediction flips,
- entropy,
- internal statistics,
- and disagreement cases.

So the final contribution of this work is a **mechanism-level explanation framework** for comparing test-time adaptation methods beyond a single accuracy table.

## 17. Code and Contribution Statement

This project includes both reused and newly written components.

### Reused or paper-derived components

- `TDA` logic was checked against the official public implementation.
- `FreeTTA` was implemented and tuned in this repository using the paper description and the shared benchmark setup.

### New work in this repository

- unified comparison runners,
- tuning scripts,
- final full-suite evaluation across datasets,
- new analysis metrics and plots,
- report-ready summaries and dataset-wise interpretations.

The main project contribution is therefore:

> a unified experimental and analytical framework for understanding how `TDA` and `FreeTTA` differ in practice, especially when reproduced behavior diverges from paper-level expectations.
