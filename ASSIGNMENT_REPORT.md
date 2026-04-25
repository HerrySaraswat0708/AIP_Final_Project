# Comparative Analysis of TDA and FreeTTA for Test-Time Adaptation

This report is written for **project option 2**:

> choose one recent paper and a stronger later method for the same problem, explain why the later method should improve over the baseline, add new analysis not present in either paper, and experimentally analyze when the improvement occurs.

## 1. Motivation

CLIP is a strong zero-shot vision-language model, but it is still vulnerable to test-time distribution shift. The central question of this project is:

> if the CLIP backbone is frozen, what is the best way to improve predictions online at test time?

This report compares two recent answers to that question:

1. **TDA**: a training-free cache-based adaptation method.
2. **FreeTTA**: an online EM-style global distribution modeling method.

The main goal is **not** only to compare final accuracy, but to understand:

- how each method modifies CLIP predictions,
- how confidence and entropy evolve,
- how FreeTTA updates global statistics,
- how TDA uses local cache memory,
- and when each method succeeds or fails relative to CLIP.

## 2. Papers Chosen

This project studies:

1. **TDA**: *Training-Free Test-Time Adaptation for Vision-Language Models*.
2. **FreeTTA**: *Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM*.

`FreeTTA` is not a literal code extension of `TDA`, but it is a later method for the **same frozen-CLIP test-time adaptation problem**. That makes the comparison scientifically meaningful:

- both methods use the same frozen CLIP representation space,
- both avoid test-time backbone retraining,
- both operate online over the target stream,
- and they differ mainly in **how they modify CLIP's decision rule**.

So the true comparison is:

> given the same frozen CLIP feature space, is it better to adapt by storing local target exemplars or by estimating global class statistics online?

## 3. Shared Frozen-CLIP Setup

Let:

- `f(x)` be the normalized image feature,
- `g(t_y)` be the normalized text feature for class `y`.

Base CLIP predicts by image-text similarity:

`z_y = f(x)^T g(t_y)`

This is the frozen baseline. Neither TDA nor FreeTTA retrains CLIP. Instead, both methods modify the **output rule** on top of the same backbone.

This is important for interpretation:

- if one method wins, it is not because it has a better backbone,
- it is because its **test-time adaptation mechanism** matches the dataset better.

## 4. What Each Method Does

### 4.1 CLIP

CLIP is the fixed baseline. It does not use target-stream history. It simply scores each class by similarity to the text prompt embedding.

### 4.2 TDA

TDA is best understood as **local memory-based correction**.

It keeps:

- a **positive cache** of confident target samples,
- a **negative cache** used for suppression under uncertainty,
- entropy gates controlling whether a sample should enter memory.

TDA changes CLIP predictions by using **retrieval-like support** from nearby cached examples. So it modifies CLIP in a **local and sample-specific** way.

Intuition:

- if target samples form reliable local neighborhoods,
- then a few stored target exemplars can correct CLIP quickly.

### 4.3 FreeTTA

FreeTTA is best understood as **online global distribution modeling**.

It maintains:

- class means `mu_y`,
- class priors `pi_y`,
- a shared covariance `Sigma`,
- and an entropy-based weight controlling how strongly each sample updates the statistics.

It then combines CLIP logits with a generative score derived from those evolving statistics.

So FreeTTA modifies CLIP in a **global, distribution-level** way.

Intuition:

- if the target stream reveals coherent class-level drift,
- then moving class means and priors should correct CLIP more systematically than sparse exemplar retrieval.

## 5. Why FreeTTA Should Improve Over TDA in Theory

The theoretical case for FreeTTA is:

1. it uses relationships among **all previous target samples**, not only a tiny retained memory,
2. it compresses target evidence into global sufficient statistics instead of discarding most of it,
3. it should be stronger when the target distribution shift is **class-level and coherent**,
4. it should especially help on long streams where the class statistics become better estimated over time.

That is why the paper expectation is that FreeTTA should often outperform TDA on many datasets.

But that expectation is **not universal**. The paper itself already contains an exception:

| Dataset | TDA | FreeTTA | FreeTTA - TDA |
| --- | ---: | ---: | ---: |
| Caltech101 | 94.24 | 94.63 | +0.39 |
| DTD | 47.40 | 46.96 | -0.44 |
| EuroSAT | 58.00 | 62.93 | +4.93 |
| OxfordPets | 88.63 | 90.11 | +1.48 |
| ImageNet-style setting reported by the papers | 64.67 | 64.92 | +0.25 |

So the correct theoretical statement is:

- `FreeTTA` should often be better overall,
- but `TDA` can still be stronger when local structure matters more than global class-level modeling.

## 6. Two Evaluation Modes in This Project

This repository now uses **two different evaluation modes**, and this distinction is important for viva.

### 6.1 Best-accuracy reproduction mode

This mode is for checking how close the repository gets to the paper-level behavior.

Output folder:

- [outputs/reproduction](/home/herrys/projects/AIP-Final-project/outputs/reproduction)
- [outputs/tuning](/home/herrys/projects/AIP-Final-project/outputs/tuning)

Purpose:

- audit paper/default settings on the local benchmark,
- run per-dataset recovery sweeps for `TDA` and `FreeTTA`,
- record the strongest results we can recover before doing the deeper aligned analysis.

### 6.2 Shared-order mechanism-analysis mode

This mode is for the **deep per-sample analysis** requested in the project.

Output folder:

- [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis)

Purpose:

- keep one shared test stream order,
- align the same sample across `CLIP`, `TDA`, and `FreeTTA`,
- store per-sample logits, predictions, confidence, entropy, internal states, and failure cases.

This second mode is the right setup for questions like:

- how do the methods change CLIP predictions?
- where are the harmful flips?
- how do entropy and confidence evolve?
- how do TDA caches and FreeTTA statistics evolve?
- where do the methods disagree?

Because this mode enforces a common stream order and aligned per-sample bookkeeping, its numbers do not need to match the best-accuracy table exactly.

## 7. Paper-Default Audit and Best Recovered Results

The most important reproduction check is the paper/default audit in [outputs/reproduction](/home/herrys/projects/AIP-Final-project/outputs/reproduction):

- [paper_default_tda_full.json](/home/herrys/projects/AIP-Final-project/outputs/reproduction/paper_default_tda_full.json)
- [paper_default_freetta_full.json](/home/herrys/projects/AIP-Final-project/outputs/reproduction/paper_default_freetta_full.json)
- [reproduction_audit.md](/home/herrys/projects/AIP-Final-project/outputs/reproduction/reproduction_audit.md)

On the current benchmark, the paper/default runs are:

| Dataset | TDA default | FreeTTA default |
| --- | ---: | ---: |
| DTD | 0.4601 | 0.4394 |
| Caltech | 0.9379 | 0.9355 |
| EuroSAT | 0.5958 | 0.4825 |
| Pets | 0.8896 | 0.8839 |
| ImageNetV2 | 0.6272 | 0.6235 |

After targeted tuning and recovery sweeps, the strongest saved FreeTTA results in this repository are approximately:

| Dataset | Best saved FreeTTA |
| --- | ---: |
| DTD | 0.4681 |
| Caltech | 0.9420 |
| EuroSAT | 0.5106 |
| Pets | 0.8842 |
| ImageNetV2 | 0.6235 |

So the local recovery story is mixed:

- `FreeTTA` can be recovered close to paper behavior on `DTD` and `Caltech`,
- but on `EuroSAT`, `Pets`, and `ImageNetV2`, it still does not match the paper-level advantage over `TDA`,
- which motivates the deeper mechanism analysis in the rest of the report.

## 8. Why Paper Expectation and Local Reproduction Can Differ

The most likely reasons are:

1. **Protocol mismatch**
   Our benchmark uses extracted CLIP features and a shared analysis pipeline, rather than reproducing every paper's original raw-image environment exactly.

2. **Implementation confidence mismatch**
   `TDA` has an official public implementation that can be checked directly.
   `FreeTTA` is more dependent here on paper-based reimplementation plus tuning.

3. **Different geometry in the current benchmark**
   Even if the method is theoretically stronger, it still depends on the actual frozen feature geometry seen in this repo.

4. **Low headroom on some datasets**
   On datasets where CLIP is already very strong, a theoretically better adaptive method may have little room to show gains.

5. **Mechanism mismatch**
   FreeTTA can update global statistics internally, but those updates only help if the final generative correction changes CLIP in the right direction strongly enough.

## 8.1 Shared-Order Analysis Results

From [outputs/comparative_analysis/summary_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/summary_metrics.csv), using one aligned stream order and the tuned per-dataset parameters already found in this repository:

| Dataset | CLIP | TDA | FreeTTA | FreeTTA - TDA |
| --- | ---: | ---: | ---: | ---: |
| Caltech | 0.9355 | 0.9359 | 0.9359 | +0.0000 |
| DTD | 0.4394 | 0.4516 | 0.4633 | +0.0117 |
| EuroSAT | 0.4843 | 0.5304 | 0.5106 | -0.0198 |
| ImageNet | 0.6235 | 0.6270 | 0.6235 | -0.0035 |
| Pets | 0.8839 | 0.8880 | 0.8842 | -0.0038 |

This table is the right one to use when discussing the deep per-sample mechanism analysis in this report, because all three methods are compared on the same stream order and the same sample alignment.

## 9. What We Implemented for Deep Analysis

The main new contribution of this project is a **comprehensive experimental analysis pipeline**.

Implementation:

- [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)
- [experiments/plot_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/plot_comparative_analysis.py)

For each dataset and each test sample, the pipeline stores:

- ground-truth label,
- CLIP logits,
- TDA logits,
- FreeTTA logits,
- predicted labels,
- correctness flags,
- confidence,
- entropy,
- stream position,
- method-specific internal statistics.

This exactly supports the kind of analysis requested in the project brief.

## 10. How We Analyze What the Methods Are Doing to CLIP

### 10.1 Prediction-change analysis

We explicitly measure:

- change rate,
- beneficial flips,
- harmful flips,
- net correction score.

Files:

- [outputs/comparative_analysis/flip_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_metrics.csv)
- [outputs/comparative_analysis/flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)
- per dataset: `prediction_change_analysis.png`

Why this matters:

- `TDA` and `FreeTTA` should not be judged only by final accuracy,
- they should be judged by **how safely and selectively they change CLIP**.

Interpretation:

- high beneficial flips = the method fixes CLIP mistakes,
- high harmful flips = the method overcorrects and breaks already-correct CLIP predictions,
- net correction tells whether the method is helping CLIP overall.

### 10.2 Entropy and confidence analysis

We track:

- entropy distributions for `CLIP`, `TDA`, `FreeTTA`,
- confidence distributions,
- separate statistics for correct and wrong predictions.

Files:

- [outputs/comparative_analysis/entropy_confidence_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/entropy_confidence_metrics.csv)
- [outputs/comparative_analysis/entropy_confidence_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/entropy_confidence_summary.png)
- per dataset: `entropy_confidence_analysis.png`

Why this matters:

- if a method lowers entropy and increases confidence **only on correct predictions**, adaptation is working well,
- if a method becomes highly confident on wrong predictions, it is creating **overconfident errors**.

### 10.3 Trajectory analysis

We compute rolling:

- accuracy,
- confidence,
- entropy.

Files:

- [outputs/comparative_analysis/latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)
- per dataset: `trajectory_analysis.png`

Why this matters:

- `TDA` should often help earlier because local memory can act immediately,
- `FreeTTA` may need more target evidence before its global statistics become useful.

## 11. FreeTTA Internal Statistics

We track:

- class mean drift `||mu_y(t) - mu_y(0)||`,
- prior entropy,
- covariance trace,
- EM weight,
- mean update norm.

Files:

- [outputs/comparative_analysis/internal_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/internal_metrics.csv)
- per dataset: `freetta_internal_analysis.png`

What this tells us:

- if mean drift stays near zero, FreeTTA is barely adapting,
- if mean drift becomes large but accuracy does not improve, the statistics are moving but not helping,
- prior entropy shows whether the class distribution is collapsing or remaining broad,
- covariance trace shows how much spread the model is estimating.

A very important mechanism-level point is:

> FreeTTA can update global statistics substantially without strongly changing final predictions if the generative correction is weak relative to the CLIP term.

That is exactly the kind of phenomenon the deep analysis pipeline can reveal.

## 12. TDA Internal Analysis

We track:

- positive cache size,
- negative cache size,
- negative gate activation rate,
- effective cache pressure.

Files:

- [outputs/comparative_analysis/internal_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/internal_metrics.csv)
- per dataset: `tda_internal_analysis.png`

What this tells us:

- whether TDA has enough useful local evidence in memory,
- whether the negative gate is actually active,
- whether the target stream is much larger than what TDA can retain.

This is the key structural advantage that FreeTTA is supposed to have:

- TDA stores only a limited memory,
- FreeTTA compresses evidence into global statistics.

## 13. Disagreement Analysis

We define:

`D = { i : p_tda(i) != p_freetta(i) }`

and compare:

- `Acc_tda(D)`
- `Acc_freetta(D)`

Files:

- [outputs/comparative_analysis/disagreement_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_metrics.csv)
- [outputs/comparative_analysis/disagreement_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_analysis.png)

This is one of the cleanest ways to compare the methods, because it focuses only on the samples where their inductive biases actually diverge.

## 14. Failure Buckets

The pipeline explicitly creates failure buckets such as:

1. CLIP wrong, TDA wrong, FreeTTA correct
2. CLIP wrong, TDA correct, FreeTTA wrong
3. CLIP correct, TDA wrong, FreeTTA correct
4. CLIP correct, TDA correct, FreeTTA wrong
5. all wrong

Files:

- [outputs/comparative_analysis/failure_bucket_summary.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/failure_bucket_summary.csv)
- [outputs/comparative_analysis/failure_bucket_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/failure_bucket_summary.png)
- per dataset: `failure_cases/<bucket>/contact_sheet.png`

This is especially useful in viva because it gives **actual visual examples** rather than only aggregate tables.

## 15. PCA Logit Visualization

One of the most important new analyses in this project is the PCA projection of the logit vectors:

- `Z_clip`
- `Z_tda`
- `Z_freetta`

For each sample, the analysis plots:

- the CLIP logit point,
- the TDA logit point,
- the FreeTTA logit point,
- and movement arrows from CLIP to each adapted method.

Files:

- per dataset: `pca_logit_visualization.png`
- per dataset: `pca_projection.csv`

Why this matters:

- it visually shows whether TDA and FreeTTA move logits in similar or different directions,
- it reveals whether a method makes large or small corrections,
- and it isolates special cases such as:
  - both methods agree,
  - TDA correct and FreeTTA wrong,
  - FreeTTA correct and TDA wrong,
  - both wrong.

## 16. Theoretical Conditions: When Each Method Should Succeed

### FreeTTA should be favored when:

- the target stream is long,
- class-level drift is coherent,
- global statistics are a good description of the shift,
- memory compression is important,
- late-stream gains matter more than immediate corrections.

### TDA should be favored when:

- useful information is highly local,
- classes are multi-modal or texture-heavy,
- a few strong local exemplars are enough,
- fast early corrections matter,
- global distribution drift is too noisy or unstable.

These are not absolute guarantees, but they are the clearest mechanism-level hypotheses supported by the method designs.

## 17. What to Look For in the Visuals

For viva and presentation, the most informative visual sequence is:

1. [outputs/presentation_figures/paper_vs_repo.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/paper_vs_repo.png)
   This gives the cleanest high-level paper-vs-repo comparison.

2. [outputs/comparative_analysis/flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)
   This shows whether each method changes CLIP safely.

3. [outputs/comparative_analysis/entropy_confidence_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/entropy_confidence_summary.png)
   This shows whether adaptation increases confidence in the right way.

4. [outputs/comparative_analysis/latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)
   This shows who helps earlier and who helps later.

5. per-dataset `freetta_internal_analysis.png` and `tda_internal_analysis.png`
   These explain the internal mechanism.

6. per-dataset `pca_logit_visualization.png`
   This gives the most intuitive picture of how logits move away from CLIP.

## 18. Concrete Observations From the Tuned Shared-Order Analysis

The most useful dataset-level observations from [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis) are:

### Caltech

- `CLIP`, `TDA`, and `FreeTTA` are all very close.
- `TDA` and `FreeTTA` both improve CLIP by only `+0.0004`.
- `TDA` changes more predictions than `FreeTTA`:
  roughly `0.81%` vs `0.12%`.
- `FreeTTA` shows large internal mean drift in the tuned shared-order run, but it still changes only a tiny number of predictions.

Interpretation:

- this is a low-headroom dataset,
- and it is a good example of how **internal statistic movement does not automatically translate into meaningful output correction**.

### DTD

- in the tuned shared-order analysis, `FreeTTA` beats `TDA`:
  `0.4633` vs `0.4516`.
- both methods change CLIP on roughly `12%` of samples,
- but `FreeTTA` has better beneficial-flip precision and better net correction.

Interpretation:

- this is the strongest case in the current shared-order analysis where `FreeTTA` clearly improves over `TDA`,
- so DTD should not be presented as a universal TDA win in this repository.

### EuroSAT

- both methods improve CLIP,
- but `TDA` remains stronger:
  `0.5304` vs `0.5106`.
- `TDA` changes many more predictions and gets a larger net correction.
- the geometry alignment score is negative here, meaning the local-neighbor probe is stronger than the centroid probe in this benchmark.

Interpretation:

- this is the main case where the paper expectation and the current benchmark disagree most,
- and it shows that **high stream length and cache pressure alone are not sufficient** if the feature geometry still favors local structure.

### Pets

- both methods are close to CLIP,
- `TDA` is slightly better than `FreeTTA`,
- both methods make only small corrections because CLIP is already strong.

Interpretation:

- this is another low-headroom regime,
- where small harmful flips can easily cancel small beneficial flips.

### ImageNet

- `TDA` gives a small gain,
- `FreeTTA` is effectively tied with CLIP.
- in the current shared-order analysis, `FreeTTA` barely changes predictions at all.

Interpretation:

- this is the clearest case where `FreeTTA` behaves like a near-no-op under the current benchmark,
- so the global statistic update is either too weak or too poorly matched to the feature geometry to create useful logit shifts.

These observations are exactly why the project needs more than a final accuracy table. They connect:

- what the method is changing,
- how much it is changing it,
- whether those changes are helpful,
- and how the internal state evolves while that happens.

## 19. Main Takeaway

The main conclusion of the project is:

> `TDA` and `FreeTTA` are not simply two methods where one always dominates the other. They exploit different structure in the same frozen CLIP space.

- `TDA` exploits **local target evidence** through cache retrieval.
- `FreeTTA` exploits **global target evidence** through online distribution modeling.

The papers suggest that FreeTTA should often be stronger overall, but not universally. Our local reproduction does not fully match that expected ordering. That mismatch becomes the most interesting part of the project, because it motivates a deeper comparison of:

- prediction changes,
- confidence and entropy,
- adaptation trajectories,
- global statistic drift,
- local cache behavior,
- disagreement regions,
- and visual failure cases.

So the real contribution of this project is:

> a mechanism-level analysis framework for understanding how `TDA` and `FreeTTA` differ beyond a single final accuracy table.

## 20. Contribution Statement

This project contains both reused and newly written components.

### Reused or paper-derived parts

- TDA logic was checked against the official public implementation.
- FreeTTA was implemented and tuned in this repository from the paper description and shared benchmark setup.

### New work in this repository

- unified comparison runners,
- per-sample metric collection,
- confidence and entropy analysis,
- prediction-flip analysis,
- trajectory analysis,
- internal-state tracking for both methods,
- disagreement analysis,
- failure bucket export with images,
- PCA logit movement visualization,
- report-ready figures and summaries.

That is the core project contribution for option 2.
