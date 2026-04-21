# Comparative Analysis of TDA and FreeTTA for VLM Test-Time Adaptation

This repository now includes a complete, proposal-aligned experiment pipeline to compare:

- `TDA` (memory-based training-free adaptation)
- `FreeTTA` (online EM / distribution-modeling adaptation)

## Novelty Note

The project also includes a short novelty write-up for **EdgeFreeTTA** in [novelty/README.md](/home/herrys/projects/AIP-Final-project/novelty/README.md), describing a low-rank test-time adaptation direction where the pretrained backbone stays mostly frozen and only a compact adaptation module is updated under distribution shift.

## What Was Implemented

This repo now contains three levels of work:

1. `TDA` and `FreeTTA` evaluation code on the processed CLIP feature datasets already present in `data/processed`.
2. A new `EdgeFreeTTA` prototype in [models/EdgeFreeTTA.py](/home/herrys/projects/AIP-Final-project/models/EdgeFreeTTA.py), which adapts in a compact low-rank feature-space residual instead of updating a full model.
3. End-to-end experiment runners and plotting scripts for:
   - the original `TDA` vs `FreeTTA` comparative analysis,
   - full-suite comparison with `CLIP`, `TDA`, `FreeTTA`, and `EdgeFreeTTA`,
   - report-ready figures from the saved CSV outputs.

Main scripts:

- [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py): consolidated `TDA` vs `FreeTTA` analysis.
- [experiments/plot_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/plot_comparative_analysis.py): creates graphs from `outputs/comparative_analysis`.
- [experiments/run_final_method_suite.py](/home/herrys/projects/AIP-Final-project/experiments/run_final_method_suite.py): full comparison across `CLIP`, `TDA`, `FreeTTA`, `EdgeFreeTTA`.
- [experiments/run_edgefreetta_comparison.py](/home/herrys/projects/AIP-Final-project/experiments/run_edgefreetta_comparison.py): simpler summary-table runner including `EdgeFreeTTA`.

## What the pipeline produces

For each dataset, it generates:

- Top-1 final accuracy for both methods
- Adaptation dynamics (running accuracy over sequential test samples)
- Uncertainty analysis (entropy vs correctness, entropy-bin accuracy)
- Distribution modeling visualization (PCA: initial vs adapted class means for FreeTTA)
- Computational efficiency (time/sample and adapter memory usage)
- Custom novel comparisons (not standard in TTA papers):
  - Disagreement Advantage: who wins when TDA and FreeTTA disagree
  - Prediction Churn Rate: temporal class-switch instability
  - Error Recovery Latency: average number of samples needed to recover after mistakes
  - Entropy-Conditioned Gap: FreeTTA minus TDA in easy/hard entropy bins
  - Calibration Error (ECE): confidence-quality alignment
- Architecture/loss/internal-workings analysis:
  - per-method internal signal summary with correlation to correctness
  - auto-generated mechanistic interpretation markdown per dataset

It also creates an overall summary table across datasets.

## How It Was Done

The pipeline assumes CLIP image features, text features, and labels are already extracted into `data/processed`. All methods then operate on the same frozen representation space:

- `CLIP` is the baseline classifier from image-text similarity.
- `TDA` adds positive and negative online caches for retrieval-style logit correction.
- `FreeTTA` performs online EM-style class-statistic updates in the same frozen space.
- `EdgeFreeTTA` keeps the backbone frozen and updates only a small low-rank residual adapter in feature space.

That makes the comparison fair at the representation level. The methods differ in *how* they exploit the same CLIP geometry:

- `TDA`: local memory, nearest-support style correction.
- `FreeTTA`: global class-distribution modeling.
- `EdgeFreeTTA`: compact parametric correction in a low-dimensional update space.

## Actual Outputs In This Repo

### Comparative analysis outputs

Located in [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis):

- [summary_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/summary_metrics.csv)
- [geometry_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/geometry_metrics.csv)
- [flip_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_metrics.csv)
- [disagreement_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_metrics.csv)
- [latency_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_metrics.csv)
- [difficulty_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/difficulty_metrics.csv)
- [per_sample_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/per_sample_metrics.csv)

Generated plots:

- [accuracy_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/accuracy_summary.png)
- [gain_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/gain_summary.png)
- [geometry_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/geometry_analysis.png)
- [flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)
- [disagreement_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_analysis.png)
- [latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)
- [difficulty_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/difficulty_analysis.png)

### Final full-suite outputs

Located in [outputs/final_method_suite](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite):

- summary tables for `CLIP`, `TDA`, `FreeTTA`, `EdgeFreeTTA`,
- overall accuracy plot,
- per-dataset adaptation dynamics plots,
- per-dataset `TDA` vs `FreeTTA` presentation/report plots.

## Final Tuned Full-Dataset Results

From [outputs/final_method_suite/summary_table.csv](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite/summary_table.csv), after the independent per-dataset tuning runs on `node1` with `CUDA`:

| Dataset | CLIP | TDA | FreeTTA | EdgeFreeTTA |
| --- | ---: | ---: | ---: | ---: |
| Caltech | `0.9355` | `0.9404` | `0.9359` | `0.9355` |
| DTD | `0.4394` | `0.4601` | `0.4644` | `0.4399` |
| EuroSAT | `0.4843` | `0.5958` | `0.5069` | `0.4843` |
| ImageNet | `0.6237` | `0.6272` | `0.6237` | `0.6235` |
| Pets | `0.8839` | `0.8863` | `0.8839` | `0.8839` |

What these final tuned runs show:

- `TDA` beats `CLIP` on all five datasets in the final suite.
- `FreeTTA` beats `CLIP` on `Caltech`, `DTD`, and `EuroSAT`, ties on `ImageNet` and `Pets`.
- `FreeTTA` beats `TDA` only on `DTD` in the final tuned table.
- `TDA` is the strongest overall method in this repository's reproduced benchmark.

## Comparative Findings

### Overall

From [outputs/comparative_analysis/summary_metrics.csv](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/summary_metrics.csv) and the final tuned full-suite reruns:

- the comparative-analysis folder captures the mechanism-level behavior of `TDA` and `FreeTTA`,
- the final-method-suite folder captures the tuned, report-facing accuracy table across all methods,
- after tuning, `TDA` is the strongest and most consistent adaptive method in this repo,
- `FreeTTA` is competitive on some datasets, but it does not consistently surpass `TDA` in this reproduction.

## What Comparisons Were Done Between TDA and FreeTTA

The TDA vs FreeTTA study in this repo is not just a final accuracy table. It compares the two methods along six separate axes:

1. **Final accuracy and gain over CLIP**
   This is the standard comparison: which method ends with the higher top-1 accuracy, and by how much each improves over the frozen CLIP baseline.

2. **Geometry comparison**
   This asks whether the frozen CLIP feature space looks more favorable to:
   `TDA`-style local exemplar retrieval, or `FreeTTA`-style global class-center modeling.

3. **Prediction-flip analysis**
   This measures whether each method's changes to CLIP predictions are mostly helpful corrections or harmful overcorrections.

4. **Disagreement analysis**
   This isolates samples where `TDA` and `FreeTTA` predict different labels and asks which method is more reliable exactly in those contested cases.

5. **Adaptation-latency analysis**
   This measures how quickly each method starts helping, rather than only how well it finishes.

6. **Difficulty and uncertainty analysis**
   This compares the methods across easy, medium, and hard samples, typically based on CLIP entropy or confidence.

These are implemented through:

- [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)
- [experiments/plot_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/plot_comparative_analysis.py)
- [queries.md](/home/herrys/projects/AIP-Final-project/queries.md)

## What Is New Beyond Both Papers

The new part of this repo is not only running `TDA` and `FreeTTA`, but building a common comparison framework around them. The following analyses were added here and were not presented as a unified evaluation package in either original paper:

- **Geometry Alignment Score**
  This compares centroid-style structure against local-neighbor structure in the same CLIP space.

- **Beneficial Flip Precision**
  This asks how often a method's changed prediction is a true correction rather than just a change.

- **Harmful Flip Rate**
  This measures how often the method damages predictions that CLIP already had correct.

- **Break-Even Latency**
  This measures how many stream samples are needed before adaptation becomes net-positive.

- **Cache Pressure**
  This connects target-stream length to TDA's finite cache capacity, which is important for understanding when exemplar memory can become a bottleneck.

- **Disagreement Accuracy**
  This evaluates which method should be trusted when the two adaptive methods disagree.

- **Internal-signal diagnostics**
  This relates accuracy behavior back to the internal state each method maintains, such as TDA cache usage or FreeTTA distribution drift.

The point of these additions is to turn the comparison from:

- "which method got the bigger final number?"

into:

- "why did this method win on this dataset, and what property of the data or adaptation mechanism explains it?"

## What Each Metric Means and How To Interpret It

The formal definitions are in [queries.md](/home/herrys/projects/AIP-Final-project/queries.md). The practical reading is:

- **Final Accuracy / Gain over CLIP**
  Higher is better. This tells you whether adaptation is useful at all relative to doing nothing beyond frozen CLIP.

- **Geometry Alignment Score (GAS)**
  Positive `GAS` means the dataset looks more centroid-friendly, which is more favorable to `FreeTTA`.
  Negative `GAS` means local neighbors are more informative, which is more favorable to `TDA`.

- **Beneficial Flip Precision (BFP)**
  High `BFP` means a method changes CLIP predictions selectively and intelligently.
  A low value means the method is changing predictions often without enough payoff.

- **Harmful Flip Rate (HFR)**
  Low is better.
  A high `HFR` means the method is overcorrecting and breaking already-correct CLIP predictions.

- **Break-Even Latency (BEL)**
  Low is better when fast online benefit matters.
  A high `BEL` means the method needs a longer stream before its adaptation state becomes useful.

- **Cache Pressure**
  High cache pressure means `TDA` is trying to summarize a long target stream with limited memory.
  That is where `FreeTTA` can have a structural advantage, because it stores class statistics rather than sparse exemplars.

- **Disagreement Accuracy**
  This is one of the most informative metrics in the repo.
  It answers: when `TDA` and `FreeTTA` disagree, which one is actually right more often?

- **Difficulty / Entropy-conditioned gap**
  This shows whether a method helps only on easy samples, or whether it also gives real value on uncertain and hard cases.

- **Internal drift and cache metrics**
  These help interpret failure modes.
  For `TDA`, they reveal whether retrieval support is strong enough.
  For `FreeTTA`, they reveal whether the distribution estimate is stable or drifting too far.

### Method-Level Advantages and Disadvantages

#### CLIP

Advantages:

- zero adaptation cost,
- stable baseline,
- strong when target shift is mild or adaptation is noisy.

Disadvantages:

- cannot exploit target-stream evidence,
- leaves performance on the table when online structure is informative.

#### TDA

Advantages:

- fast online gains,
- strong when local exemplar neighborhoods are informative,
- works well when a few confident samples can seed useful retrieval corrections.

Disadvantages:

- bounded memory,
- can become unstable under heavy cache pressure,
- depends on local neighborhood quality and entropy gating.

#### FreeTTA

Advantages:

- principled global distribution modeling,
- can benefit from long coherent streams,
- more compact state than exemplar memory when sufficient statistics are a good abstraction.

Disadvantages:

- slower to become useful,
- sensitive to mismatch between class geometry and Gaussian-like assumptions,
- in this repo, weaker reproduction than the paper-level expectation on most datasets.

#### EdgeFreeTTA

Advantages:

- adaptation stays in a compact low-rank space,
- cheap state compared with full-model updating,
- easy to keep the pretrained backbone frozen.

Disadvantages:

- currently a prototype on frozen feature tensors rather than a full end-to-end backbone adaptation setup,
- in the present runs it mostly matches `CLIP` or `FreeTTA` rather than exceeding `TDA`.

## Dataset-Wise Interpretation

### Caltech

- `CLIP` is already very strong.
- `TDA` is best in the tuned full-suite run.
- `FreeTTA` is only a marginal gain over `CLIP`.
- Geometry is centroid-friendly, which is consistent with `FreeTTA` being competitive.
- Adaptation headroom is small, so differences are minor.

### DTD

- both adaptive methods improve clearly over `CLIP`.
- `FreeTTA` slightly beats `TDA` in the tuned final run.
- Texture classes appear to benefit more from local exemplar memory than from a single global class-statistic model.
- This is the dataset where `FreeTTA` tuning helped the most relative to its untuned behavior.

### EuroSAT

- `TDA` is the best method in this reproduction by a large margin.
- `FreeTTA` improves over `CLIP`, but not enough to beat `TDA`.
- The geometry probe here is actually more favorable to local structure than to centroid-style modeling in these extracted features.
- This is the strongest dataset for showing adaptation gains overall.

### ImageNet

- Gains are small for both adaptive methods.
- `TDA` gives a small but consistent lift over `CLIP`.
- `FreeTTA` is tied with `CLIP` in the tuned final run.
- High centroid advantage in the geometry probe does not translate into large practical `FreeTTA` gains here, likely because CLIP is already strong and the stream is diverse.

### Pets

- `CLIP` is already high-performing.
- `TDA` gives a small gain over `CLIP`.
- `FreeTTA` is tied with `CLIP` in the tuned final run.
- This is a low-headroom dataset where adaptation errors can easily outweigh benefits.

## Why The Results Differ By Dataset

The main explanation is that the same frozen CLIP space supports different kinds of structure:

- datasets with useful local neighborhoods favor `TDA`,
- datasets with coherent global class statistics can favor `FreeTTA`,
- datasets with little headroom favor simply staying close to `CLIP`,
- low-rank correction ideas like `EdgeFreeTTA` become more useful when a compact parametric residual is enough to absorb shift.

In this repository's current extracted-feature setup, `TDA` is the most consistently strong adaptive method.

## How To Regenerate Everything

Comparative-analysis CSVs:

```bash
python experiments/run_comparative_analysis.py --output-dir outputs/comparative_analysis
```

Comparative-analysis plots:

```bash
python experiments/plot_comparative_analysis.py --input-dir outputs/comparative_analysis
```

Final full-suite outputs:

```bash
python experiments/run_final_method_suite.py --device cuda --stream-seed 1
```

## Dataset Features Expected

Place feature files in `data/processed` using names:

- `<dataset>_image_features.npy`
- `<dataset>_text_features.npy`
- `<dataset>_labels.npy`

The loader is case-insensitive and supports aliases like `caltech101 -> caltech`, `pet -> pets`.

## Old Proposal Commands

The original proposal-oriented commands below are kept for reference. Some paths still mention `proposal_comparison`; the actual finalized scripts used in this repo are the ones listed above under "How To Regenerate Everything".

```bash
python experiments/run_project_comparison.py \
  --features-dir data/processed \
  --output-dir outputs/proposal_comparison \
  --datasets dtd caltech eurosat pets imagenet
```

`imagenet` is optional and will be skipped automatically if feature files are not present.

## (Optional) Extract ImageNet Features

This repo maps `imagenet` to the public `ImageNetV2 matched-frequency` evaluation set, because the official ILSVRC validation set is gated. The extractor will download and cache it under `data/raw/IMAGENET` automatically:

```bash
python -m src.imagenet_extractor
```

## Quick Debug Run

```bash
python experiments/run_project_comparison.py --max-samples 500
```

## Main Outputs

- `outputs/proposal_comparison/summary_metrics.csv`
- `outputs/proposal_comparison/run_report.json`
- `outputs/proposal_comparison/<dataset>/dataset_report.json`
- `outputs/proposal_comparison/<dataset>/adaptation_dynamics.csv`
- `outputs/proposal_comparison/<dataset>/adaptation_dynamics.png`
- `outputs/proposal_comparison/<dataset>/entropy_analysis.png`
- `outputs/proposal_comparison/<dataset>/distribution_modeling_pca.png`
- `outputs/proposal_comparison/<dataset>/efficiency_summary.png`
- `outputs/proposal_comparison/<dataset>/custom_comparisons.png`
- `outputs/proposal_comparison/<dataset>/custom_entropy_conditioned_gains.csv`
- `outputs/proposal_comparison/<dataset>/internal_metric_summary.csv`
- `outputs/proposal_comparison/<dataset>/architecture_internal_analysis.md`
