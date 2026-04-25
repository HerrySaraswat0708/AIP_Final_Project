# Comparative Analysis of TDA and FreeTTA

This repository is organized for the course-project **option 2** narrative:

> compare a baseline paper and a later method for the same problem, explain why the later method should improve, and add new analysis beyond either paper.

The chosen pair is:

- `TDA` as the baseline method
- `FreeTTA` as the later method

Both operate on the same **frozen CLIP** feature space. The project compares not only final accuracy, but also how each method changes CLIP predictions over time.

## Repository Layout

The repo is intentionally reduced to the final submission structure:

- `models/`
  - [FreeTTA.py](/home/herrys/projects/AIP-Final-project/models/FreeTTA.py)
  - [TDA.py](/home/herrys/projects/AIP-Final-project/models/TDA.py)
- `experiments/`
  - [run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)
  - [plot_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/plot_comparative_analysis.py)
  - [plot_presentation_figures.py](/home/herrys/projects/AIP-Final-project/experiments/plot_presentation_figures.py)
  - [tune_freetta.py](/home/herrys/projects/AIP-Final-project/experiments/tune_freetta.py)
  - [tune_tda.py](/home/herrys/projects/AIP-Final-project/experiments/tune_tda.py)
- `src/`
  - kept as the data-loading and feature-processing support code
- `outputs/`
  - all generated tables, plots, tuning results, and reproduction audits

## What The Project Does

The project has three layers:

1. **Paper/default reproduction**
   - verify paper-style/default settings on the local benchmark
   - compare recovered numbers against the reported paper tables

2. **Per-dataset tuning**
   - tune `TDA` and `FreeTTA` independently on the frozen-feature benchmark
   - save best parameters under `outputs/tuning`

3. **Deep comparative analysis**
   - collect per-sample logits, predictions, confidence, entropy, and internal states
   - analyze when each method helps or hurts relative to CLIP
   - generate report-ready and presentation-ready plots

## Main Scripts

- [experiments/tune_tda.py](/home/herrys/projects/AIP-Final-project/experiments/tune_tda.py)
  - searches TDA hyperparameters and writes [outputs/tuning/best_tda_run_results.json](/home/herrys/projects/AIP-Final-project/outputs/tuning/best_tda_run_results.json)

- [experiments/tune_freetta.py](/home/herrys/projects/AIP-Final-project/experiments/tune_freetta.py)
  - searches FreeTTA hyperparameters and writes [outputs/tuning/best_freetta_run_results.json](/home/herrys/projects/AIP-Final-project/outputs/tuning/best_freetta_run_results.json)

- [experiments/run_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/run_comparative_analysis.py)
  - runs the aligned per-sample analysis and writes [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis)

- [experiments/plot_comparative_analysis.py](/home/herrys/projects/AIP-Final-project/experiments/plot_comparative_analysis.py)
  - turns the saved analysis CSVs into summary plots

- [experiments/plot_presentation_figures.py](/home/herrys/projects/AIP-Final-project/experiments/plot_presentation_figures.py)
  - creates slide-ready figures in [outputs/presentation_figures](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures)

## Output Folders

- [outputs/reproduction](/home/herrys/projects/AIP-Final-project/outputs/reproduction)
  - paper-default audit JSONs
  - targeted FreeTTA recovery sweeps
  - reproduction notes

- [outputs/tuning](/home/herrys/projects/AIP-Final-project/outputs/tuning)
  - best saved tuning results for `TDA` and `FreeTTA`

- [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis)
  - per-sample metrics
  - logits
  - internal-state metrics
  - disagreement, flip, entropy, latency, and trajectory plots
  - per-dataset reports and failure-case exports

- [outputs/presentation_figures](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures)
  - high-level figures for the PPT

## Reproduction Status

What is solid in this repo:

- `TDA` was checked against the official public implementation logic and is reproduced much more closely.
- `FreeTTA` was reimplemented from the paper and supplementary details, then tuned extensively on `node1` with CUDA.

What is **not** solid enough to claim:

- a clean reproduction of the full `FreeTTA` paper table on this benchmark
- a truthful statement that `FreeTTA` dominates `TDA` on all five datasets here

The strongest saved `FreeTTA` recoveries in this repo are:

- `DTD`: `0.4681`
- `Caltech`: `0.9420`
- `EuroSAT`: `0.5106`
- `Pets`: `0.8842`
- `ImageNetV2`: `0.6235`

The strongest saved `TDA` recoveries in this repo are:

- `DTD`: `0.4601`
- `Caltech`: `0.9396`
- `EuroSAT`: `0.5958`
- `Pets`: `0.8883`
- `ImageNetV2`: `0.6272`

So the honest summary is:

- `FreeTTA` is recovered well on `DTD` and `Caltech`
- `TDA` remains stronger on `EuroSAT`, `Pets`, and `ImageNetV2` in this benchmark
- the report uses that mismatch as the central analysis question instead of hiding it

## What The Comparative Analysis Measures

The main analysis is not just final accuracy. It studies:

- how often `TDA` and `FreeTTA` change CLIP predictions
- which changes are beneficial and which are harmful
- confidence and entropy evolution
- rolling accuracy over the target stream
- `FreeTTA` global statistic drift
- `TDA` cache growth and gate activity
- disagreement cases between `TDA` and `FreeTTA`
- failure buckets relative to CLIP
- PCA movement of logits from `CLIP -> TDA` and `CLIP -> FreeTTA`

The formal question list is in [queries.md](/home/herrys/projects/AIP-Final-project/queries.md). The full written interpretation is in [ASSIGNMENT_REPORT.md](/home/herrys/projects/AIP-Final-project/ASSIGNMENT_REPORT.md).

## Most Useful Files For Submission

- [ASSIGNMENT_REPORT.md](/home/herrys/projects/AIP-Final-project/ASSIGNMENT_REPORT.md)
- [COURSE_PROJECT_STATUS.md](/home/herrys/projects/AIP-Final-project/COURSE_PROJECT_STATUS.md)
- [PRESENTATION_OUTLINE.md](/home/herrys/projects/AIP-Final-project/PRESENTATION_OUTLINE.md)
- [queries.md](/home/herrys/projects/AIP-Final-project/queries.md)
- [outputs/comparative_analysis/analysis_summary.md](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/analysis_summary.md)
- [outputs/reproduction/reproduction_audit.md](/home/herrys/projects/AIP-Final-project/outputs/reproduction/reproduction_audit.md)

## Regeneration

Tune `TDA`:

```bash
python experiments/tune_tda.py
```

Tune `FreeTTA`:

```bash
python experiments/tune_freetta.py
```

Run the aligned comparative analysis:

```bash
python experiments/run_comparative_analysis.py --device cuda --output-dir outputs/comparative_analysis
```

Generate the analysis summary plots:

```bash
python experiments/plot_comparative_analysis.py --input-dir outputs/comparative_analysis
```

Generate presentation figures:

```bash
python experiments/plot_presentation_figures.py --input-dir outputs/comparative_analysis --output-dir outputs/presentation_figures
```
