# Comparative Analysis of TDA and FreeTTA for VLM Test-Time Adaptation

This repository now includes a complete, proposal-aligned experiment pipeline to compare:

- `TDA` (memory-based training-free adaptation)
- `FreeTTA` (online EM / distribution-modeling adaptation)

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

## Dataset Features Expected

Place feature files in `data/processed` using names:

- `<dataset>_image_features.npy`
- `<dataset>_text_features.npy`
- `<dataset>_labels.npy`

The loader is case-insensitive and supports aliases like `caltech101 -> caltech`, `pet -> pets`.

## Run the Full Comparison

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
