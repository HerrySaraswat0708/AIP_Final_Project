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
It also recognizes `imagenet64`, `imagenet64_val`, and `imagenet_64eval -> imagenet64`.

## Run the Full Comparison

```bash
python experiments/run_project_comparison.py \
  --features-dir data/processed \
  --output-dir outputs/proposal_comparison \
  --datasets dtd caltech eurosat pets imagenet
```

`imagenet` is optional and will be skipped automatically if feature files are not present.

You can also run the downsampled ImageNet64 validation dump as a separate dataset:

```bash
python comparison.py --datasets imagenet64 --paper-backbone off
```

## (Optional) Extract ImageNet Features

If you have the official ImageNet validation set at `data/raw/IMAGENET`:

```bash
python src/imagenet_extractor.py --root data/raw/IMAGENET --batch-size 32
```

The extractor supports either:

- a `torchvision`-style ImageNet root with `meta.bin` and `val/`, or
- a folder layout like `images/val/<wnid>/*.JPEG` plus `classnames.txt`

## (Optional) Extract ImageNet64 Features

If you have a downsampled validation dump such as `data/raw/Imagenet64_val/val_data`:

```bash
python src/imagenet64_extractor.py --root data/raw/Imagenet64_val
```

If you already extracted the image features once and only need to refresh the label prompts:

```bash
python src/imagenet64_extractor.py \
  --root data/raw/Imagenet64_val \
  --skip-image-features
```

For the best label mapping, place `map_clsloc.txt` next to `val_data`.
The repo now prefers that file automatically when building ImageNet64 text prompts.

`imagenet64` is included as a convenience dataset for the downsampled dump, but it is not the same benchmark as the paper's full-resolution `ImageNet-1K` validation set.

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
