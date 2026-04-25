# Full-Dataset Reproduction Audit

This audit checks whether the gap to the paper numbers was caused by evaluating on only a subset of samples, and whether there were implementation issues in the current repo.

## 1. Full-dataset check

The current benchmark was already using the full stored evaluation split for each dataset:

| Dataset | Samples used in repo |
| --- | ---: |
| Caltech | 2465 |
| DTD | 1880 |
| EuroSAT | 8100 |
| Pets | 3669 |
| ImageNetV2 matched-frequency | 10000 |

So the reproduction gap was **not** caused by accidentally evaluating on a small subset.

## 2. Code issues found and fixed

### 2.1 FreeTTA covariance inverse mismatch

The FreeTTA supplementary describes a trace-regularized inverse for the covariance term. The repo implementation was using a simpler inverse of `Sigma + eps I` instead. The model was updated to use the paper-style regularized inverse in [models/FreeTTA.py](/home/herrys/projects/AIP-Final-project/models/FreeTTA.py).

### 2.2 Wrong output path in `run_best_freetta.py`

The script was writing to an absolute `"/tuning/..."` path by mistake. This is fixed in [experiments/run_best_freetta.py](/home/herrys/projects/AIP-Final-project/experiments/run_best_freetta.py).

### 2.3 Stale scaled-logit comparison path

The simple comparison runner was feeding `100 * cosine` logits into FreeTTA even though the FreeTTA paper defines the CLIP probabilities from cosine logits directly. This was fixed in [comparison.py](/home/herrys/projects/AIP-Final-project/comparison.py).

## 3. Paper-default full-dataset reproduction on `node1`

The following runs were executed on `node1` with CUDA and saved to:

- [paper_default_tda_full.json](/home/herrys/projects/AIP-Final-project/outputs/reproduction/paper_default_tda_full.json)
- [paper_default_freetta_full.json](/home/herrys/projects/AIP-Final-project/outputs/reproduction/paper_default_freetta_full.json)

### 3.1 TDA

| Dataset | Repo full-dataset TDA | Paper target | Gap |
| --- | ---: | ---: | ---: |
| Caltech | 93.79 | 94.24 | -0.45 |
| DTD | 46.01 | 47.40 | -1.39 |
| EuroSAT | 59.58 | 58.00 | +1.58 |
| Pets | 88.96 | 88.63 | +0.33 |
| ImageNetV2 matched-frequency | 62.72 | 64.67 | -1.95 |

### 3.2 FreeTTA

| Dataset | Repo full-dataset FreeTTA | Paper target | Gap |
| --- | ---: | ---: | ---: |
| Caltech | 93.55 | 94.63 | -1.08 |
| DTD | 43.94 | 46.96 | -3.02 |
| EuroSAT | 48.25 | 62.93 | -14.68 |
| Pets | 88.39 | 90.11 | -1.72 |
| ImageNetV2 matched-frequency | 62.35 | 64.92 | -2.57 |

## 4. Main conclusion

After switching to full-dataset runs and fixing the clear FreeTTA implementation mismatches, the remaining gap is **not** explained by sample count.

The main remaining explanation is a **protocol mismatch** between the current repo benchmark and the FreeTTA paper:

- the repo benchmark follows the `split_zhou` test protocol used throughout the TDA-style setup,
- the FreeTTA supplementary reports different dataset sizes for some benchmarks, most notably `EuroSAT = 5400`, while the current repo benchmark uses `8100`,
- therefore "paper-number reproduction" and "fair same-split comparison" are not the same experiment for every dataset.

## 5. Practical interpretation

- `TDA` is already close to its paper numbers on the current full-dataset benchmark.
- `FreeTTA` still does not reproduce its paper table on the current benchmark even after the implementation fixes.
- The largest unresolved gap is `EuroSAT`, and that is also the clearest protocol mismatch case.

## 6. What to do next if strict paper matching is required

1. Add a dedicated `FreeTTA paper protocol` evaluation path for datasets whose split differs from the current TDA benchmark.
2. Re-extract or slice features for those exact paper splits, especially `EuroSAT`.
3. Report two separate tables:
   - paper-protocol reproduction,
   - same-split fair comparison.
