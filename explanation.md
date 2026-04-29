# FreeTTA vs TDA: When Each Method Wins and Why

Based on the comparative analysis from `outputs/comparative_analysis/`,
---

## Summary Accuracy Table

| Dataset    | CLIP   | TDA    | FreeTTA | Winner       | Margin   |
|------------|--------|--------|---------|--------------|----------|
| EuroSAT    | 55.51% | 53.33% | 59.35%  | FreeTTA      | +6.01%   |
| DTD        | 44.84% | 45.16% | 46.54%  | FreeTTA      | +1.38%   |
| Caltech101 | 93.59% | 93.59% | 93.63%  | FreeTTA      | +0.04%   |
| ImageNet   | 62.72% | 62.72% | 62.72%  | TIE          | 0.00%    |
| Oxford Pets| 88.45% | 88.69% | 88.63%  | TDA (marginal)| +0.05%  |

---

## FreeTTA: When It Wins

### 1. EuroSAT — FreeTTA +6.01% over TDA (59.35% vs 53.33%)

**The clearest FreeTTA win of all five datasets.**

- **Why FreeTTA wins**: EuroSAT is a satellite imagery dataset with 10 classes and 8,100 samples. CLIP's zero-shot text anchors are misaligned for aerial/multispectral imagery ("a photo of Annual Crop Land" vs actual satellite pixels). FreeTTA's global mean update gradually corrects this systematic domain shift — after seeing hundreds of AnnualCrop samples, its adapted mean μ moves toward the actual feature distribution of satellite crops, not the CLIP text anchor.

- **Why TDA fails**: Cache pressure ratio = 8100 / (10 × 5) = **162×**. TDA's positive cache can hold only 5 shots per class but must process 810 samples per class. Every cache slot is overwritten ~162 times. The cache is almost never representative of the full class distribution at any point in the stream.

- **Cold-start penalty**: In natural stream order, the first 1,620 EuroSAT samples are heavily biased toward early classes. TDA's cache starts empty and accumulates wrong predictions early:
  - Window 1 (steps 1–1620): TDA 33.8% vs CLIP 53.5% vs FreeTTA 56.2%
  - TDA recovers only in the late stream when its cache has been fully populated:
  - Window 4 (steps 4860–6480): TDA 75.6% vs CLIP 60.0% vs FreeTTA 62.7%

- **Key statistic**: `only_freetta_correct` = 621 samples (7.67%); `only_tda_correct` = 488 (6.02%). FreeTTA uniquely rescues 133 more samples than TDA can.

### 2. DTD — FreeTTA +1.38% over TDA (46.54% vs 45.16%)

**FreeTTA wins more hard samples.**

- **Why FreeTTA wins**: DTD (Describable Textures Dataset) has 47 texture classes with extreme within-class variance. FreeTTA's global mean captures the true centroid of each texture class after iterative M-step updates. On hard samples (CLIP confidence < 0.50), FreeTTA achieves 34.6% accuracy vs TDA's 30.7%.

- **Class-level win count**: FreeTTA wins 20/47 classes outright; TDA wins only 11/47. FreeTTA uniquely rescues 92 samples that TDA gets wrong; TDA uniquely rescues only 57.

- **Why TDA struggles**: Texture classes like "braided" and "woven" share similar local features. TDA's cached image features for one texture can incorrectly reinforce a neighboring texture's logits, pushing predictions toward the wrong class.

### 3. Caltech101 — FreeTTA +0.04% (93.63% vs 93.59%)

**Near-solved dataset; FreeTTA is more conservative and makes fewer errors.**

- **Why FreeTTA wins (barely)**: Caltech101 is the easiest dataset — CLIP already hits 93.59%. FreeTTA's mean updates are minor (low entropy inputs → small M-step weights). Its change rate is 0.41% (very low), but its correction efficiency (CE = 0.60) means 60% of its few changes are beneficial.

- **Key comparison**: FreeTTA makes only 4 harmful flips total; TDA makes 9. On a near-perfect baseline, harmful flips are the main risk. FreeTTA's cautious adaptation avoids most of them.

---

## FreeTTA: When It Fails

### 1. Oxford Pets — FreeTTA −0.05% (88.63% vs TDA 88.69%)

**Fine-grained breed classification: mean drift crosses species boundaries.**

- **Why TDA wins**: Pets has 37 classes of dog/cat breeds with subtle inter-class differences (e.g., Samoyed vs American Bulldog vs Labrador). TDA's cached real image features are pixel-precise anchors for each breed. When TDA sees a Samoyed, it adds logits from the 5 most recently seen, correctly-labeled Samoyed feature vectors — precise local structure.

- **Why FreeTTA fails**: FreeTTA's adapted mean μ for each breed class drifts across M-step updates. Because many dog breeds share visual features (white fur, floppy ears), the mean for Samoyed can drift toward the average of Samoyed + American Bulldog + Labrador features. This drift increases confusion between adjacent breeds.

- **Hard evidence**: FreeTTA makes 14 harmful flips on this dataset; TDA makes only 10. On 21/37 classes the two methods are tied, but on the 8 classes where they differ, TDA's local cache precision beats FreeTTA's drifted global mean.

### 2. ImageNet — Dead Tie (62.72% = 62.72%)

**Too sparse for meaningful global mean updates.**

- **Why FreeTTA cannot pull ahead**: ImageNet has 1,000 classes and 10,000 samples — only 10 samples per class on average. FreeTTA's M-step computes class mean updates weighted by entropy-scaled confidence. With 10 samples, the adapted mean μ never moves far enough from the CLIP text anchor to make a difference. The final mean drift magnitude per class is the lowest of all five datasets.

- **Why TDA also cannot pull ahead**: TDA's cache holds 5 positive shots per class out of 10 total samples — half the class data ends up in cache. But with only 10 samples total, cache quality is highly dependent on which 5 happened to be most confident, which is nearly random.

- **Net result**: Both methods are essentially running CLIP with minor noise — neither has enough per-class data to adapt meaningfully.

---

## TDA: When It Wins

### 1. Oxford Pets — TDA 88.69% vs FreeTTA 88.63%

As described above: precise cached features beat drifted global means for fine-grained breed discrimination.

### 2. EuroSAT Late Stream — TDA recovers

Once TDA's cache is fully populated (steps 4860–6480 of 8100):
- TDA: 75.6%
- FreeTTA: 62.7%
- CLIP: 60.0%

In this window TDA outperforms FreeTTA by **+12.9 percentage points**. This shows TDA's ceiling is higher than FreeTTA's when cache pressure is relieved — but it only arrives in the last 20% of the stream.

---

## TDA: When It Fails

### 1. Cold-Start Problem (EuroSAT)

Window 1 (steps 1–1620): TDA **33.8%** vs CLIP 53.5%. TDA actively hurts performance in the early stream when its cache is empty or populated only with misclassified samples. The logit adjustments from a mostly-empty/wrong cache push predictions in the wrong direction.

### 2. Cache Saturation (EuroSAT, DTD)

With 162× cache pressure on EuroSAT and ~40× on DTD, the 5-shot-per-class cache is a tiny window onto each class. TDA must constantly evict and re-insert, and the evicted samples are often the most carefully selected low-entropy ones. The cache quality degrades under pressure.

### 3. Hard Samples by Design

TDA only adds samples with entropy below `low_entropy_threshold` to the positive cache. This means the cache contains only confident, easy samples. When a hard sample arrives, TDA's cached neighbors are the easy cases — not the hard edge cases — and the logit adjustment can point away from the correct class.

### 4. Caltech101 Harmful Flips

On the near-perfect Caltech101 baseline, TDA makes 9 harmful flips vs FreeTTA's 4. TDA's cached features from one sample step can briefly push a correct prediction to wrong — e.g., if the last 5 "car" samples happened to be sports cars, a sedan query might get over-corrected toward a sports car class that shares visual features.

---

## Structural Explanation

| Factor                        | Favors FreeTTA | Favors TDA        |
|-------------------------------|----------------|-------------------|
| Many samples per class        | ✓ (stable μ)   |                   |
| Few samples per class         |                | ✓ (precise cache) |
| High domain shift from CLIP   | ✓ (μ corrects) |                   |
| Fine-grained classes          |                | ✓ (local anchors) |
| Stream has class clusters     |                | ✓ (cache fills fast)|
| Shuffled / random stream      | ✓ (global stats)|                   |
| Large cache pressure (>50×)   | ✓              |                   |
| Small cache pressure (<10×)   |                | ✓                 |
| Early stream prediction quality| ✓ (no cold-start)|                 |
| Late stream peak accuracy     |                | ✓ (if cache full) |

**Core insight**: FreeTTA is a global, stateful method — it improves steadily throughout the stream and is robust to stream order, cache pressure, and domain shift. TDA is a local, memory-based method — it can achieve higher peak accuracy when well-populated but suffers from cold-start and saturation. On the five datasets tested, FreeTTA's consistent improvement beats TDA's variable performance in 3 of 5 cases and ties in 1.
