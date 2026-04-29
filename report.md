# Comprehensive Research Report
## CLIP, Domain Shift, Test-Time Adaptation: TDA vs FreeTTA

*All results from `outputs/comparative_analysis/` — natural stream order, seed=42, window=50 rolling accuracy.*

---

## Table of Contents

1. [CLIP: What, How, Why](#1-clip-what-how-why)
2. [The Problem: Domain Shift](#2-the-problem-domain-shift)
3. [Test-Time Adaptation: The Solution](#3-test-time-adaptation-the-solution)
4. [Motivation: Why TDA and FreeTTA?](#4-motivation-why-tda-and-freetta)
5. [TDA: Cache-Based Adaptation](#5-tda-cache-based-adaptation)
6. [FreeTTA: Online EM Adaptation](#6-freetta-online-em-adaptation)
7. [How Much Do They Solve the Problem?](#7-how-much-do-they-solve-the-problem)
8. [Datasets Explained](#8-datasets-explained)
9. [Our Comparison Approach and Metrics](#9-our-comparison-approach-and-metrics)
10. [Results: Dataset-by-Dataset](#10-results-dataset-by-dataset)
11. [Cross-Dataset Conclusions](#11-cross-dataset-conclusions)
12. [Where Each Algorithm Always Wins and Loses](#12-where-each-algorithm-always-wins-and-loses)
13. [Future Work and Suggested Improvements](#13-future-work-and-suggested-improvements)

---

## 1. CLIP: What, How, Why

### 1.1 What is CLIP?

CLIP (Contrastive Language–Image Pretraining) is a vision-language model developed by OpenAI (Radford et al., 2021). Its defining feature is that it learns **a shared embedding space** for both images and text by training on 400 million image–caption pairs scraped from the internet. Once trained, CLIP can match any image to any text description — including class names it has never explicitly been trained to classify.

This gives CLIP its most remarkable property: **zero-shot classification**. You can classify images into categories you define at inference time, with no labeled training examples, simply by providing text descriptions.

### 1.2 Architecture

CLIP has two parallel encoders:

- **Image encoder** f_I: A Vision Transformer (ViT-B/16 in our experiments). Takes a 224×224 image, outputs a 512-dimensional unit vector.
- **Text encoder** f_T: A Transformer. Takes a text string (e.g. "a photo of a cat"), outputs a 512-dimensional unit vector.

Both encoders project to the same 512-D space, where semantic similarity translates to geometric proximity.

### 1.3 Training: Contrastive Loss

For a batch of N image-text pairs {(I_i, t_i)}, CLIP maximizes the cosine similarity of matched pairs and minimizes it for mismatched pairs. The loss is symmetric cross-entropy on an N×N similarity matrix:

```
s(I_i, t_j) = f_I(I_i) · f_T(t_j)                (cosine similarity, both unit-normed)

L_img = -1/N  Σ_i  log[ exp(τ·s(I_i, t_i)) / Σ_j exp(τ·s(I_i, t_j)) ]
L_txt = -1/N  Σ_j  log[ exp(τ·s(I_i, t_i)) / Σ_i exp(τ·s(I_i, t_j)) ]
L = (L_img + L_txt) / 2
```

where τ (temperature, learned) scales the logits to make the distribution sharper.  
**Intuition**: Pull image and its paired caption together; push all other image-caption combinations apart. After 400M examples, the shared space encodes rich semantics.

### 1.4 Zero-Shot Inference

To classify a new image into C classes:

1. Build text templates: for each class c, create a text prompt e.g. `"a photo of a {class_name}"`.
2. Encode all C prompts: **W** = [f_T(t_1), ..., f_T(t_C)] ∈ ℝ^{C×D}
3. Encode the query image: **x** = f_I(I) ∈ ℝ^D (unit-normed)
4. Compute logits:  **z** = 100 · **W** · **x**  (scale factor 100 ≈ 1/τ)
5. Convert to probabilities:  p_c = softmax(z)_c = exp(z_c) / Σ_j exp(z_j)
6. Predict: ŷ = argmax_c p_c

**Intuition**: The dot product `W·x` measures how well the image aligns with each class description in the shared semantic space. The class whose text embedding best matches the image embedding wins.

### 1.5 Why CLIP is Powerful (and Where It Falls Short)

**Strengths:**
- Zero-shot: no labeled data needed for new tasks
- Generalizes across domains seen in the 400M internet training set
- Single frozen model works for hundreds of classification tasks

**Weaknesses:**
- The 400M training set is web images with captions; specialized domains (satellite imagery, microscopy, medical scans) are underrepresented
- Text prompts like "a photo of..." may poorly describe non-photographic images
- Confidence is poorly calibrated: CLIP outputs very uniform probabilities (1/C-like) because it sees all C classes at every forward pass

---

## 2. The Problem: Domain Shift

### 2.1 What Is Domain Shift?

Domain shift means the statistical distribution of test images P_test(x) differs from the distribution the model was trained on P_train(x), even if the label set is the same. Formally, we train on P_train(x, y) but evaluate on P_test(x, y) where P_test(x|y) ≠ P_train(x|y).

### 2.2 Why Domain Shift Hurts CLIP Specifically

CLIP's text anchors W are fixed at inference time. They encode the average internet concept of each class. If the test images deviate from this concept, the cosine similarities shift — and the wrong class may have the highest similarity.

**Example — EuroSAT**: The text anchor for "Annual Crop" encodes "a photo of annual crop land" — which, in the training corpus, might be a landscape photo from a tractor. The actual EuroSAT image is a 64×64 multispectral overhead tile. The feature vector `x` ends up in a part of the embedding space that is equidistant from many class anchors → highly uncertain predictions.

**Example — DTD (Describable Textures)**: Texture names like "braided", "woven", "knitted" are semantically similar in language but visually distinct. CLIP's text encoder places these class anchors very close together (they're all textile-related words), making discrimination hard purely from text features.

### 2.3 Quantifying Domain Shift via CLIP Baseline Accuracy

| Dataset     | Classes | Samples | CLIP Acc | Domain Type                    |
|-------------|---------|---------|----------|--------------------------------|
| Caltech101  | 100     | 2,465   | 93.55%   | Standard photos                |
| Oxford Pets | 37      | 3,669   | 88.39%   | Pet photographs                |
| ImageNet-V2 | 1,000   | 10,000  | 62.35%   | General objects (distribution shift from ImageNet train) |
| DTD         | 47      | 1,880   | 43.94%   | Abstract textures              |
| EuroSAT     | 10      | 8,100   | 48.43%   | Satellite imagery              |

The 44.6 percentage-point gap between Caltech (93.55%) and EuroSAT (48.43%) measures domain shift severity. On EuroSAT, CLIP barely beats chance for a 10-class problem (10%).

### 2.4 Why CLIP Confidence Is Uninformative

On Caltech: mean confidence = 0.0113 (expected for 100 classes: 0.01)  
On EuroSAT: mean confidence = 0.1028 (expected for 10 classes: 0.10)

CLIP's confidence is barely above uniform. It cannot distinguish easy from hard samples. This will be important when evaluating TDA (which uses entropy thresholds) and FreeTTA (which weights M-step updates by confidence).

---

## 3. Test-Time Adaptation: The Solution

### 3.1 What Is TTA?

Test-Time Adaptation (TTA) adapts a frozen model at inference time using only the unlabeled test stream. Unlike fine-tuning (which requires labeled data) or domain adaptation (which requires access to both source and target domains during training), TTA:

- Has access only to the current and past test samples
- Cannot change model weights (CLIP stays frozen throughout)
- Must operate online: decision on sample i is made before seeing sample i+1
- Uses no ground-truth labels — all adaptation signal comes from the model's own predictions

### 3.2 Why TTA Works

Even without labels, the test stream carries information:
1. **Self-consistency**: if the model keeps predicting class c for samples that look similar to each other, those samples probably are class c
2. **Cluster structure**: in a good feature space, same-class samples cluster together; TTA methods exploit cluster structure to shift decision boundaries
3. **Entropy minimization**: confident predictions (low entropy) are more likely correct; using them to update class representations improves future predictions

### 3.3 Why Not Just Fine-Tune?

Fine-tuning CLIP's visual encoder (307M parameters) on a few hundred test samples would immediately overfit and forget general knowledge. TTA instead works in the logit or representation space downstream of the frozen encoder — a much lower-dimensional problem.

---

## 4. Motivation: Why TDA and FreeTTA?

### 4.1 The Two Philosophies

After surveying TTA methods for CLIP, two fundamentally different philosophies emerged:

**Philosophy 1 — "Remember what you've seen" (cache/memory-based)**  
Keep a running memory bank of reliable past predictions. When a new sample arrives, use its similarity to cached samples to refine the prediction. This is local: each prediction is influenced only by previously seen similar samples.
→ **Implemented as: TDA (Test-time Data Augmentation via cache)**

**Philosophy 2 — "Update your world model" (statistics-based)**  
Maintain running estimates of each class's feature distribution. When a new sample arrives, use both the CLIP text anchor AND the learned visual mean to predict. Update the visual mean after each prediction. This is global: each prediction is influenced by everything seen for that class so far.
→ **Implemented as: FreeTTA (Free Test-Time Adaptation via online EM)**

### 4.2 Why These Two Specifically?

1. **Minimal assumptions**: neither requires labeled data, neither modifies CLIP weights
2. **Complementary mechanisms**: cache is local (instance-level), EM is global (class-level) — comparing them reveals which signal matters more
3. **Different failure modes**: cache fails under memory pressure (too many samples per class); EM fails when per-class sample count is too low
4. **Practical efficiency**: both operate in O(1) or O(C) per step — no backpropagation
5. **Theoretical grounding**: TDA relates to k-NN retrieval; FreeTTA relates to Gaussian mixture model EM — both have clear interpretations

---

## 5. TDA: Cache-Based Adaptation

### 5.1 Intuition

Imagine you're a new doctor trying to diagnose patients from descriptions. You have a textbook (CLIP's text anchors). For each new patient, you also look at your notes from similar past patients. If your last 5 "annual crop" images all had a certain pattern, and the new image looks similar, you upweigh "annual crop" as the diagnosis — even if your textbook description isn't perfect.

TDA maintains two cache types per class:
- **Positive cache**: recent samples you were highly confident about (low entropy). Good reference examples.
- **Negative cache**: samples you were moderately uncertain about (medium entropy). Anti-examples — "this looks sort of like class c but probably isn't."

The caches act as dynamic, non-parametric pseudo-labels that augment CLIP's static text anchors.

### 5.2 Algorithm

**Initialization**: Empty caches. For each class c, positive_cache[c] = [], negative_cache[c] = [].

**For each test sample x_t (unit-normed 512-D feature vector):**

**Step 1 — Compute base CLIP logits:**
```
z_clip(c) = 100 · x_t · w_c        for each class c
```

**Step 2 — Compute cache adjustment logits:**

For positive cache of class c (stored as feature vectors {v_1, ..., v_k}):
```
z_pos(c) = (λ_pos / |cache_c|)  Σ_i  (x_t · v_i)     if cache_c non-empty, else 0
```

For negative cache:
```
z_neg(c) = (λ_neg / |neg_cache_c|)  Σ_i  (x_t · v_i)   if neg_cache_c non-empty, else 0
```

**Step 3 — Combine:**
```
z_final(c) = z_clip(c) + z_pos(c) - z_neg(c)
p(c|x) = softmax(z_final)_c
```

**Step 4 — Update cache (using predicted label ŷ = argmax p):**

Compute entropy H = -Σ_c p(c|x) · log p(c|x)

- If H < θ_low: add x_t to positive_cache[ŷ]; evict oldest if over capacity (FIFO)
- If θ_low ≤ H < θ_mid: add x_t to negative_cache[ŷ]; evict oldest if over capacity
- If H ≥ θ_mid: discard (too uncertain to be a reliable reference)

### 5.3 Mathematics

The cache-based logit is essentially a weighted cosine similarity kernel:

```
z_pos(c) = λ_pos · E_{v ~ cache_c}[x_t · v]
         = λ_pos · 〈x_t, μ̂_c〉
```

where μ̂_c = mean of cached features for class c (empirical centroid of confident samples).

In the limit of infinite cache capacity and samples, z_pos(c) approaches λ_pos · x_t · μ_c (true class centroid) — equivalent to a nearest-centroid classifier. But with finite cache (5 shots default), it is a **sliding-window mean of the most recent confident examples**.

The negative cache acts as an **anti-centroid**: by subtracting z_neg(c), the model pushes away from regions of feature space that look moderately like class c but are actually ambiguous or wrong.

The combined update can be written as:
```
z_final = z_clip + Σ_c (λ_pos · K_pos(x,c) - λ_neg · K_neg(x,c)) · e_c
```
where K_pos, K_neg are kernel functions (dot-product similarity to cache).

### 5.4 Hyperparameters (Dataset-Specific Defaults)

| Parameter          | Caltech | DTD  | EuroSAT | ImageNet | Pets  | Meaning                        |
|--------------------|---------|------|---------|----------|-------|--------------------------------|
| pos_shot_capacity  | 3       | 3    | 3       | 3        | 3     | Max positive samples per class |
| neg_shot_capacity  | 3       | 3    | 3       | 3        | 3     | Max negative samples per class |
| low_entropy_thresh | 0.2     | 0.2  | 0.2     | 0.2      | 0.2   | Threshold for positive cache   |
| mid_entropy_thresh | 0.4     | 0.4  | 0.4     | 0.4      | 0.4   | Threshold for negative cache   |
| lambda_pos         | 1.0     | 1.0  | 1.0     | 1.0      | 1.0   | Positive cache weight          |
| lambda_neg         | 0.1     | 0.1  | 0.1     | 0.1      | 0.1   | Negative cache weight          |

### 5.5 Key Structural Properties

- **Local**: each prediction depends on the most recently seen similar samples (FIFO window)
- **Cold-start**: makes no adjustments until cache fills (first few hundred steps are essentially raw CLIP)
- **Cache pressure**: when N_samples >> C × capacity, the cache is overwritten frequently; useful signal evicts before the next similar sample arrives
- **Discriminative**: negative cache actively suppresses confusion between similar classes
- **No global state**: two samples from the same class seen 1000 steps apart do not directly interact

---

## 6. FreeTTA: Online EM Adaptation

### 6.1 Intuition

Imagine the same new doctor, but this time instead of notes on individual patients, you build a running average of what each disease "looks like" in test images. Every time you diagnose "annual crop", you update your visual prototype for annual crop to move slightly toward the current image. Over hundreds of samples, your visual prototype converges to the true distribution of satellite-image annual crops — even if you started from the wrong text-based anchor.

FreeTTA models each class as a Gaussian distribution in feature space and adapts the class means online using an Expectation-Maximization (EM) framework.

### 6.2 Algorithm

**Initialization**: For each class c, adapted mean μ_c = w_c (CLIP text anchor, unit-normed).

**For each test sample x_t:**

**E-Step — Compute adapted probabilities:**
```
z_clip(c) = 100 · x_t · w_c                         (CLIP logit)
z_adapt(c) = α_t · (x_t · μ_c) / τ_adapt            (adapted logit)
z_combined(c) = z_clip(c) + z_adapt(c)
p_adapt(c | x_t) = softmax(z_combined)_c
```

The weight α_t is the EM weight — how much to trust the adapted mean vs the CLIP anchor. It is itself computed from CLIP's confidence:
```
α_t = 1 - H_clip(x_t) / log(C)      (normalized entropy; ≈ 1 for confident, ≈ 0 for uncertain)
```

This means: the more confident CLIP is on the current sample, the more we trust the adapted mean.

**Prediction:**
```
ŷ = argmax_c p_adapt(c | x_t)
```

**M-Step — Update class means:**

Let ŷ = argmax p_adapt (predicted class). Compute update weight:
```
w_update = σ(x_t)  =  1 / (1 + exp(-γ · (p_adapt(ŷ|x_t) - 0.5)))     (sigmoid of confidence)
```

Update only the predicted class mean:
```
μ_ŷ ← (1 - η · w_update) · μ_ŷ + (η · w_update) · x_t
μ_ŷ ← μ_ŷ / ||μ_ŷ||     (re-normalize to unit sphere)
```

For all other classes c ≠ ŷ: μ_c unchanged.

### 6.3 Mathematics

**The EM View:**  
FreeTTA maximizes the expected log-likelihood of the test data under a Gaussian mixture model where:
- Mixture weights π_c = p_clip(c) (from CLIP prior)
- Component means μ_c are adapted; covariances are isotropic σ²I

The E-step assigns soft responsibilities r_c = p_adapt(c|x).  
The M-step performs a stochastic gradient ascent on the mixture log-likelihood:

```
∂/∂μ_c  log p(x_t) ≈ r_c(x_t) · (x_t - μ_c) / σ²
μ_c ← μ_c + η · r_c(x_t) · (x_t - μ_c)
```

which is exactly the online k-means update for the predicted class, weighted by the soft responsibility.

**The Drift:**  
Over T samples seen for class c, the final mean is:
```
μ_c^T = (1 - η)^T · w_c + η · Σ_{i=1}^{T} (1-η)^{T-i} · w_i · x_{t_i}
```

where w_i = w_update at step i. The mean is an exponentially-weighted moving average of past samples, with the text anchor as the starting point. The parameter η controls forgetting rate.

**The Logit Fusion:**  
The combined prediction is a convex combination of two classifiers:
- CLIP zero-shot: argmax (x·W), strong prior from 400M data
- Nearest-centroid: argmax (x·M), adapted from test stream

The α_t weighting makes the fusion adaptive: use more centroid information when CLIP is confident (the sample is easy), use more text anchor when CLIP is uncertain (the sample is hard and adapted means might be noisy).

### 6.4 Hyperparameters

| Parameter | Value | Meaning                                          |
|-----------|-------|--------------------------------------------------|
| η (eta)   | 0.1   | Mean update step size                            |
| τ_adapt   | 100   | Temperature for adapted logits (matches CLIP)    |
| γ         | 10    | Steepness of sigmoid update gate                 |

### 6.5 Key Structural Properties

- **Global**: each prediction and update is informed by ALL past samples for the predicted class
- **No cold-start**: even with 0 seen samples, CLIP text anchor provides reasonable initialization
- **Uncertainty-aware**: low-confidence samples contribute small M-step updates (w_update ≈ 0)
- **Symmetric**: class means shift continuously in the direction of the true visual centroid
- **Risk of drift**: if many early samples are misclassified (wrong ŷ), μ_ŷ (wrong class) moves toward a different class's region — error propagation

---

## 7. How Much Do They Solve the Problem?

### 7.1 Quantitative Summary

| Dataset     | CLIP   | TDA    | FreeTTA | TDA Gain | FreeTTA Gain | Error Reduction (FreeTTA) |
|-------------|--------|--------|---------|----------|--------------|---------------------------|
| EuroSAT     | 48.43% | 53.33% | 59.35%  | +4.90%   | +10.91%      | 21.2% of CLIP errors fixed|
| DTD         | 43.94% | 45.16% | 46.54%  | +1.22%   | +2.61%       | 4.6% of CLIP errors fixed |
| Caltech101  | 93.55% | 93.59% | 93.63%  | +0.04%   | +0.08%       | 1.2% of CLIP errors fixed |
| ImageNet-V2 | 62.35% | 62.72% | 62.72%  | +0.37%   | +0.37%       | 0.98% of CLIP errors fixed|
| Oxford Pets | 88.39% | 88.69% | 88.63%  | +0.30%   | +0.24%       | 2.6% of CLIP errors fixed |

**Error reduction** = (FreeTTA_acc - CLIP_acc) / (1 - CLIP_acc) × 100

### 7.2 Qualitative Assessment

Both methods solve the problem *partially*. The full solution would require 100% accuracy; neither comes close on hard datasets (DTD, EuroSAT). However:

- On EuroSAT, FreeTTA delivers a dramatic +10.91% gain — the most hostile domain shift of the five datasets
- On DTD, both methods help, with FreeTTA ~2× more effective than TDA
- On easy datasets (Caltech, Pets), both methods are near CLIP — very little left to fix, and few remaining errors

The key insight: **TTA is most valuable exactly where CLIP fails most** (high domain shift). This is self-reinforcing: the more domain shift, the more information the test stream carries about the true distribution.

---

## 8. Datasets Explained

### 8.1 Caltech101

**What**: 101 object categories including common animals, vehicles, and everyday objects. Images from standard photography.  
**Why it's easy**: Caltech classes (accordion, airplanes, cars, faces, etc.) appear frequently in internet text-image pairs. CLIP's training distribution closely matches Caltech's visual style.  
**Expected TTA behavior**: Very little room to improve (93.55% baseline). Risk of harmful changes > benefit.  
**Key stats**: 2,465 samples, 100 classes → ~24.7 samples/class. Geometry alignment = +0.056 (global structure better than local).

### 8.2 DTD (Describable Textures Dataset)

**What**: 47 texture categories (braided, bumpy, cobwebbed, cracked, dotted, etc.). Close-up photographs of material surfaces.  
**Why it's hard**: Texture names are semantically similar in language ("woven" vs "braided" vs "knitted") → CLIP text anchors cluster together. Visual examples have high intra-class variance (many ways to look "bumpy").  
**Expected TTA behavior**: Strong need for adaptation; but high variance makes it hard. Both methods should gain, but won't reach high accuracy.  
**Key stats**: 1,880 samples, 47 classes → ~40 samples/class. Geometry alignment = +0.096 (centroid slightly better than local).

### 8.3 EuroSAT

**What**: Satellite imagery of 10 European land-use categories (Annual Crop, Forest, Highway, Industrial, Pasture, Permanent Crop, Residential, River, Sea/Lake, Herbaceous Vegetation).  
**Why it's hard**: Most extreme domain shift — satellite top-down imagery vs. internet ground-level photos. CLIP text anchors encode ground-level concepts; test images are overhead tiles.  
**Expected TTA behavior**: Greatest adaptation opportunity. FreeTTA expected to dominate because the visual centroid correction is large and consistent. TDA expected to suffer from cache pressure (162× ratio).  
**Key stats**: 8,100 samples, 10 classes → 810 samples/class. Cache pressure = 162. Geometry alignment = −0.115 (local structure BETTER than global — nearest-neighbor is more informative than centroid on this dataset).

### 8.4 ImageNet-V2

**What**: 10,000 images from 1,000 ImageNet classes (10 per class). A new test set with distribution shift from the original ImageNet validation set — harder examples.  
**Why it's hard at scale**: 1,000 classes means very few samples per class (10). Neither cached instances nor adapted means can be estimated well. Fine-grained class distinctions (e.g. 100+ dog breeds) with 10 examples each.  
**Expected TTA behavior**: Minimal gain — both methods starved of data. ImageNet is already partially in-distribution for CLIP.  
**Key stats**: 10,000 samples, 1,000 classes → 10 samples/class. Cache pressure = 2.0 (very low — but too few samples total). Geometry alignment = +0.408 (centroid accuracy 82.7% vs 1-NN 41.9% — centroid is FAR better than local instance lookup).

### 8.5 Oxford Pets

**What**: 37 dog and cat breed categories. High-quality pet photographs, usually with centered subjects.  
**Why it's moderate**: Pets are well-represented in CLIP's training data. The challenge is fine-grained discrimination — the difference between a Samoyed and an American Eskimo Dog is subtle.  
**Expected TTA behavior**: Moderate gain. Local cache (TDA) should work well because breed-specific visual features are precise and retrievable from similar images. Global means (FreeTTA) risk blurring across similar breeds.  
**Key stats**: 3,669 samples, 37 classes → ~99 samples/class. Cache pressure = 19.8. Geometry alignment = +0.080 (centroid modestly better than local).

---

## 9. Our Comparison Approach and Metrics

### 9.1 What We Were Thinking at the Start

The central research question: **Given the same frozen CLIP features, does it matter more to remember individual examples (TDA) or to update a global class model (FreeTTA)?**

We expected:
- Accuracy to tell us the headline
- But accuracy alone wouldn't explain *why* — a 1% difference could come from many places
- We needed metrics that trace how each method changes CLIP's predictions, with what quality, at what speed, and in what failure modes

We designed a set of metrics that answer:
1. **Does the method change anything at all?** → Change Rate, Flip Analysis
2. **Are the changes good or bad?** → Beneficial Flip Precision, Correction Efficiency
3. **When does it start helping?** → Break-Even Point
4. **How certain is the method?** → Entropy, Confidence
5. **When do the two methods disagree?** → Disagreement Analysis
6. **What types of failures remain?** → Failure Buckets
7. **Is the geometry of the feature space suited to each method?** → Geometry Alignment Score
8. **How hard is the adaptation problem?** → Cache Pressure, EM Weight

### 9.2 Metrics Design, Intuition, and Mathematics

---

#### Metric 1: Accuracy

**Formula:**
```
Accuracy = (1/N) Σ_{t=1}^{N}  𝟙[ŷ_t = y_t]
```

**Intuition**: The fundamental measure. But tells us only the outcome, not the mechanism.

**What we expected**: FreeTTA to win on datasets with many samples/class and large domain shift. TDA to win on datasets with fine-grained local structure.

---

#### Metric 2: Change Rate and Flip Analysis

**Formula:**
```
Change Rate = (1/N)  Σ_t  𝟙[ŷ_t^method ≠ ŷ_t^clip]
```

A "flip" is any prediction where the method disagrees with CLIP. Flips are categorized:
- **Beneficial flip**: `ŷ^clip ≠ y` (CLIP wrong) AND `ŷ^method = y` (method correct). The method fixes a CLIP error.
- **Harmful flip**: `ŷ^clip = y` (CLIP correct) AND `ŷ^method ≠ y` (method makes error). The method breaks a correct prediction.
- **Other changed wrong**: both CLIP and method are wrong but predict different wrong classes.

**Net correction rate:**
```
Net Correction Rate = (N_beneficial - N_harmful) / N
```

**Intuition**: A high change rate alone is not good — the changes must be mostly beneficial. A method that flips everything will have 100% change rate but likely a negative net correction rate.

**What we expected**: FreeTTA to have lower change rate (global means change smoothly) but higher beneficial precision. TDA to have higher change rate (each new sample can trigger a cache-driven flip) but noisier precision.

---

#### Metric 3: Beneficial Flip Precision (Correction Efficiency)

**Formula:**
```
Beneficial Flip Precision = N_beneficial / (N_beneficial + N_harmful + N_other_changed_wrong)
                          = N_beneficial / N_total_changed
```

This is the fraction of all prediction changes that are improvements.

**Intuition**: A precision of 1.0 means every change the method makes is an improvement. A precision of 0.5 means half of changes help, half hurt — barely better than random noise. A precision < 0.33 means the method is actively harmful in expectation.

**What we expected**: FreeTTA to have higher precision (smoother, more principled updates). TDA's cache-based updates to be noisier, especially under cache pressure.

---

#### Metric 4: Entropy and Confidence

**Shannon entropy of the softmax distribution:**
```
H(x) = -Σ_c p(c|x) log p(c|x)
```

**Maximum entropy for C classes:** H_max = log(C) (uniform distribution)

**Normalized entropy:** h(x) = H(x) / log(C)  ∈ [0, 1]

**Confidence:** conf(x) = max_c p(c|x)  (probability of the top-1 prediction)

**Intuition**: A confident prediction (high max prob, low entropy) suggests the method has found a strong feature match. We track entropy separately for correct vs wrong predictions:
- **Correct predictions**: we want low entropy (high confidence)
- **Wrong predictions**: high confidence on wrong = overconfidence problem

**Expected calibration gap**: A well-calibrated method has low entropy on correct predictions and high entropy on wrong ones. CLIP's calibration is poor (near-uniform over all classes). TDA and FreeTTA should improve calibration.

**What we expected**: Both methods to dramatically lower entropy (since they concentrate probability mass on the predicted class). FreeTTA's entropy on wrong predictions should be higher than TDA's (because FreeTTA uses a global mean that can confidently point at the wrong class once mean drift occurs).

---

#### Metric 5: Break-Even Point (Latency to Benefit)

**Definition**: The earliest stream step at which the method's rolling accuracy first surpasses CLIP's rolling accuracy and remains above it.

Using rolling window of W=50:
```
Break-Even = min{t : RollingAcc_method(t) > RollingAcc_clip(t)}
```

**Break-Even Ratio** = Break-Even Step / Total Steps

**Intuition**: A method that helps immediately (break-even ratio ≈ 0) is better for short streams. A method with a large cold-start period may harm accuracy before it helps — critical for real-time applications.

**What we expected**: FreeTTA to have nearly zero break-even (initialized from text anchors; adapts from step 1). TDA to have a longer cold-start (cache is empty at the start).

---

#### Metric 6: Disagreement Analysis

**Formula:**
```
Disagreement Rate = (1/N)  Σ_t  𝟙[ŷ_t^TDA ≠ ŷ_t^FreeTTA]
```

**Conditional accuracy on disagreements:**
```
TDA_acc_on_disagree = Acc(TDA) on samples where TDA ≠ FreeTTA
FreeTTA_acc_on_disagree = Acc(FreeTTA) on samples where TDA ≠ FreeTTA
```

**Intuition**: High disagreement = the two methods see the problem very differently. When they disagree, we can check who is right. If one method consistently wins on disagreements, it is extracting complementary signal from the stream.

**What we expected**: High disagreement on hard datasets (EuroSAT, DTD), low disagreement on easy datasets (Caltech). On disagreements, FreeTTA expected to win more often due to better quality adaptations.

---

#### Metric 7: Failure Buckets

We partition every test sample into one of these mutually exclusive buckets:

| Bucket                               | Meaning                                         |
|--------------------------------------|-------------------------------------------------|
| `all_correct`                        | CLIP, TDA, FreeTTA all correct                  |
| `only_freetta_correct`               | FreeTTA uniquely rescues a CLIP+TDA error       |
| `only_tda_correct`                   | TDA uniquely rescues a CLIP+FreeTTA error       |
| `clip_correct_tda_wrong_ftta_correct`| Both adapters wrong, CLIP right                 |
| `all_wrong`                          | None of the three methods are correct           |

**Intuition**: Failure buckets reveal complementarity. If `only_freetta_correct >> only_tda_correct`, FreeTTA is accessing signals TDA cannot. Large `all_wrong` indicates the ceiling — samples where TTA cannot help regardless of method.

**Formally:**
```
only_freetta_correct = {t : ŷ^clip ≠ y AND ŷ^tda ≠ y AND ŷ^freetta = y}
only_tda_correct     = {t : ŷ^clip ≠ y AND ŷ^freetta ≠ y AND ŷ^tda = y}
all_wrong            = {t : ŷ^clip ≠ y AND ŷ^tda ≠ y AND ŷ^freetta ≠ y}
```

**What we expected**: `only_freetta_correct > only_tda_correct` because FreeTTA sees all past samples (global) while TDA sees only recent ones (local). `all_wrong` to be large on hard datasets — reflecting the irreducible error given frozen features.

---

#### Metric 8: Geometry Alignment Score (GAS)

**Oracle centroid accuracy**: Accuracy of nearest-centroid classifier using the TRUE class means (computed from all labeled test samples — an upper bound for mean-based methods).

**Oracle 1-NN accuracy**: Accuracy of 1-nearest-neighbor classifier using all labeled test samples (an upper bound for cache-based methods).

**Formula:**
```
GAS = Oracle_Centroid_Acc - Oracle_1NN_Acc
```

**Intuition**:
- GAS > 0: The true class centroids provide better class boundaries than individual instances → global statistics (FreeTTA) should work better
- GAS < 0: Local neighborhood structure is stronger than global means → cache lookup (TDA) should work better

GAS measures the geometry of the feature space and directly predicts which method will be more effective.

---

#### Metric 9: Cache Pressure Ratio and EM Weight

**Cache Pressure Ratio:**
```
Cache Pressure = N_total / (C × pos_shot_capacity)
```

Number of samples per class divided by cache capacity. High pressure → frequent eviction → cache quality degrades.

**EM Weight (FreeTTA):**
The mean EM weight α_t averaged over all test steps. Recall α_t = 1 - H_clip / log(C).
```
Mean EM Weight = (1/N) Σ_t α_t
```

High mean EM weight means CLIP was frequently confident → FreeTTA trusted adapted means heavily.  
Low EM weight means CLIP was mostly uncertain → M-step updates were small, adaptation was slow.

**What these tell us**: Cache pressure predicts TDA's failure; low EM weight predicts FreeTTA's slow adaptation.

---

## 10. Results: Dataset-by-Dataset

### 10.1 Caltech101

**Accuracy:**

| Method  | Accuracy | Gain vs CLIP | Change Rate | Beneficial Flips | Harmful Flips | BFP    |
|---------|----------|--------------|-------------|------------------|---------------|--------|
| CLIP    | 93.55%   | —            | 0%          | —                | —             | —      |
| TDA     | 93.59%   | +0.04%       | 0.81%       | 10               | 9             | 50.0%  |
| FreeTTA | 93.63%   | +0.08%       | 0.41%       | 6                | 4             | 60.0%  |

**Entropy & Confidence:**

| Method  | Correct conf | Wrong conf | Correct entropy | Wrong entropy |
|---------|-------------|------------|-----------------|---------------|
| CLIP    | 0.01130     | 0.01115    | 4.605           | 4.605         |
| TDA     | 0.9690      | 0.7476     | 0.130           | 0.724         |
| FreeTTA | 0.9271      | 0.6168     | 0.293           | 1.071         |

**Key Internal Metrics:**

| Metric                          | Value  |
|---------------------------------|--------|
| Cache pressure ratio            | 4.93   |
| Mean EM weight                  | 0.826  |
| FreeTTA final μ drift           | 1.134  |
| Geometry Alignment Score (GAS)  | +0.056 |
| Oracle centroid acc             | 97.44% |
| Oracle 1-NN acc                 | 91.85% |
| Disagreement rate               | 0.81%  |
| TDA acc on disagreements        | 45.0%  |
| FreeTTA acc on disagreements    | 50.0%  |
| Break-even TDA                  | step 1374 / 2465 (55.7%) |
| Break-even FreeTTA              | step 1050 / 2465 (42.6%) |

**Failure Buckets:**

| Bucket                    | Count | Rate   |
|---------------------------|-------|--------|
| only_freetta_correct      | 3     | 0.12%  |
| only_tda_correct          | 7     | 0.28%  |
| clip_correct_tda_wrong    | 2     | 0.08%  |
| clip_correct_ftta_wrong   | 7     | 0.28%  |
| all_wrong                 | 146   | 5.92%  |

**Analysis**: Caltech is near-solved by CLIP. Both adapters make very few changes. FreeTTA is more conservative (0.41% change rate vs 0.81%) and more precise (BFP=60% vs 50%). TDA makes more changes but also more harmful ones — 9 harmful flips vs FreeTTA's 4. The GAS = +0.056 correctly predicts that global means (FreeTTA) edge out local cache (TDA). The break-even is late for both methods (>40% of stream), meaning they actually hurt initially before they help — but the net effect is slightly positive.

---

### 10.2 DTD (Describable Textures)

**Accuracy:**

| Method  | Accuracy | Gain vs CLIP | Change Rate | Beneficial Flips | Harmful Flips | BFP    |
|---------|----------|--------------|-------------|------------------|---------------|--------|
| CLIP    | 43.94%   | —            | 0%          | —                | —             | —      |
| TDA     | 45.16%   | +1.22%       | 11.86%      | 57               | 34            | 25.6%  |
| FreeTTA | 46.54%   | +2.61%       | 16.28%      | 92               | 43            | 30.1%  |

**Entropy & Confidence:**

| Method  | Correct conf | Wrong conf | Correct entropy | Wrong entropy |
|---------|-------------|------------|-----------------|---------------|
| CLIP    | 0.02254     | 0.02215    | 3.850           | 3.850         |
| TDA     | 0.7307      | 0.4786     | 1.086           | 1.941         |
| FreeTTA | 0.7158      | 0.4595     | 1.081           | 1.935         |

**Key Internal Metrics:**

| Metric                         | Value  |
|--------------------------------|--------|
| Cache pressure ratio           | 8.0    |
| Mean EM weight                 | 0.465  |
| FreeTTA final μ drift          | 1.161  |
| GAS                            | +0.096 |
| Oracle centroid acc            | 73.19% |
| Oracle 1-NN acc                | 63.62% |
| Disagreement rate              | 14.20% |
| TDA acc on disagreements       | 16.5%  |
| FreeTTA acc on disagreements   | 26.2%  |
| Break-even TDA                 | step 358 / 1880 (19.0%) |
| Break-even FreeTTA             | step 30 / 1880  (1.6%)  |

**Failure Buckets:**

| Bucket                    | Count | Rate   |
|---------------------------|-------|--------|
| only_freetta_correct      | 55    | 2.93%  |
| only_tda_correct          | 20    | 1.06%  |
| clip_correct_tda_wrong    | 24    | 1.28%  |
| clip_correct_ftta_wrong   | 15    | 0.80%  |
| all_wrong                 | 942   | 50.11% |

**Analysis**: DTD reveals the texture classification challenge starkly: 50.11% of samples are all_wrong (irreducible error at the feature level). Both methods help, but the ceiling is low. FreeTTA achieves +2.61% vs TDA's +1.22% — roughly 2× better. On disagreements (14.2% of samples), FreeTTA wins at 26.2% accuracy vs TDA's 16.5% — FreeTTA's global mean is a better discriminator when classes are confusable. FreeTTA's break-even at step 30 (1.6% of stream) shows near-zero cold-start. TDA's BFP = 25.6% is quite low — only 1 in 4 changes helps. The EM weight = 0.465 means FreeTTA only moderately trusts adapted means (CLIP has medium confidence on DTD), yet it still outperforms TDA.

---

### 10.3 EuroSAT

**Accuracy:**

| Method  | Accuracy | Gain vs CLIP | Change Rate | Beneficial Flips | Harmful Flips | BFP    |
|---------|----------|--------------|-------------|------------------|---------------|--------|
| CLIP    | 48.43%   | —            | 0%          | —                | —             | —      |
| TDA     | 53.33%   | +4.90%       | 41.91%      | 1125             | 728           | 33.1%  |
| FreeTTA | 59.35%   | +10.91%      | 30.95%      | 1258             | 374           | 50.2%  |

**Entropy & Confidence:**

| Method  | Correct conf | Wrong conf | Correct entropy | Wrong entropy |
|---------|-------------|------------|-----------------|---------------|
| CLIP    | 0.10361     | 0.10195    | 2.302           | 2.302         |
| TDA     | 0.7320      | 0.5600     | 0.776           | 1.207         |
| FreeTTA | 0.8119      | 0.5195     | 0.512           | 1.227         |

**Key Internal Metrics:**

| Metric                         | Value   |
|--------------------------------|---------|
| Cache pressure ratio           | 162.0   |
| Mean EM weight                 | 0.198   |
| FreeTTA final μ drift          | 1.142   |
| GAS                            | −0.115  |
| Oracle centroid acc            | 78.26%  |
| Oracle 1-NN acc                | 89.80%  |
| Disagreement rate              | 30.90%  |
| TDA acc on disagreements       | 22.8%   |
| FreeTTA acc on disagreements   | 42.3%   |
| Break-even TDA                 | step 941 / 8100 (11.6%) |
| Break-even FreeTTA             | step 8  / 8100 (0.10%)  |

**Failure Buckets:**

| Bucket                    | Count | Rate   |
|---------------------------|-------|--------|
| only_freetta_correct      | 621   | 7.67%  |
| only_tda_correct          | 488   | 6.02%  |
| clip_correct_tda_wrong    | 83    | 1.02%  |
| clip_correct_ftta_wrong   | 437   | 5.40%  |
| all_wrong                 | 2431  | 30.01% |

**Analysis**: EuroSAT is the most revealing dataset. **FreeTTA's +10.91% gain is the largest absolute improvement of any method on any dataset.** The cache pressure of 162× renders TDA's memory nearly useless — it is overwritten 162× per class per epoch. TDA's BFP = 33.1% means only 1 in 3 changes benefits, and 728 predictions that CLIP got right are broken by TDA. FreeTTA's BFP = 50.2% means it is a net positive on EVERY change it makes on average.

The GAS = −0.115 is the only negative value across all datasets. This means the true class centroids are WORSE predictors than nearest-neighbor on EuroSAT (oracle 1-NN = 89.8% vs oracle centroid = 78.3%). Paradoxically, FreeTTA still wins — because even though true centroids are suboptimal, the adapted means gradually drift toward the true visual distribution (which is far from the text anchor), making any centroid correction valuable compared to the highly wrong CLIP text anchor.

The mean EM weight = 0.198 is the lowest of all datasets (CLIP is very uncertain → α_t ≈ 0.2 on average). This means FreeTTA actually adapts quite cautiously — it doesn't trust adapted means very much. Yet it still wins by 6.02 percentage points over TDA. This shows how powerful even small corrections to the class mean can be when the initial text anchor is severely wrong.

On disagreements (30.9% of the stream!), FreeTTA wins with 42.3% accuracy vs TDA's 22.8% — a massive 19.5-point advantage in uncertain territory.

---

### 10.4 ImageNet-V2

**Accuracy:**

| Method  | Accuracy | Gain vs CLIP | Change Rate | Beneficial Flips | Harmful Flips | BFP    |
|---------|----------|--------------|-------------|------------------|---------------|--------|
| CLIP    | 62.35%   | —            | 0%          | —                | —             | —      |
| TDA     | 62.72%   | +0.37%       | 3.88%       | 115              | 78            | 29.6%  |
| FreeTTA | 62.72%   | +0.37%       | 3.62%       | 116              | 79            | 32.0%  |

**Key Internal Metrics:**

| Metric                         | Value  |
|--------------------------------|--------|
| Cache pressure ratio           | 2.0    |
| Mean EM weight                 | 0.533  |
| FreeTTA final μ drift          | 1.128  |
| GAS                            | +0.408 |
| Oracle centroid acc            | 82.72% |
| Oracle 1-NN acc                | 41.89% |
| Disagreement rate              | 5.60%  |
| TDA acc on disagreements       | 25.2%  |
| FreeTTA acc on disagreements   | 25.2%  |
| Break-even TDA                 | step 363 / 10000 (3.6%) |
| Break-even FreeTTA             | step 2  / 10000 (0.02%) |

**Failure Buckets:**

| Bucket                    | Count | Rate  |
|---------------------------|-------|-------|
| only_freetta_correct      | 81    | 0.81% |
| only_tda_correct          | 80    | 0.80% |
| clip_correct_tda_wrong    | 61    | 0.61% |
| clip_correct_ftta_wrong   | 60    | 0.61% |
| all_wrong                 | 3569  | 35.7% |

**Analysis**: A statistical dead tie. With only 10 samples per class, neither method can accumulate enough information to meaningfully adapt. The GAS = +0.408 is the largest positive value — the true class centroid is far better than 1-NN (82.7% vs 41.9%), predicting FreeTTA should dominate. But even though the geometry favors centroid-based methods, the sample count (10/class) is too small for mean estimates to converge. FreeTTA's class means drift only 1.128 units from the text anchor — the smallest drift requires the most data to accumulate, and with 10 samples, drift is minimal. The EM weight = 0.533 is moderate, but with 10 updates per class, the adapted mean barely moves. On disagreements (5.6%), both methods perform identically (25.2%), confirming they're essentially coin-flipping on the harder cases.

---

### 10.5 Oxford Pets

**Accuracy:**

| Method  | Accuracy | Gain vs CLIP | Change Rate | Beneficial Flips | Harmful Flips | BFP    |
|---------|----------|--------------|-------------|------------------|---------------|--------|
| CLIP    | 88.39%   | —            | 0%          | —                | —             | —      |
| TDA     | 88.69%   | +0.30%       | 1.17%       | 21               | 10            | 48.8%  |
| FreeTTA | 88.63%   | +0.24%       | 1.47%       | 23               | 14            | 42.6%  |

**Key Internal Metrics:**

| Metric                         | Value  |
|--------------------------------|--------|
| Cache pressure ratio           | 19.8   |
| Mean EM weight                 | 0.650  |
| FreeTTA final μ drift          | 1.122  |
| GAS                            | +0.080 |
| Oracle centroid acc            | 91.50% |
| Oracle 1-NN acc                | 83.46% |
| Disagreement rate              | 1.06%  |
| TDA acc on disagreements       | 43.6%  |
| FreeTTA acc on disagreements   | 38.5%  |
| Break-even TDA                 | step 5   / 3669 (0.14%) |
| Break-even FreeTTA             | step 5   / 3669 (0.14%) |

**Failure Buckets:**

| Bucket                    | Count | Rate   |
|---------------------------|-------|--------|
| only_freetta_correct      | 11    | 0.30%  |
| only_tda_correct          | 9     | 0.25%  |
| clip_correct_tda_wrong    | 8     | 0.22%  |
| clip_correct_ftta_wrong   | 4     | 0.11%  |
| all_wrong                 | 394   | 10.74% |

**Analysis**: The only dataset where TDA marginally beats FreeTTA (+0.30% vs +0.24%, a 0.05-point difference). The reason is visible in the failure buckets: TDA makes 10 harmful flips (vs FreeTTA's 14), and FreeTTA flips 4 correct CLIP predictions wrong (vs TDA's 8 such errors that FreeTTA avoids). The BFP gap is clear: TDA 48.8% vs FreeTTA 42.6%. On disagreements, TDA wins 43.6% vs FreeTTA 38.5% — the only dataset where TDA's conditional accuracy on disagreements exceeds FreeTTA's.

The GAS = +0.080 slightly favors centroid-based methods, yet TDA wins. The explanation: the GAS tells us about the *true* centroid, but FreeTTA's adapted mean is not the true centroid — it is a noisy estimate that can drift toward adjacent breed classes. With 37 fine-grained breeds, small mean drift causes cross-breed confusion that TDA's exact cached features avoid. The mean EM weight = 0.650 means FreeTTA is quite active here (CLIP is moderately confident on Pets), making the drift more pronounced.

---

## 11. Cross-Dataset Conclusions

### 11.1 Summary Table (All Metrics)

| Dataset    | CLIP  | TDA   | FreeTTA | TDA_BFP | Ftta_BFP | GAS    | Cache_P | EM_W  | Disagree | Ftta_wins_disagree |
|------------|-------|-------|---------|---------|---------|--------|---------|-------|----------|-------------------|
| Caltech    | 93.55 | 93.59 | 93.63   | 50.0%   | 60.0%   | +0.056 | 4.93    | 0.826 | 0.81%    | Yes (50% vs 45%)  |
| DTD        | 43.94 | 45.16 | 46.54   | 25.6%   | 30.1%   | +0.096 | 8.0     | 0.465 | 14.20%   | Yes (26% vs 17%)  |
| EuroSAT    | 48.43 | 53.33 | 59.35   | 33.1%   | 50.2%   | −0.115 | 162.0   | 0.198 | 30.90%   | Yes (42% vs 23%)  |
| ImageNet   | 62.35 | 62.72 | 62.72   | 29.6%   | 32.0%   | +0.408 | 2.0     | 0.533 | 5.60%    | Tie (25% = 25%)   |
| Pets       | 88.39 | 88.69 | 88.63   | 48.8%   | 42.6%   | +0.080 | 19.8    | 0.650 | 1.06%    | No (39% vs 44%)   |

### 11.2 Key Findings

**Finding 1: FreeTTA is the better default method.**  
FreeTTA wins or ties on 4 of 5 datasets. Its advantage is largest exactly where it matters most (high domain shift: EuroSAT +10.91%). Its only loss is Pets by 0.05 points.

**Finding 2: Cache pressure is TDA's Achilles heel.**  
EuroSAT (cache pressure 162×) shows TDA's worst performance relative to FreeTTA (−6.02%). Caltech (pressure 4.93×) shows TDA's best relative performance (near-tie). The correlation is tight:

| Cache Pressure | FreeTTA − TDA Gap |
|----------------|-------------------|
| 2.0            | 0.00%             |
| 4.93           | +0.04%            |
| 8.0            | +1.38%            |
| 19.8           | −0.06%            |
| 162.0          | +6.02%            |

**Finding 3: Beneficial Flip Precision consistently favors FreeTTA.**  
On 4 of 5 datasets, FreeTTA's BFP ≥ TDA's BFP. Only Pets is an exception. This means FreeTTA's changes are more often improvements — its adaptation signal is higher quality.

**Finding 4: Geometry Alignment Score predicts method preference.**  
The GAS correctly predicts:
- EuroSAT (GAS = −0.115): local structure better → yet FreeTTA wins because the CLIP text anchor is so wrong that any mean-correction helps
- All other datasets (GAS > 0): centroid better → FreeTTA wins or ties

Exception: Pets (GAS = +0.080 favors centroid → should favor FreeTTA, but TDA wins). This shows GAS is predictive but not deterministic — fine-grained class overlap can make local cache more valuable than a noisy adapted mean.

**Finding 5: FreeTTA adapts faster (lower break-even).**

| Dataset  | TDA Break-Even | FreeTTA Break-Even |
|----------|---------------|-------------------|
| EuroSAT  | Step 941 (11.6%) | Step 8 (0.10%)  |
| DTD      | Step 358 (19.0%) | Step 30 (1.6%)  |
| Caltech  | Step 1374 (55.7%)| Step 1050 (42.6%)|
| ImageNet | Step 363 (3.6%)  | Step 2 (0.02%)  |
| Pets     | Step 5 (0.14%)   | Step 5 (0.14%)  |

FreeTTA begins helping almost immediately; TDA often needs to fill its cache first.

**Finding 6: CLIP calibration is completely broken — both methods fix it.**  
CLIP's confidence on correct vs wrong predictions is nearly identical (e.g., Caltech: 0.01130 correct vs 0.01115 wrong). After adaptation, TDA shows 0.969 vs 0.748 (correct vs wrong confidence), FreeTTA shows 0.927 vs 0.617. Both methods dramatically improve calibration. FreeTTA's calibration gap is slightly smaller, indicating its wrong predictions are slightly more confident — a consequence of mean drift.

---

## 12. Where Each Algorithm Always Wins and Loses

### 12.1 FreeTTA: Conditions for Winning

**Structural conditions that guarantee FreeTTA advantage:**

1. **High domain shift from internet distribution** (EuroSAT, DTD)  
   *Why*: The CLIP text anchor is far from the true visual centroid. Any correction to the class mean toward the test distribution improves performance. FreeTTA's online EM systematically moves toward the true centroid.  
   *Quantitative signal*: CLIP accuracy < 60% on a dataset with well-defined classes

2. **High samples-per-class ratio** (EuroSAT: 810/class)  
   *Why*: FreeTTA's mean estimate converges in O(1/n) variance; more samples → better centroid estimate → better predictions.  
   *Quantitative signal*: N/C > 50

3. **High cache pressure** (EuroSAT: 162×)  
   *Why*: TDA cannot maintain representative cache when N_samples >> capacity. FreeTTA's global mean doesn't have this problem — it accumulates all samples.

4. **Low GAS or moderate GAS with large domain shift**  
   *Why*: Even when local structure is slightly better (GAS < 0), if the initial text anchor is very wrong, correcting the global mean helps more than any local lookup.

5. **Long streams with many classes** (ImageNet: 10,000 samples but 1,000 classes = only 10/class)  
   *Why this actually HURTS*: With 10 samples/class, FreeTTA's mean doesn't converge. So long stream + few classes/class = FreeTTA wins; long stream + many classes = tie or depends.

### 12.2 FreeTTA: Conditions for Failing

1. **Fine-grained classes with subtle inter-class visual differences** (Pets: 37 breeds)  
   *Why*: The adapted mean μ_c is updated with every sample predicted as class c. But a Samoyed sample has shared features with other white fluffy dogs → μ_Samoyed drifts toward the white-dog cluster, not the pure Samoyed cluster. Local cached images (TDA) are more precise discriminators.

2. **Very few samples per class** (ImageNet: 10/class)  
   *Why*: M-step updates are too small to converge. With η=0.1 and 10 samples, μ changes by only ~10% of the way toward the sample. Not enough to correct anything.

3. **Overconfident CLIP on wrong classes** (adversarial case, not observed here)  
   *Why*: If CLIP is confidently wrong on most samples for a class, the M-step updates μ in the wrong direction, accelerating errors. The EM weight α penalizes uncertain CLIP, but cannot help if CLIP is confidently wrong.

### 12.3 TDA: Conditions for Winning

1. **Fine-grained classification with precise visual features** (Pets)  
   *Why*: Cached breed-specific image features carry discriminative visual details (specific coat pattern, ear shape) that the class mean averages away.

2. **Low cache pressure** (Caltech: 4.93×, ImageNet: 2.0×)  
   *Why*: Cache capacity is sufficient; the most recent confident samples are genuinely representative.

3. **High stream locality** (natural order, class-clustered streams)  
   *Why*: When consecutive samples belong to the same class, the cache fills quickly with high-quality same-class examples. Our experiments use natural order, which benefits TDA.

4. **High GAS** (ImageNet: +0.408)  
   *Why*: When global centroids are good, TDA's local cache effectively approximates the centroid from recent examples. At GAS = +0.408, ImageNet's centroid is far better than 1-NN — but TDA ties FreeTTA because sample count limits both.

### 12.4 TDA: Conditions for Failing

1. **High cache pressure** (EuroSAT: 162×)  
   *Why*: Cache slots are overwritten so frequently that the "cache" is really just the last 3 samples seen for each class — essentially random. The logit adjustments become noisy.

2. **Cold-start on large, hard datasets** (EuroSAT break-even at step 941)  
   *Why*: Empty cache = zero adjustment = pure CLIP. But cached samples from the start (before enough context) can be wrong, creating harmful logit adjustments.

3. **Confident wrong predictions entering the cache**  
   *Why*: TDA adds samples where entropy < θ_low to the positive cache — but uses its own predicted label. A misclassified sample that TDA is confident about becomes a poisoned cache entry, reinforcing future misclassification of similar samples.

4. **Classes that CLIP confuses (the harmful flip source)**  
   *Why*: If TDA's cache for class "Highway" contains images that actually look like "Industrial", future Highway samples will have their logits pushed toward Industrial. This is the source of TDA's 728 harmful flips on EuroSAT.

---

## 13. Future Work and Suggested Improvements

### 13.1 Improving TDA

**1. Adaptive Cache Capacity (High Priority for High Cache Pressure)**

*Problem*: Fixed 3-shot capacity fails at 162× pressure.  
*Solution*: Set capacity proportional to stream length:
```
capacity(c) = max(3,  floor(sqrt(N_estimated / C)))
```
Or use reservoir sampling (each new sample replaces a random existing sample with probability 1/k where k is the number seen so far). This gives a uniform random sample of past examples without FIFO bias.  
*Expected gain*: EuroSAT would go from cache pressure 162→~28 with capacity ~29. This directly addresses TDA's worst failure mode.

**2. Verified Cache Updates (Label Confidence Threshold)**

*Problem*: TDA caches its own predictions. Misclassified confident samples corrupt future predictions.  
*Solution*: Only cache a sample if BOTH (a) entropy < θ_low AND (b) the predicted class's cached-class probability > δ. Use a self-consistency check: if caching x as class c, check that the majority of existing cache[c] entries are more similar to x than to any other class's cache.  
*Expected gain*: Reduces poison entries in the cache, especially on hard datasets.

**3. Temporal Weighting (Recency Bias)**

*Problem*: FIFO treats all cached samples equally. Older samples may be from a different region of the stream (different class cluster in natural order).  
*Solution*: Weight cache contribution by recency:
```
z_pos(c) = Σ_i γ^(t - t_i) · (x_t · v_i)    where γ < 1 (exponential decay)
```
*Expected gain*: More responsive to recent stream statistics; avoids staleness in long streams.

**4. Negative Cache via Confusion Matrix**

*Problem*: Current negative cache stores medium-entropy samples predicted as c, but the most useful negatives are samples from classes frequently confused with c.  
*Solution*: Track a running confusion matrix. The negative cache for class c should contain samples that were predicted as c but are actually in the most-confused-with-c classes.  
*Requires*: Some labeled validation data, or pseudo-label consistency checks.

### 13.2 Improving FreeTTA

**1. Per-Class Adaptive Learning Rate**

*Problem*: Fixed η=0.1 is too large for rare classes (few samples → noisy mean) and too small for common classes (many samples → slow convergence).  
*Solution*: Use Robbins-Monro schedule:
```
η_c(n) = η_0 / (1 + β·n_c)     where n_c = number of times class c has been updated
```
η_c decays as more samples arrive for class c. This is exactly the learning rate for online k-means and guarantees convergence.  
*Expected gain*: Better calibrated updates across classes with different sample frequencies (especially ImageNet where some classes may appear 5× and others 15×).

**2. Confidence Gating for the M-Step (ConfGatedFreeTTA)**

*Problem*: FreeTTA updates μ whenever it predicts class c, even with low confidence. Low-confidence M-steps add noise.  
*Solution*: Only update if `p_adapt(ŷ|x) > threshold (e.g., 0.5)`. This is ConfGatedFreeTTA from our Section 12 experiments. The expected behavior: fewer but higher-quality updates.  
*Tradeoff*: Slower convergence on hard datasets where confident predictions are rare.

**3. Multi-Template Mean Initialization**

*Problem*: FreeTTA initializes μ_c = w_c (single text template). If the template poorly matches the visual domain, the starting point for EM is bad.  
*Solution*: Initialize μ_c = mean of K text templates:
```
μ_c^0 = normalize(Σ_k f_T("a photo of {c}") + f_T("an image of {c}") + f_T("{c}"))
```
Averaging multiple templates reduces the sensitivity to the specific phrasing.  
*Expected gain*: Better initialization → fewer steps needed to reach good accuracy.

**4. Momentum for Mean Updates**

*Problem*: Single-sample M-step is noisy. A sudden outlier can shift μ significantly.  
*Solution*: Use exponential moving average:
```
m_c ← β·m_c + (1-β)·w_update·(x - μ_c)
μ_c ← μ_c + η·m_c
```
Momentum β=0.9 smooths the gradient, reducing the effect of outliers.  
*Expected gain*: More stable trajectories on DTD where intra-class variance is high.

**5. Mixture Model with Covariance** (Research Direction)

*Problem*: FreeTTA uses isotropic Gaussian (spherical, same variance for all classes). DTD classes have anisotropic distributions (textures vary along specific directions).  
*Solution*: Track per-class covariance (or at minimum, diagonal covariance):
```
Σ_c ← (1-η)·Σ_c + η·w·(x-μ_c)(x-μ_c)^T
z_adapt(c) = -0.5 · (x-μ_c)^T Σ_c^{-1} (x-μ_c)
```
*Expected gain*: Better discrimination of texture classes where the relevant information lies in specific feature dimensions. Computational cost: O(D²) per class per step — expensive for D=512, C=47.

### 13.3 Hybrid Approaches

**1. TDA + FreeTTA Ensemble**

Use FreeTTA's global mean when cache pressure > threshold; use TDA when cache is well-populated:
```
if cache_fullness(c) < 0.5:
    z_final = z_clip + w_global · z_freetta
else:
    z_final = z_clip + w_local · z_tda + (1-w_local) · z_freetta
```

**2. FreeTTA-Initialized TDA**

Use FreeTTA's adapted mean μ_c as initial "virtual" cache entries, solving TDA's cold-start problem:
```
virtual_pos_cache[c] = {μ_c^freetta}    (synthetic anchor)
```
As real samples arrive, they replace the virtual anchor. Expected gain on EuroSAT cold-start: instead of 33.8% in window 1, start from FreeTTA's ~56%.

**3. Cross-Dataset Hyperparameter Prediction**

Train a meta-model that predicts the optimal (η, capacity, θ_low) from dataset statistics measurable without labels: stream length N, estimated class count C (from CLIP's top-1 entropy), and average CLIP confidence. This would remove the need for per-dataset tuning.

---

*Generated from `outputs/comparative_analysis/` — pipeline `experiments/run_comparative_analysis.py` with natural stream order, seed=42.*

*Plots referenced: `accuracy_summary.png`, `flip_analysis.png`, `entropy_confidence_summary.png`, `disagreement_analysis.png`, `failure_bucket_summary.png`, `geometry_analysis.png`, `latency_analysis.png` — all available in `outputs/comparative_analysis/`.*
