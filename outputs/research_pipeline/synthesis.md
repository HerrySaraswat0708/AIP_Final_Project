# Section 11 — Synthesis: Deep Insights

This section derives evidence-based observations from the computed metrics.
Conclusions are grounded in numbers from the data, not assumptions.

## 1. Accuracy Insights

### CALTECH
- CLIP baseline: 0.9355  TDA: 0.9343 (-0.0012)  FreeTTA: 0.9375 (+0.0020)  Hybrid: 0.9387
- Best adapter: **hybrid**
- TDA and FreeTTA are roughly on par for this dataset.

### DTD
- CLIP baseline: 0.4394  TDA: 0.4606 (+0.0213)  FreeTTA: 0.4596 (+0.0202)  Hybrid: 0.4479
- Best adapter: **ent_tda**
- TDA and FreeTTA are roughly on par for this dataset.

### EUROSAT
- CLIP baseline: 0.4843  TDA: 0.6215 (+0.1372)  FreeTTA: 0.5974 (+0.1131)  Hybrid: 0.6156
- Best adapter: **ent_tda**
- TDA outperforms FreeTTA by 0.0241.  Local cache memory is more effective, suggesting the test stream has within-class consistency that the cache can exploit.

### IMAGENET
- CLIP baseline: 0.6235  TDA: 0.6289 (+0.0054)  FreeTTA: 0.6268 (+0.0033)  Hybrid: 0.6250
- Best adapter: **ent_tda**
- TDA and FreeTTA are roughly on par for this dataset.

### PETS
- CLIP baseline: 0.8839  TDA: 0.8964 (+0.0125)  FreeTTA: 0.8893 (+0.0055)  Hybrid: 0.8907
- Best adapter: **tda**
- TDA and FreeTTA are roughly on par for this dataset.
## 2. Flip / Correction Efficiency Insights

### CALTECH
- **tda**: change_rate=0.019 CE=0.413  net_correction=-0.0012
- **freetta**: change_rate=0.002 CE=1.000  net_correction=+0.0020
  → High CE (1.000): when freetta changes a prediction it is usually right.
- **conf_ftta**: change_rate=0.002 CE=1.000  net_correction=+0.0020
  → High CE (1.000): when conf_ftta changes a prediction it is usually right.
- **ent_tda**: change_rate=0.018 CE=0.467  net_correction=+0.0004
- **hybrid**: change_rate=0.020 CE=0.520  net_correction=+0.0032

### DTD
- **tda**: change_rate=0.127 CE=0.298  net_correction=+0.0213
  → Low CE (0.298): tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **freetta**: change_rate=0.168 CE=0.273  net_correction=+0.0202
  → Low CE (0.273): freetta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **conf_ftta**: change_rate=0.215 CE=0.205  net_correction=-0.0016
  → Low CE (0.205): conf_ftta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **ent_tda**: change_rate=0.128 CE=0.311  net_correction=+0.0245
  → Low CE (0.311): ent_tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **hybrid**: change_rate=0.142 CE=0.236  net_correction=+0.0085
  → Low CE (0.236): hybrid makes many harmful changes — the cache/statistics are corrupting more than they help.

### EUROSAT
- **tda**: change_rate=0.385 CE=0.475  net_correction=+0.1372
- **freetta**: change_rate=0.334 CE=0.449  net_correction=+0.1131
- **conf_ftta**: change_rate=0.371 CE=0.397  net_correction=+0.1021
  → Low CE (0.397): conf_ftta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **ent_tda**: change_rate=0.387 CE=0.475  net_correction=+0.1383
- **hybrid**: change_rate=0.340 CE=0.485  net_correction=+0.1312

### IMAGENET
- **tda**: change_rate=0.038 CE=0.315  net_correction=+0.0054
  → Low CE (0.315): tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **freetta**: change_rate=0.038 CE=0.307  net_correction=+0.0033
  → Low CE (0.307): freetta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **conf_ftta**: change_rate=0.046 CE=0.284  net_correction=+0.0019
  → Low CE (0.284): conf_ftta makes many harmful changes — the cache/statistics are corrupting more than they help.
- **ent_tda**: change_rate=0.027 CE=0.399  net_correction=+0.0063
  → Low CE (0.399): ent_tda makes many harmful changes — the cache/statistics are corrupting more than they help.
- **hybrid**: change_rate=0.031 CE=0.316  net_correction=+0.0015
  → Low CE (0.316): hybrid makes many harmful changes — the cache/statistics are corrupting more than they help.

### PETS
- **tda**: change_rate=0.040 CE=0.600  net_correction=+0.0125
- **freetta**: change_rate=0.016 CE=0.593  net_correction=+0.0055
- **conf_ftta**: change_rate=0.017 CE=0.516  net_correction=+0.0035
- **ent_tda**: change_rate=0.036 CE=0.591  net_correction=+0.0095
- **hybrid**: change_rate=0.020 CE=0.635  net_correction=+0.0068
## 3. Entropy / Confidence Insights

### CALTECH
- CLIP: confidence(correct)=0.922  confidence(wrong)=0.614  entropy(wrong)=1.089
- TDA: confidence(correct)=0.971  confidence(wrong)=0.767  entropy(wrong)=0.622
  → TDA produces overconfident wrong predictions (conf=0.767).  OER risk is elevated.
- FREETTA: confidence(correct)=0.932  confidence(wrong)=0.623  entropy(wrong)=1.033

### DTD
- CLIP: confidence(correct)=0.595  confidence(wrong)=0.323  entropy(wrong)=2.555
- TDA: confidence(correct)=0.717  confidence(wrong)=0.442  entropy(wrong)=2.087
- FREETTA: confidence(correct)=0.726  confidence(wrong)=0.454  entropy(wrong)=1.943

### EUROSAT
- CLIP: confidence(correct)=0.599  confidence(wrong)=0.335  entropy(wrong)=1.809
- TDA: confidence(correct)=0.775  confidence(wrong)=0.568  entropy(wrong)=1.196
- FREETTA: confidence(correct)=0.822  confidence(wrong)=0.615  entropy(wrong)=0.969

### IMAGENET
- CLIP: confidence(correct)=0.758  confidence(wrong)=0.471  entropy(wrong)=2.050
- TDA: confidence(correct)=0.828  confidence(wrong)=0.557  entropy(wrong)=1.828
- FREETTA: confidence(correct)=0.781  confidence(wrong)=0.501  entropy(wrong)=1.844

### PETS
- CLIP: confidence(correct)=0.875  confidence(wrong)=0.552  entropy(wrong)=1.133
- TDA: confidence(correct)=0.898  confidence(wrong)=0.594  entropy(wrong)=1.024
- FREETTA: confidence(correct)=0.887  confidence(wrong)=0.570  entropy(wrong)=1.066
## 4. Disagreement Analysis Insights

### CALTECH
- Disagreement rate: 0.017.  When they disagree: TDA wins 15 times, FreeTTA wins 23 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.

### DTD
- Disagreement rate: 0.144.  When they disagree: TDA wins 60 times, FreeTTA wins 58 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.

### EUROSAT
- Disagreement rate: 0.233.  When they disagree: TDA wins 746 times, FreeTTA wins 551 times.
  → High disagreement rate: the two methods respond very differently to the same inputs — their mechanisms are genuinely complementary.

### IMAGENET
- Disagreement rate: 0.060.  When they disagree: TDA wins 164 times, FreeTTA wins 143 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.

### PETS
- Disagreement rate: 0.037.  When they disagree: TDA wins 72 times, FreeTTA wins 46 times.
  → Low disagreement rate: TDA and FreeTTA make similar predictions, suggesting one method's decisions dominate the other.
## 5. Improvement Method Insights

### CALTECH
- ConfGatedFreeTTA vs FreeTTA: +0.0000 → **NEUTRAL**
- EntropyGatedTDA vs TDA: +0.0016 → **IMPROVED**
- Hybrid vs best(TDA, FreeTTA): +0.0012 → **IMPROVED**
  → Hybrid gains from combining local cache (short-range) and global statistics (domain shift).  The two mechanisms are complementary here.

### DTD
- ConfGatedFreeTTA vs FreeTTA: -0.0218 → **DEGRADED**
  → Gating the M-step hurt FreeTTA here.  Discarding uncertain updates removed too much signal, starving the class means of the adaptation they need.
- EntropyGatedTDA vs TDA: +0.0032 → **IMPROVED**
- Hybrid vs best(TDA, FreeTTA): -0.0128 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

### EUROSAT
- ConfGatedFreeTTA vs FreeTTA: -0.0110 → **DEGRADED**
  → Gating the M-step hurt FreeTTA here.  Discarding uncertain updates removed too much signal, starving the class means of the adaptation they need.
- EntropyGatedTDA vs TDA: +0.0011 → **IMPROVED**
- Hybrid vs best(TDA, FreeTTA): -0.0059 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

### IMAGENET
- ConfGatedFreeTTA vs FreeTTA: -0.0014 → **DEGRADED**
- EntropyGatedTDA vs TDA: +0.0009 → **NEUTRAL**
- Hybrid vs best(TDA, FreeTTA): -0.0039 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

### PETS
- ConfGatedFreeTTA vs FreeTTA: -0.0019 → **DEGRADED**
- EntropyGatedTDA vs TDA: -0.0030 → **DEGRADED**
  → Adaptive entropy gating shrank the cache too aggressively.  The original fixed threshold already selects good samples.
- Hybrid vs best(TDA, FreeTTA): -0.0057 → **DEGRADED**
  → Combining local and global signals did not help.  When both components contribute noise, the hybrid inherits both errors.  The optimal weighting is likely dataset-specific.

## 6. Novel Metric Commentary (Section 10)

### Stability Scores
| dataset   |   clip_stability |   tda_stability |   freetta_stability |   conf_ftta_stability |   ent_tda_stability |   hybrid_stability |
|:----------|-----------------:|----------------:|--------------------:|----------------------:|--------------------:|-------------------:|
| caltech   |         0.96809  |        0.96814  |            0.968993 |              0.968993 |            0.9688   |           0.968341 |
| dtd       |         0.93398  |        0.929607 |            0.921535 |              0.921109 |            0.929559 |           0.934345 |
| eurosat   |         0.93648  |        0.928755 |            0.926563 |              0.928368 |            0.930053 |           0.930566 |
| imagenet  |         0.939155 |        0.938639 |            0.939833 |              0.939843 |            0.939236 |           0.939787 |
| pets      |         0.954016 |        0.959068 |            0.954085 |              0.953205 |            0.958266 |           0.956402 |
- caltech: most stable method = **freetta**
- dtd: most stable method = **clip**
- eurosat: most stable method = **clip**
- imagenet: most stable method = **freetta**
- pets: most stable method = **tda**

### Overconfidence Error Rate (OER > 0.1 highlighted)
| dataset   | method    |      oer |
|:----------|:----------|---------:|
| caltech   | tda       | 0.358025 |
| caltech   | ent_tda   | 0.379747 |
| caltech   | hybrid    | 0.192053 |
| eurosat   | freetta   | 0.135235 |
| eurosat   | conf_ftta | 0.105672 |
| imagenet  | tda       | 0.112369 |
| imagenet  | ent_tda   | 0.112642 |
High OER entries indicate methods that produce confident wrong predictions.  This is a calibration failure and is dangerous in deployment.

## 7. Summary Hypotheses
Based on the above analysis, the following hypotheses are proposed:

1. **FreeTTA effectiveness scales with domain shift magnitude**: datasets with larger domain gap from ImageNet training benefit more from global statistics adaptation (EuroSAT > DTD > Caltech).

2. **TDA cache saturation limits long-stream gains**: the fixed capacity (pos_shot_capacity * C slots) caps how much the cache can represent.  On large-class datasets (ImageNet, 1000 classes) the cache is too sparse per class to help much.

3. **Confidence gating is a double-edged sword**: on datasets where CLIP is uncertain frequently (DTD textures), gating the M-step removes most of the adaptation signal and hurts FreeTTA.  On datasets where CLIP is mostly confident (Caltech), gating is nearly neutral.

4. **Hybrid gains require the two methods to be complementary**: when TDA and FreeTTA disagree frequently the hybrid can benefit from both.  When they agree, combining them adds noise without signal.