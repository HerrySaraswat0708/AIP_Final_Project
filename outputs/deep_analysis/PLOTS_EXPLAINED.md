# Plot & Result Explanations
## Every figure in `outputs/deep_analysis/` explained

---

## Standard Required Plots

---

### `accuracy_vs_samples.png`

**What it shows:**
Rolling accuracy (window = ~2.5% of stream) for CLIP (grey), TDA (blue), and FreeTTA (red) as a function of the sample index, for all 5 datasets side by side.

**How to read it:**
- X-axis: sample index (0 to N). Each point is a rolling window average of the last W predictions.
- Y-axis: accuracy (%) within that rolling window.
- A flat line at CLIP's level means the method isn't adapting.
- A rising line means the method is learning over time.

**What we see:**
- **EuroSAT**: FreeTTA rises steeply and consistently from ~48% to ~80% by the end. TDA rises too but slower and peaks lower (~57%). CLIP stays flat at 48%.
- **DTD**: FreeTTA climbs faster than TDA in early samples, then both plateau around the same level.
- **Caltech, Pets, ImageNet**: All three methods are nearly flat and overlapping — CLIP is already near-optimal, little room for improvement.

**Why it matters:**
This plot shows that FreeTTA's improvement is not just from final accuracy — it comes from continuous, cumulative adaptation. TDA's early saturation is visible in EuroSAT where the blue line flattens well before the red line.

---

### `change_rate_vs_accuracy.png`

**What it shows:**
A scatter plot with change rate (% of CLIP predictions modified) on X-axis and accuracy on Y-axis. One point per dataset, separately for TDA (left panel) and FreeTTA (right panel).

**How to read it:**
- Points near the top-right mean: high change rate AND high accuracy — the changes were mostly beneficial.
- Points near the top-left mean: low change rate but high accuracy — changes were rare but precise.
- There is no expected linear relationship — this plot tests whether "changing more" equals "doing better."

**What we see:**
- EuroSAT (the outlier): TDA changes 41.9% of predictions and gets 53.3%; FreeTTA changes only 31% and gets 59.4% — FreeTTA is more accurate *with fewer changes*.
- ImageNet/Caltech: low change rate, moderate accuracy — both methods are conservative.
- No positive correlation between change rate and accuracy is visible.

**Why it matters:**
This validates that **change rate alone is a misleading metric**. High change rate can mean the method is randomly "exploring" rather than confidently correcting. Beneficial Flip Precision (BFP) is the better signal.

---

### `bfp_vs_thresholds.png`

**What it shows:**
Grouped bar chart showing Beneficial Flip Precision (BFP) for TDA (blue) and FreeTTA (red) across all 5 datasets.

**BFP formula:**
BFP = beneficial_flips / (beneficial_flips + harmful_flips)

A flip is beneficial if the prediction changes from wrong to right. Harmful if it changes from right to wrong.

**How to read it:**
- BFP = 100%: every prediction change is a fix. Perfect.
- BFP = 50%: random — as many harms as fixes.
- BFP < 50%: harmful — the method is making things worse more often than better.

**What we see:**
- EuroSAT FreeTTA = 77.1%: 3 out of every 4 changes FreeTTA makes are fixes.
- EuroSAT TDA = 60.7%: still positive but lower.
- Pets TDA = 67.7% (slight TDA advantage on a low-shift dataset).
- ImageNet: nearly tied (59.6% vs 59.5%).
- No dataset has BFP below 50% — both methods never cause net harm in aggregate.

**Why it matters:**
BFP directly measures the *quality* of each method's decisions. It correctly predicts the winner (the method with higher BFP wins accuracy) in 4 out of 5 datasets.

---

### `entropy_confidence_plots.png`

**What it shows:**
Two rows of 5 plots each:
- **Top row**: Overlapping histograms of entropy for CLIP (grey), TDA (blue), FreeTTA (red) predictions.
- **Bottom row**: Scatter of CLIP confidence (x) vs whether the CLIP prediction was correct (0/1, y).

**How to read it:**
- Top row: CLIP entropy is far right (near log(C), maximally uncertain). After adaptation, TDA and FreeTTA entropy distributions shift left (more confident).
- Bottom row: Points at the bottom are CLIP errors. Points clustered at low confidence = most errors happen when CLIP is uncertain.

**What we see:**
- **Caltech/Pets/ImageNet**: CLIP entropy is tightly packed near log(C) — uniform. TDA and FreeTTA show bimodal distributions (very confident on easy samples, uncertain on hard ones).
- **EuroSAT**: CLIP has a wider entropy range (some confident, some not). Both methods shift it left significantly.
- Bottom scatter: EuroSAT shows a clear negative slope — lower confidence → more errors. Caltech shows almost no slope (errors are not predicted by confidence).

**Why it matters:**
The top row explains why TDA's negative cache gate fires 0% — CLIP's raw entropy is always near log(C), so the gate condition (0.2 < H_norm < 0.5) is never met. The bottom scatter explains when entropy is a useful signal.

---

### `break_even_plots.png`

**What it shows:**
Bar chart showing the sample index (absolute number) at which each method first achieves net positive cumulative accuracy gain over CLIP. Two bars per dataset: TDA (blue) and FreeTTA (red).

**How to read it:**
- Low bar = method starts helping early (good).
- High bar close to N = method only helps near the end (bad — you're almost done by then).
- If bar = N, method never breaks even.

**What we see:**
- **EuroSAT**: TDA breaks even at 7,224 out of 8,100 samples (89% into the stream). FreeTTA breaks even at sample 7. This 1000× difference is the starkest result.
- **DTD**: FreeTTA at 29, TDA at 443.
- **ImageNet**: FreeTTA at 1, TDA at 362.
- **Pets**: Both at sample 4.
- **Caltech**: TDA never meaningfully breaks even (2,304 of 2,465).

**Why it matters:**
Break-even timing answers: "How much of the stream do I need to see before this method starts helping?" TDA on EuroSAT needs 89% of the data before it helps — making it practically useless for online deployment where you may not have 7,000 samples.

---

### `disagreement_analysis.png`

**What it shows:**
Two panels:
- **Left**: Disagreement rate (%) — how often TDA and FreeTTA predict different classes.
- **Right**: Among the samples where they disagree, what fraction does TDA win vs FreeTTA win.

**How to read it:**
- Low disagreement = both methods behave similarly.
- High disagreement + FreeTTA wins more = FreeTTA is meaningfully better in those cases.

**What we see:**
- EuroSAT: 28% disagreement — the highest. FreeTTA wins ~63% of disagreements, TDA wins 37%.
- Caltech: 0.8% disagreement — methods are essentially identical.
- ImageNet/Pets: TDA wins slightly more of its disagreements.

**Why it matters:**
High disagreement on EuroSAT confirms the methods are fundamentally diverging in behavior (not just marginally different). The fact that FreeTTA wins more of these disagreements directly explains the accuracy gap.

---

### `failure_buckets.png`

**What it shows:**
Stacked bar chart where each bar is a dataset. Colors represent the fraction of samples in each failure category:
- Green: All-correct (CLIP + TDA + FreeTTA all right)
- Shades of red/blue: Various mixed failure buckets
- Dark: All-wrong (all three models fail)

**Bucket definitions:**
- `CLIP✗ TDA✗ FT✓`: FreeTTA uniquely rescues
- `CLIP✗ TDA✓ FT✗`: TDA uniquely rescues
- `CLIP✓ TDA✗ FT✓`: TDA uniquely harms
- `CLIP✓ TDA✓ FT✗`: FreeTTA uniquely harms
- All-wrong: No method works

**What we see:**
- All datasets: 80–90% of samples are all-correct.
- EuroSAT has the largest all-wrong fraction (13.3%) — genuinely hard satellite images.
- FreeTTA unique rescues (0.45%) > TDA unique rescues (0.28%) across all datasets.
- TDA unique harms (0.36%) > FreeTTA unique harms (0.22%).

**Why it matters:**
The all-wrong category is the most revealing: it is nearly identical across TDA and FreeTTA, proving both methods fail on the *same* hard samples — a fundamental CLIP limitation, not an adaptation failure.

---

### `gas_vs_performance.png`

**What it shows:**
Two scatter plots:
- **Left**: GAS (%) on X-axis vs accuracy gain over CLIP on Y-axis. Two point series: TDA (blue squares) and FreeTTA (red circles). One point per dataset.
- **Right**: GAS (%) on X-axis vs FreeTTA_gain − TDA_gain on Y-axis.

**How to read it:**
- If GAS correlates with FreeTTA advantage: points in right panel should follow a positive slope.
- If GAS fails: points deviate (EuroSAT should stand out).

**What we see:**
- Left: EuroSAT has the highest FreeTTA gain (10.9%) despite having negative GAS (−11.5%).
- Right: EuroSAT is an outlier — GAS is negative but FreeTTA advantage is +6%.
- Other datasets roughly follow the expected positive relationship.

**Why it matters:**
GAS is a useful predictor but not infallible. It fails when domain shift is extreme — the EM updates can escape the geometry constraint captured by GAS.

---

### `cache_pressure_plots.png`

**What it shows:**
Two line plots (Caltech and EuroSAT) showing TDA's positive cache size (blue) and negative cache size (orange) growing over the sample stream.

**How to read it:**
- X-axis: sample index. Y-axis: number of cached entries.
- Both lines grow from 0 and plateau when the cache is full.
- The plateau height = C × K (positive) or C × K_neg (negative).

**What we see:**
- **Caltech** (C=100, K=3): Positive cache reaches 300 entries within ~500 samples (20% of stream). After that, only eviction-replacement.
- **EuroSAT** (C=10, K=3): Cache fills in ~30 samples (0.37% of stream). Extremely fast saturation.
- Negative cache is much smaller and grows slowly (orange line stays low).

**Why it matters:**
Cache saturation is the root cause of TDA's break-even problem. On EuroSAT, the cache fills in 30 samples — meaning TDA has no new "slots" to learn from samples 31–8,100. FreeTTA has no such limit.

---

### `em_weight_analysis.png`

**What it shows:**
For each of the 5 datasets, a scatter plot of the FreeTTA EM weight (α_t = exp(−β·H_norm)) on Y-axis vs sample index on X-axis.

**How to read it:**
- α_t close to 1: highly confident sample, large update to class means.
- α_t close to 0: very uncertain sample, tiny update (soft gating).
- Stable horizontal band = consistent confidence throughout stream.
- Downward trend = model becomes more uncertain over time.

**What we see:**
- **Caltech**: α_t tightly clustered near 0.8–0.9. Very confident throughout — makes sense (CLIP is already good on Caltech).
- **EuroSAT**: α_t scattered widely 0–0.5. Low average (0.20) — inputs are uncertain, but positive weights accumulate over 8,100 samples.
- **DTD**: Intermediate spread with mean ~0.46.

**Why it matters:**
This plot validates FreeTTA's soft gating. Even on EuroSAT where α_t is low (~0.2), the cumulative product of 8,100 small updates significantly shifts the centroids. Compare with TDA's neg-cache gate which fires 0 times.

---

## Section-Specific Plots

---

### `sec1_all_metrics.png`

**What it shows:**
A 2×5 grid of bar charts, one chart per metric, showing values for TDA and FreeTTA across all 5 datasets. Covers all 10 core metrics: accuracy, change rate, BFP, entropy, gain, GAS, cache fill rate, mean EM weight, oracle accuracy probes, and a winner summary.

**How to read it:**
Each subplot is self-contained. The X-axis always shows the 5 datasets. The Y-axis units change per metric (%, fraction, nats).

**Key takeaways:**
- Accuracy: EuroSAT shows the biggest gap (FreeTTA 59.4% vs TDA 53.3%).
- BFP: FreeTTA is higher in 3/5 cases.
- GAS: EuroSAT uniquely has negative GAS (centroid geometry is weaker than instance similarity there).
- Cache fill: All datasets near 100% — cache always saturates.
- EM weight: EuroSAT has the lowest (0.20), Caltech the highest (0.83).
- Winner summary: FreeTTA wins 3, TDA wins 2 (Pets + ImageNet).

---

### `sec2_sample_grid.png`

**What it shows:**
A 2×5 grid. Top row: absolute accuracy (CLIP, TDA, FreeTTA) vs stream fraction (5%–100%). Bottom row: accuracy gain vs stream fraction. One column per dataset.

**How to read it:**
- X-axis: fraction of the stream used (5% to 100%), log-scaled.
- Top row: are the method lines diverging or converging as stream grows?
- Bottom row: is the gain increasing, decreasing, or flat with more data?

**What we see:**
- **EuroSAT top**: FreeTTA line rises noticeably from 5% to 100%; TDA rises slowly.
- **EuroSAT bottom**: FreeTTA gain goes from +5.1% to +10.9%; TDA from +2.9% to +4.9%. FreeTTA is the only method that keeps gaining throughout.
- **Caltech/Pets/ImageNet**: All lines nearly flat — more data doesn't help much because CLIP is already near-optimal.

**Why it matters:**
This is a controlled experiment — by subsampling the stream at different fractions, we simulate different dataset sizes. It shows that TDA's advantage plateaus around 50% stream while FreeTTA keeps improving.

---

### `sec3_adaptation_dynamics.png`

**What it shows:**
A 3×5 grid for all 5 datasets:
- **Row 1**: Rolling accuracy (CLIP, TDA, FreeTTA) — how does performance evolve?
- **Row 2**: Cumulative gain over CLIP — is the method net-positive?
- **Row 3**: Rolling change rate — how often is each method changing predictions?

**How to read it:**
- Row 2 above zero = net-positive adaptation. Below zero = method is hurting overall.
- The X-axis position where Row 2 first crosses zero = break-even point.

**What we see:**
- **EuroSAT Row 1**: FreeTTA rises smoothly; TDA rises but slower.
- **EuroSAT Row 2**: FreeTTA is positive from sample 7; TDA stays negative until sample 7224.
- **EuroSAT Row 3**: TDA change rate is high (41.9%) but many changes are harmful early on.
- **Caltech Row 2**: Both near zero throughout — barely any gain.

**Why it matters:**
The cumulative gain plot (Row 2) is the definitive picture of each method's value. It makes the break-even problem visible in a way that a single accuracy number cannot.

---

### `sec4_uncertainty_analysis.png`

**What it shows:**
A 3×5 grid:
- **Row 1**: Histogram of normalized CLIP entropy (H/log C) for each dataset. Vertical dashed lines at 0.33 and 0.67 divide low/mid/high entropy buckets.
- **Row 2**: Accuracy (CLIP, TDA, FreeTTA) at each entropy bucket — 3 points per method.
- **Row 3**: Histogram of FreeTTA EM weights (α_t).

**How to read it:**
- Row 1: Is entropy spread out or bunched near 1.0? Bunched near 1.0 = TDA gate permanently closed.
- Row 2: Does accuracy drop as entropy increases? Which method handles high-entropy samples better?
- Row 3: Is α_t bimodal (confident / uncertain) or uniform?

**What we see:**
- **Caltech/Pets Row 1**: Histogram is a narrow spike at H_norm = 1.0 — all samples are maximally uncertain. TDA's gate in (0.2, 0.5) range = 0% of samples.
- **EuroSAT Row 1**: Wider distribution (0.5–1.0 range). Some gate firings.
- **EuroSAT Row 2**: FreeTTA consistently higher than TDA in all 3 entropy buckets.
- **Row 3**: EuroSAT shows wide spread; Caltech clusters near 0.83.

---

### `sec5_distribution_modeling.png`

**What it shows:**
A 2×5 grid:
- **Row 1**: 2D PCA of CLIP logits, colored by true class label. Each point is a test sample.
- **Row 2**: Histogram of FreeTTA centroid drift (‖μ_t − μ_0‖) per sample.

**How to read it:**
- Row 1: Are classes well-separated in CLIP logit space? Tight clusters = CLIP is already discriminative. Overlapping = high uncertainty.
- Row 2: Large drift = FreeTTA moved the class centroids far from their text-embedding initialization. Concentrated near 0 = centroids barely moved.

**What we see:**
- **EuroSAT Row 1**: 10 classes are moderately separated in 2D — some overlap (explains the 13.3% all-wrong).
- **ImageNet Row 1**: 1000 classes compressed into 2D → no visible structure (PC1+PC2 explain <12% variance).
- **EuroSAT Row 2**: Drift peaks around 1.1–1.3 — centroids moved substantially from text embeddings.
- **Caltech Row 2**: Drift peaks around 0.8 — less movement because fewer training samples are needed to converge.

---

### `sec6_efficiency.png`

**What it shows:**
Three bar charts:
1. Break-even sample index (TDA vs FreeTTA, per dataset).
2. Break-even / stream length as a fraction (% of stream needed before payoff).
3. Estimated memory usage in KB (TDA vs FreeTTA, per dataset).

**How to read it:**
- Panel 1: Lower bar = method pays off sooner.
- Panel 2: Same as Panel 1 but normalized. A bar at 89% means you need 89% of your test data before the method helps.
- Panel 3: TDA bars are always taller (more memory) due to C × K × D cache vs FreeTTA's C × D means.

**What we see:**
- FreeTTA bars in panels 1 & 2 are uniformly shorter.
- Memory (Panel 3): On ImageNet, TDA requires ~30 MB (C=1000, K=5, D=512) vs FreeTTA ~2 MB.

---

### `sec7_architecture_analysis.png`

**What it shows:**
A 2×5 grid:
- **Row 1**: Binned EM weight (X) vs FreeTTA accuracy at that bin (Y). Tests if high-α_t samples are predicted more correctly.
- **Row 2**: Normalized TDA cache size (blue, left axis) and rolling TDA gain (orange, right axis) over the stream. Tests if cache growth correlates with TDA gain.

**How to read it:**
- Row 1: Positive slope = high confidence samples are more accurately predicted.
- Row 2: If the orange line rises when the blue line plateaus, TDA gain is NOT driven by continued learning — it's a one-time boost.

**What we see:**
- **Row 1 EuroSAT**: Clear positive slope — higher α_t → higher FreeTTA accuracy. EM weight is a valid quality proxy.
- **Row 2 EuroSAT**: Cache fills rapidly (blue plateaus early), TDA gain grows only slowly after that.
- **Row 1 ImageNet**: Flat — EM weight doesn't predict accuracy well when C=1000 (too dispersed).

---

### `sec8_confidence_subset.png`

**What it shows:**
A 2×5 grid where samples are split by CLIP confidence into low/mid/high percentile buckets:
- **Row 1**: Accuracy at each confidence bucket (CLIP, TDA, FreeTTA).
- **Row 2**: Accuracy gain at each confidence bucket (TDA vs FreeTTA vs CLIP baseline).

**How to read it:**
- Row 1: Does accuracy increase with confidence? Which method helps most in the low-confidence (hardest) bucket?
- Row 2: Bars above zero = method helps at this confidence level. Bars below zero = method hurts.

**What we see:**
- **EuroSAT**: FreeTTA positive in all 3 buckets; TDA also positive but smaller gains.
- **Pets low-confidence**: Both methods slightly hurt performance (TDA/FreeTTA changes make it worse for already-uncertain samples on fine-grained classes).
- **ImageNet**: Gains are largest in mid-confidence bucket for both methods.

---

### `sec9_spc_regime.png`

**What it shows:**
A 2×5 grid (log-scaled X-axis):
- **Row 1**: Gain over CLIP (TDA and FreeTTA) vs samples per class.
- **Row 2**: FreeTTA centroid drift vs samples per class.

**How to read it:**
- Row 1: Does the FreeTTA gain keep growing with more samples per class? Does TDA plateau?
- Row 2: Does drift increase with samples? (Expected: more samples → more centroid movement).

**What we see:**
- **EuroSAT Row 1**: FreeTTA gain rises from +5.1% (SPC=40) to +10.9% (SPC=810). TDA from +2.9% to +4.9%. FreeTTA's slope is steeper.
- **Row 2**: Drift increases monotonically with samples per class across all datasets.

---

### `sec10_initialization.png`

**What it shows:**
A 2×5 grid:
- **Row 1**: FreeTTA centroid drift (‖μ_t − μ_0‖) over the stream. Shows how far class means move from text embeddings.
- **Row 2**: FreeTTA prior entropy H(N_y) over the stream.

**How to read it:**
- Row 1: Monotonically increasing = FreeTTA always learning. Plateau = convergence reached.
- Row 2: Decreasing H(N_y) = model is becoming more "committed" to specific classes (soft counts accumulating).

**What we see:**
- **EuroSAT Row 1**: Drift keeps rising through the entire stream (never plateaus) — still learning at sample 8,100.
- **Caltech Row 1**: Rises then flattens around sample 500 — converged quickly (CLIP already good).
- **Row 2**: All datasets show monotonically decreasing prior entropy as N_y accumulates.

---

### `sec11_gas_validation.png`

**What it shows:**
Three panels:
1. **Scatter**: GAS (%) on X vs FreeTTA_gain − TDA_gain on Y. One point per dataset. Tests if GAS sign predicts FreeTTA advantage.
2. **Bar**: Oracle-Centroid vs Oracle-1NN accuracy per dataset. The difference = GAS.
3. **Bar**: GAS values per dataset (can be negative).

**How to read it:**
- Panel 1: Points above zero on Y = FreeTTA wins. If GAS > 0 correlates with Y > 0, GAS is a valid predictor.
- Panel 2: If oracle-centroid bar > oracle-1NN bar, centroid geometry dominates → FreeTTA structure is appropriate.
- Panel 3: Negative bar = instance similarity beats centroid means (EuroSAT).

**What we see:**
- Panel 1: EuroSAT is the outlier — GAS negative but FreeTTA wins strongly. Other 4 datasets roughly follow GAS sign.
- Panel 2: EuroSAT oracle-1NN (89.8%) > oracle-centroid (78.3%) by 11.5% — yet FreeTTA beats TDA.
- Panel 3: ImageNet has the largest positive GAS (+40.8%) but gains are modest (both tied at 62.72%).

---

### `sec12_failure_analysis.png`

**What it shows:**
Two panels:
1. **Grouped bar**: Failure rate (%) per bucket per dataset. 6 colored groups side by side.
2. **Stacked bar**: Same data stacked — shows composition of failures for each dataset.

**How to read it:**
- Panel 1: Compare bucket heights across datasets. EuroSAT should have taller "all-wrong" bars.
- Panel 2: What fraction of each dataset is in each bucket? Larger all-correct (green) = easier dataset.

**What we see:**
- EuroSAT: Largest all-wrong (13.3%) and largest FreeTTA-unique-rescue bars.
- Caltech: Nearly all all-correct (90.5%). Very small failure bars.
- ImageNet: Moderate all-wrong (6.0%). Both rescue rates are low (not much room to improve on a 62% baseline).

---

## Summary Table

| Plot | Core Question Answered |
|---|---|
| `accuracy_vs_samples.png` | Does performance improve over time? |
| `change_rate_vs_accuracy.png` | Is changing more predictions = better accuracy? |
| `bfp_vs_thresholds.png` | How often are changes beneficial vs harmful? |
| `entropy_confidence_plots.png` | How uncertain is CLIP, and does entropy predict errors? |
| `break_even_plots.png` | When does each method start paying off? |
| `disagreement_analysis.png` | Where do TDA and FreeTTA disagree, and who wins? |
| `failure_buckets.png` | What fraction of samples does each method uniquely rescue? |
| `gas_vs_performance.png` | Does geometry (GAS) predict the winner? |
| `cache_pressure_plots.png` | How quickly does TDA's cache saturate? |
| `em_weight_analysis.png` | How aggressive are FreeTTA's updates? |
| `sec1_all_metrics.png` | Holistic view of all 10 metrics at once |
| `sec2_sample_grid.png` | How does performance change with stream size? |
| `sec3_adaptation_dynamics.png` | Rolling accuracy + cumulative gain + change rate |
| `sec4_uncertainty_analysis.png` | Entropy buckets + EM weight distribution |
| `sec5_distribution_modeling.png` | Logit space geometry + centroid drift |
| `sec6_efficiency.png` | Memory + break-even efficiency |
| `sec7_architecture_analysis.png` | Mechanism correlation: cache growth vs TDA gain |
| `sec8_confidence_subset.png` | Confidence percentile breakdown |
| `sec9_spc_regime.png` | Samples-per-class scaling behavior |
| `sec10_initialization.png` | Convergence curves + prior entropy decay |
| `sec11_gas_validation.png` | GAS as a geometry-based predictor |
| `sec12_failure_analysis.png` | Failure bucket composition |
