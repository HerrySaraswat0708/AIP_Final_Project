# Presentation Outline

This file contains slide-ready content for a **15-minute viva / 15-slide maximum** presentation.

## Slide 1: Title

**Comparative Analysis of TDA and FreeTTA for Test-Time Adaptation**

- Course project option 2
- Compare a recent paper and a later stronger method for the same problem
- Main contribution: new analysis beyond both papers

## Slide 2: Motivation

- CLIP works well zero-shot but suffers under distribution shift
- test-time adaptation tries to improve predictions without retraining on source data
- key question:
  which kind of adaptation is better for a frozen CLIP model?

## Slide 3: Objective

- compare `TDA` and `FreeTTA` under one shared benchmark
- reproduce their behavior on five datasets
- understand not only final accuracy, but also:
  - what they do to CLIP outputs
  - how they evolve through the stream
  - when one should theoretically beat the other

## Slide 4: Papers Chosen

- `TDA`: memory-based training-free adaptation
- `FreeTTA`: online EM-style global distribution modeling
- both solve the same frozen-CLIP TTA problem
- `FreeTTA` is not a literal extension of `TDA`, but a later stronger method for the same task

## Slide 5: Method 1 in Simple Words

**TDA**

- stores confident target examples in positive/negative caches
- changes CLIP logits using retrieval-style support and suppression
- good when local neighborhood structure is informative
- risk:
  limited memory, cache pressure, misleading neighbors

## Slide 6: Method 2 in Simple Words

**FreeTTA**

- maintains online class means, priors, and covariance
- combines CLIP logits with a generative score
- good when global class-level target drift is coherent
- risk:
  wrong distribution drift, poor Gaussian fit, slower adaptation

## Slide 7: Paper Expectation

- papers suggest `FreeTTA` should usually be competitive or stronger overall
- strongest expected positive case: `EuroSAT`
- exception case exists: `DTD`
- theory:
  global class statistics should help more when the stream is long and coherent

## Slide 8: Our Reproduced Results

Use:

- [outputs/final_method_suite/summary_table.csv](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite/summary_table.csv)
- [outputs/final_method_suite/overall_accuracy.png](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite/overall_accuracy.png)
- [outputs/presentation_figures/paper_vs_repo.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/paper_vs_repo.png)

Speaker note:

- our benchmark does not fully match the paper-level ordering
- `TDA` is stronger across all five datasets in this repo
- this mismatch becomes the research question of the project

## Slide 9: Our Main Contribution

We do not stop at the final accuracy table.

We add new comparisons beyond both papers:

- geometry alignment
- beneficial vs harmful flips
- disagreement reliability
- adaptation latency
- cache pressure
- internal-state diagnostics

## Slide 10: Geometry Comparison

Use:

- [outputs/comparative_analysis/geometry_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/geometry_analysis.png)

Key point:

- compare centroid-style structure vs local-neighbor structure
- this tests whether the benchmark geometry actually matches the theory behind each method

## Slide 11: What They Do To CLIP Outputs

Use:

- [outputs/comparative_analysis/flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)

Key point:

- TDA changes CLIP locally using retrieved neighbors
- FreeTTA changes CLIP globally through moving class statistics
- beneficial and harmful flips show whether those changes help or damage CLIP

## Slide 12: Trajectory Through the Stream

Use:

- [outputs/comparative_analysis/latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)
- and optionally one dataset plot from `outputs/final_method_suite/<dataset>/adaptation_dynamics.png`

Key point:

- TDA often helps earlier
- FreeTTA may need more evidence
- final accuracy alone hides this timing difference

## Slide 13: When They Disagree

Use:

- [outputs/comparative_analysis/disagreement_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_analysis.png)

Key point:

- disagreement samples are the most informative region
- if both methods agree, there is little to learn
- disagreement accuracy tells us whose inductive bias is better in contested cases

## Slide 14: Main Observations

- `TDA` is strongest overall in this reproduced benchmark
- `FreeTTA` still improves over CLIP on several datasets, but not enough to match paper-level expectations here
- the winner depends on:
  - local vs global geometry
  - stream length
  - headroom over CLIP
  - stability of the updated statistics

## Slide 15: Conclusion

- both methods modify frozen CLIP in fundamentally different ways
- `TDA`:
  local memory and fast correction
- `FreeTTA`:
  global distribution modeling and slower correction
- our main contribution is explaining **when and why** one method wins rather than giving only a final accuracy table

## Recommended Visuals

Use these first:

- [outputs/final_method_suite/overall_accuracy.png](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite/overall_accuracy.png)
- [outputs/comparative_analysis/geometry_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/geometry_analysis.png)
- [outputs/comparative_analysis/flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)
- [outputs/comparative_analysis/disagreement_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_analysis.png)
- [outputs/comparative_analysis/latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)

If you want more visuals, also use:

- [outputs/comparative_analysis/accuracy_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/accuracy_summary.png)
- [outputs/comparative_analysis/gain_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/gain_summary.png)
- [outputs/presentation_figures/gain_over_clip.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/gain_over_clip.png)
- [outputs/presentation_figures/clip_change_behavior.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/clip_change_behavior.png)
- [outputs/presentation_figures/internal_state_summary.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/internal_state_summary.png)
- any dataset-specific `adaptation_dynamics.png` under [outputs/final_method_suite](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite)
