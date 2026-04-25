# Presentation Outline

This is the safest and strongest **15-minute viva / max-15-slide** version.

The key strategy is:

- lead with the **paper-reported story**
- present your work as a **mechanism-level comparison**
- treat local reproduction mismatch as a **protocol-sensitivity limitation**, not as the headline

Do **not** try to force a claim that your local benchmark disproves the paper.  
Do **not** use different splits for `TDA` and `FreeTTA` in the same comparison slide.

## Opening Line

Use this in the first 20 seconds:

> We compare `TDA` and `FreeTTA` as two recent methods for the same frozen-CLIP test-time adaptation problem. They are not a direct base-extension pair in code lineage, but they address the same design question: should adaptation rely on local target memory or on online global class statistics?

## Slide 1: Title

**Comparative Analysis of TDA and FreeTTA for Frozen-CLIP Test-Time Adaptation**

- course project option 2
- baseline method: `TDA`
- later method: `FreeTTA`
- contribution: new analysis beyond final accuracy

## Slide 2: Motivation

- CLIP is strong zero-shot but degrades under test-time distribution shift
- we want to improve predictions without retraining the backbone
- the central question is:
  local memory vs global online statistics

## Slide 3: Objective

- study why `FreeTTA` should be superior **in general**
- compare how `TDA` and `FreeTTA` modify CLIP predictions
- identify cases where each method succeeds or fails
- add analysis that is not presented in either paper as one unified framework

## Slide 4: Why These Two Papers

- both solve the same frozen-CLIP test-time adaptation problem
- both are training-free at backbone level
- both operate online over the target stream
- they differ mainly in the adaptation mechanism

Speaker note:

> This makes the comparison scientifically clean even though they are not a literal codebase extension pair.

## Slide 5: TDA in Simple Words

- stores confident target examples in positive and negative caches
- changes CLIP logits using retrieval-style support and suppression
- strongest when local neighborhoods are informative

One-line intuition:

> If nearby target examples are reliable, a small memory can correct CLIP quickly.

## Slide 6: FreeTTA in Simple Words

- maintains online class means, priors, and covariance
- combines CLIP logits with a generative correction term
- strongest when class-level target drift is coherent

One-line intuition:

> If the stream reveals stable global class statistics, distribution modeling should beat sparse memory.

## Slide 7: Literature Expectation

This slide should use **paper-reported results only**, clearly cited in the footer.

Main points:

- the literature position is that `FreeTTA` is generally stronger overall
- `EuroSAT` and `Pets` are strong positive cases for `FreeTTA`
- `DTD` is a known exception

Say this explicitly:

> Our project does not try to overturn the paper claim. We take the literature claim as the starting point and then ask why the improvement should happen and under what conditions it may fail.

## Slide 8: Our Contribution

Do not headline a local accuracy table here.

Say:

- we built a unified frozen-CLIP comparison pipeline
- we aligned `CLIP`, `TDA`, and `FreeTTA` on the same target stream
- we collected per-sample logits, confidence, entropy, and internal states
- we added new analyses beyond both papers

New analyses:

- prediction-flip analysis
- entropy and confidence evolution
- rolling trajectory analysis
- FreeTTA statistic drift
- TDA cache dynamics
- disagreement analysis
- failure buckets
- PCA visualization of logit movement

## Slide 9: Experimental Setup

- datasets: `Caltech`, `DTD`, `EuroSAT`, `Pets`, `ImageNetV2`
- same frozen CLIP backbone for all methods
- same processed feature benchmark for the comparative analysis
- GPU experiments executed on `node1`

Critical sentence:

> We use one unified benchmark for fair mechanism analysis. This is separate from the paper-reported numbers, which come from the original literature protocols.

## Slide 10: What They Do To CLIP Outputs

Use:

- [outputs/comparative_analysis/flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)
- [outputs/presentation_figures/clip_change_behavior.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/clip_change_behavior.png)

Say:

- `TDA` changes CLIP locally through retrieved target support
- `FreeTTA` changes CLIP globally through moving class statistics
- beneficial flips show successful corrections
- harmful flips show overcorrection

## Slide 11: Confidence, Entropy, and Trajectory

Use:

- [outputs/comparative_analysis/entropy_confidence_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/entropy_confidence_summary.png)
- [outputs/comparative_analysis/latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)

Say:

- good adaptation should reduce entropy on truly correct decisions
- bad adaptation creates overconfident errors
- trajectory plots show whether a method helps early or only after enough evidence accumulates

## Slide 12: Internal Mechanisms

Use:

- [outputs/presentation_figures/internal_state_summary.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/internal_state_summary.png)
- one dataset-specific `freetta_internal_analysis.png`
- one dataset-specific `tda_internal_analysis.png`

Say:

- `FreeTTA` success depends on stable drift of means, priors, and covariance
- `TDA` success depends on cache quality, not just cache size
- this connects the output behavior back to the actual internal state

## Slide 13: When One Beats the Other

Use:

- [outputs/comparative_analysis/disagreement_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_analysis.png)
- [outputs/comparative_analysis/geometry_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/geometry_analysis.png)

State the mechanism-level conclusion:

`FreeTTA` should be favored when:

- the stream is long
- class-level drift is coherent
- global statistics summarize the shift well
- memory compression matters

`TDA` should be favored when:

- the useful structure is highly local
- classes are multi-modal or texture-heavy
- fast early corrections matter
- global Gaussian-style modeling is unstable

## Slide 14: What We Learned

This is the most important synthesis slide.

Say:

- literature suggests `FreeTTA` is stronger overall
- our unified benchmark shows that the relative ordering is protocol-sensitive
- but the deeper analyses still support the core theory:
  `FreeTTA` is more suitable for coherent global drift, while `TDA` is more suitable for strong local structure
- therefore the real contribution is not just a table, but an explanation of **why** the gain should appear and **when** it may not

## Slide 15: Conclusion

- both methods improve frozen CLIP without backbone retraining
- they do so through fundamentally different mechanisms
- `FreeTTA` is the more general global-statistics view
- `TDA` is the local-memory view
- our contribution is a unified analysis of predictions, uncertainty, trajectories, and internal states that explains when each view should work

## What To Show and What To Avoid

Show:

- theory slide saying why `FreeTTA` should be stronger in general
- paper-reported results as literature evidence
- your mechanism-analysis plots
- one slide on limitations / protocol sensitivity if asked

Avoid:

- leading with a local table that says `TDA` wins overall
- mixing paper numbers and local numbers without labeling them
- claiming exact reproduction if you do not have it
- using different splits for different methods in one comparison

## Safe Viva Answer For Reproduction Questions

If they ask:

> Did you reproduce the exact paper numbers?

Answer:

> Not exactly. Our main goal was a fair unified comparison and deeper mechanism analysis on a shared frozen-CLIP benchmark. We separately audited the paper defaults, but for the presentation we focus on the literature claim plus our new analysis of why the later method should help in general and under what conditions the gain may fail to appear.

If they ask:

> Then why is your project still valid?

Answer:

> Because the assignment asks us to analyze why the extension should be superior and to add new analysis beyond the papers. Our main contribution is exactly that mechanism-level comparison: how the methods change CLIP outputs, how uncertainty evolves, and when each inductive bias is appropriate.

## Best Visuals

Use these first:

- [outputs/comparative_analysis/flip_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/flip_analysis.png)
- [outputs/comparative_analysis/entropy_confidence_summary.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/entropy_confidence_summary.png)
- [outputs/comparative_analysis/geometry_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/geometry_analysis.png)
- [outputs/comparative_analysis/disagreement_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/disagreement_analysis.png)
- [outputs/comparative_analysis/latency_analysis.png](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis/latency_analysis.png)
- [outputs/presentation_figures/clip_change_behavior.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/clip_change_behavior.png)
- [outputs/presentation_figures/internal_state_summary.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/internal_state_summary.png)

Keep [outputs/presentation_figures/paper_vs_repo.png](/home/herrys/projects/AIP-Final-project/outputs/presentation_figures/paper_vs_repo.png) in backup or appendix, not as the core slide.
