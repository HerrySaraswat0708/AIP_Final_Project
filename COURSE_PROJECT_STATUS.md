# Course Project Status

This repository is now aligned to the **second project option** from the course brief.

## Selected Track

The selected submission track is:

> choose one recent paper and a stronger later method for the same problem, analyze why the later method should improve over the baseline, add new analysis not present in either paper, and experimentally analyze when the improvement occurs.

For this repository, the final project framing is:

- **baseline paper**: `TDA`
- **later comparative method**: `FreeTTA`
- **main project contribution**: a new mechanism-level analysis framework comparing how the two methods behave on the same frozen CLIP benchmark

`EdgeFreeTTA` remains in the repo as extra exploration, but it is **not** the main grading narrative for the selected option.

## What Is Already Covered

The repository already contains all core deliverables needed for option 2:

- a written comparative report:
  [ASSIGNMENT_REPORT.md](/home/herrys/projects/AIP-Final-project/ASSIGNMENT_REPORT.md)

- an overall project guide:
  [README.md](/home/herrys/projects/AIP-Final-project/README.md)

- formal comparison questions and metric definitions:
  [queries.md](/home/herrys/projects/AIP-Final-project/queries.md)

- comparison outputs and plots:
  [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis)

- final tuned full-dataset comparison:
  [outputs/final_method_suite](/home/herrys/projects/AIP-Final-project/outputs/final_method_suite)

- presentation-ready slide content:
  [PRESENTATION_OUTLINE.md](/home/herrys/projects/AIP-Final-project/PRESENTATION_OUTLINE.md)

## Final Story For Submission

The final viva/story should be:

1. `TDA` and `FreeTTA` solve the same frozen-CLIP test-time adaptation problem.
2. The papers suggest `FreeTTA` should often be stronger overall, but not uniformly on every dataset.
3. Our reproduction does not fully match that paper-level ordering.
4. That mismatch is the core research question addressed in this project.
5. We therefore compare the two methods not only by final accuracy, but also by:
   - geometry,
   - prediction flips,
   - disagreement cases,
   - adaptation trajectory,
   - entropy/statistics,
   - internal state evolution.

## Important Viva Positioning

Do **not** claim:

- that `FreeTTA` is literally an extension module built on top of `TDA`,
- or that we perfectly reproduced every paper number in the official paper setup.

Do claim:

- that we compared two recent methods for the same problem under one shared benchmark,
- that we checked `TDA` against the official public implementation logic,
- that we built a deeper analysis framework beyond either paper,
- and that we studied why the expected ordering can fail in reproduction.

## Optional Extra Work

The repo also contains:

- [novelty/README.md](/home/herrys/projects/AIP-Final-project/novelty/README.md)
- [models/EdgeFreeTTA.py](/home/herrys/projects/AIP-Final-project/models/EdgeFreeTTA.py)

This can be mentioned as additional exploration if useful, but it should not dominate the main project presentation.
