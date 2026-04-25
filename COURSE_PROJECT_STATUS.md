# Course Project Status

This repository is now aligned to **project option 2**.

## Selected Track

The project framing is:

- baseline paper: `TDA`
- later method: `FreeTTA`
- contribution: a deeper mechanism-level comparison on the same frozen CLIP benchmark

The repo is intentionally reduced to that story:

- `models/` only contains `TDA.py` and `FreeTTA.py`
- `experiments/` only contains the two tuners, the comparative runner, and the two plotting scripts
- `src/` is left intact as support code
- all generated artifacts live under `outputs/`

## Current Submission Story

The safe and defensible viva narrative is:

1. `TDA` and `FreeTTA` solve the same frozen-CLIP test-time adaptation problem.
2. Theory and paper results suggest `FreeTTA` should often be stronger overall.
3. Our local benchmark does not reproduce that ordering cleanly.
4. Instead of hiding that, we treat the gap itself as the main analysis problem.
5. We then compare the methods through prediction flips, entropy, trajectories, disagreement cases, internal states, and failure buckets.

## What Is Already Present

- report: [ASSIGNMENT_REPORT.md](/home/herrys/projects/AIP-Final-project/ASSIGNMENT_REPORT.md)
- repo guide: [README.md](/home/herrys/projects/AIP-Final-project/README.md)
- comparison questions: [queries.md](/home/herrys/projects/AIP-Final-project/queries.md)
- slide outline: [PRESENTATION_OUTLINE.md](/home/herrys/projects/AIP-Final-project/PRESENTATION_OUTLINE.md)
- deep analysis outputs: [outputs/comparative_analysis](/home/herrys/projects/AIP-Final-project/outputs/comparative_analysis)
- reproduction audit: [outputs/reproduction](/home/herrys/projects/AIP-Final-project/outputs/reproduction)
- tuned configs: [outputs/tuning](/home/herrys/projects/AIP-Final-project/outputs/tuning)

## Important Positioning

Do claim:

- the project compares two recent methods for the same problem under one shared benchmark
- `TDA` was checked against the official implementation logic
- `FreeTTA` was reimplemented and tuned from the paper details
- the main contribution is the deeper comparative analysis beyond final accuracy

Do not claim:

- perfect reproduction of every paper number
- that `FreeTTA` dominates `TDA` on this repo’s benchmark
- that there is an official public `FreeTTA` code match in this repo
