# Novelty

## Overview

We propose **EdgeFreeTTA**, an efficient test-time adaptation framework that updates the model in a **compact low-dimensional space** instead of adapting the full parameter set.

The central idea is to keep the pretrained backbone largely frozen and perform adaptation through a **small low-rank module** that captures the correction needed under distribution shift. This makes adaptation substantially lighter than full-model updates while preserving the robustness of the original pretrained representation.

By restricting optimization to a low-rank adaptation space, EdgeFreeTTA reduces memory and compute overhead, avoids unnecessary parameter drift, and is better suited for **real-time or edge deployment** where latency and efficiency matter.

## Key Idea

- Freeze the main pretrained model to preserve its strong general representation.
- Insert a lightweight low-rank adaptation module to absorb target-domain shift.
- Update only this compact module during test-time adaptation.
- Achieve faster, cheaper adaptation with lower risk of overfitting than full-model updates.

## Motivation

Existing test-time adaptation methods often improve robustness at the cost of expensive parameter updates. That tradeoff is unattractive in resource-constrained settings. EdgeFreeTTA is designed to preserve adaptation quality while making the update path compact enough for deployment in streaming and low-latency environments.
