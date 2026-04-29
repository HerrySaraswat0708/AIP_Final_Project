from __future__ import annotations

DEFAULT_DATASETS = ("caltech", "dtd", "eurosat", "pets", "imagenet")


PAPER_TDA_TARGETS = {
    "vit_b16": {
        "caltech": 94.24,
        "dtd": 47.40,
        "eurosat": 58.00,
        "pets": 88.63,
        # This repo evaluates "imagenet" on ImageNetV2 matched-frequency.
        "imagenet": 64.67,
    },
}


PAPER_FREETTA_TARGETS = {
    "vit_b16": {
        "caltech": 94.63,
        "dtd": 46.96,
        "eurosat": 62.93,
        "pets": 90.11,
        # This repo evaluates "imagenet" on ImageNetV2 matched-frequency.
        "imagenet": 64.92,
    },
    "rn50": {
        "caltech": 90.12,
        "dtd": 44.21,
        "eurosat": 43.64,
        "pets": 86.44,
    },
}


PAPER_TDA_DEFAULTS = {
    "dtd": {
        "cache_size": 1000,
        "shot_capacity": 5,
        "pos_shot_capacity": 5,
        "neg_shot_capacity": 2,
        "k": 0,
        "alpha": 4.0,
        "beta": 4.5,
        "low_entropy_thresh": 0.05,
        "high_entropy_thresh": 0.4,
        "neg_alpha": 0.05,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": False,
        "fallback_margin": 0.0,
    },
    "caltech": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "pos_shot_capacity": 3,
        "neg_shot_capacity": 2,
        "k": 0,
        "alpha": 5.0,
        "beta": 5.0,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.117,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": False,
        "fallback_margin": 0.0,
    },
    "eurosat": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "pos_shot_capacity": 3,
        "neg_shot_capacity": 2,
        "k": 0,
        "alpha": 4.0,
        "beta": 8.0,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.117,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": False,
        "fallback_margin": 0.0,
    },
    "pets": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "pos_shot_capacity": 3,
        "neg_shot_capacity": 2,
        "k": 0,
        "alpha": 2.0,
        "beta": 7.0,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.117,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": False,
        "fallback_margin": 0.0,
    },
    "imagenet": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "pos_shot_capacity": 3,
        "neg_shot_capacity": 2,
        "k": 0,
        "alpha": 1.0,
        "beta": 8.0,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.117,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": False,
        "fallback_margin": 0.0,
    },
}


DEFAULT_FREETTA_PARAMS = {
    # Per-dataset tuned configs for FreeTTA (tuned on GPU via batch-EM sweep + exact verification).
    # alpha: generative influence (higher → more reliance on adapted means).
    # beta:  entropy gating strength with normalised entropy ∈ [0,1].
    #        Higher beta → only very confident samples update the means.
    "caltech":  {"alpha": 0.02,  "beta": 3.0},   # conservative; CLIP already 93.5%
    "dtd":      {"alpha": 0.1,   "beta": 3.0},   # DTD: conservative; TDA wins (paper: TDA=47.40 > FreeTTA=46.96)
    "eurosat":  {"alpha": 0.8,   "beta": 3.0},   # 10 classes, big domain shift → stronger adapt
    "pets":     {"alpha": 0.25,  "beta": 4.0},   # 37 classes, tuned → 88.96% > TDA 88.4%
    "imagenet": {"alpha": 0.05,  "beta": 4.0},   # 1000 classes, very conservative
}
