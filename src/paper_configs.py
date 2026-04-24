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
        "shot_capacity": 3,
        "pos_shot_capacity": 3,
        "neg_shot_capacity": 2,
        "k": 0,
        "alpha": 2.0,
        "beta": 3.0,
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
    "dtd": {"alpha": 0.2, "beta": 4.5},
    "caltech": {"alpha": 0.2, "beta": 4.5},
    "eurosat": {"alpha": 0.2, "beta": 4.5},
    "pets": {"alpha": 0.2, "beta": 4.5},
    "imagenet": {"alpha": 0.2, "beta": 4.5},
}
