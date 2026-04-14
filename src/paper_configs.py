from __future__ import annotations

DEFAULT_DATASETS = ("caltech", "dtd", "eurosat", "pets", "imagenet", "imagenet64")


PAPER_TDA_TARGETS = {
    "vit_b16": {
        "caltech": 94.24,
        "dtd": 47.40,
        "eurosat": 58.00,
        "pets": 88.63,
        "imagenet": 69.51,
    },
}


PAPER_FREETTA_TARGETS = {
    "vit_b16": {
        "caltech": 94.63,
        "dtd": 46.96,
        "eurosat": 62.93,
        "pets": 90.11,
        "imagenet": 70.21,
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
        "k": 0,
        "alpha": 2.0,
        "beta": 3.0,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.05,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": True,
        "fallback_margin": 0.0,
    },
    "caltech": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 0.75,
        "beta": 1.5,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.0,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": True,
        "fallback_margin": 0.0,
    },
    "eurosat": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 1.45,
        "beta": 3.2,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.0,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": True,
        "fallback_margin": 0.0,
    },
    "pets": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 5.9,
        "beta": 8.9,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.32,
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
        "k": 0,
        "alpha": 2.0,
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
    "imagenet64": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 2.0,
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
}


DEFAULT_FREETTA_PARAMS = {
    "dtd": {"alpha": 0.2, "beta": 2.0},
    "caltech": {"alpha": 0.1, "beta": 1.0},
    "eurosat": {"alpha": 0.3, "beta": 4.5},
    "pets": {"alpha": 0.1, "beta": 0.1},
    "imagenet": {"alpha": 0.2, "beta": 4.5},
    "imagenet64": {"alpha": 0.2, "beta": 4.5},
}
