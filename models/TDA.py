from typing import Optional

from .TDA_original import TDA as _OriginalTDA


class TDA(_OriginalTDA):
    """Compatibility wrapper with legacy argument aliases."""

    def __init__(
        self,
        text_features,
        cache_size=1000,
        k=0,
        alpha=2.0,
        beta=5.0,
        confidence_threshold=None,
        low_entropy_thresh=0.2,
        high_entropy_thresh=0.5,
        neg_alpha=0.117,
        neg_beta=1.0,
        neg_mask_lower=0.03,
        neg_mask_upper=1.0,
        shot_capacity=3,
        pos_shot_capacity: Optional[int] = None,
        neg_shot_capacity: Optional[int] = None,
        clip_scale=100.0,
        fallback_to_clip=False,
        fallback_margin=0.0,
        device="cuda",
        **kwargs,
    ):
        if confidence_threshold is not None:
            # Legacy mapping: higher confidence target -> lower normalized entropy threshold.
            low_entropy_thresh = max(0.0, min(1.0, 1.0 - float(confidence_threshold)))

        super().__init__(
            text_features=text_features,
            cache_size=cache_size,
            k=k,
            alpha=alpha,
            beta=beta,
            low_entropy_thresh=low_entropy_thresh,
            high_entropy_thresh=high_entropy_thresh,
            neg_alpha=neg_alpha,
            neg_beta=neg_beta,
            neg_mask_lower=neg_mask_lower,
            neg_mask_upper=neg_mask_upper,
            shot_capacity=shot_capacity,
            pos_shot_capacity=pos_shot_capacity,
            neg_shot_capacity=neg_shot_capacity,
            clip_scale=clip_scale,
            fallback_to_clip=fallback_to_clip,
            fallback_margin=fallback_margin,
            device=device,
        )


__all__ = ["TDA"]
