from __future__ import annotations

from .TDA_original import TDA as _OriginalTDA


class TDA(_OriginalTDA):
    """Compatibility wrapper with legacy argument aliases."""

    def __init__(
        self,
        text_features,
        cache_size: int = 1000,
        k: int = 0,
        alpha: float = 2.0,
        beta: float = 5.0,
        confidence_threshold: float | None = None,
        low_entropy_thresh: float = 0.2,
        high_entropy_thresh: float = 0.5,
        neg_alpha: float = 0.117,
        neg_beta: float = 1.0,
        neg_mask_lower: float = 0.03,
        neg_mask_upper: float = 1.0,
        shot_capacity: int = 3,
        clip_scale: float = 100.0,
        fallback_to_clip: bool = True,
        fallback_margin: float = 0.0,
        device: str = "cuda",
        **kwargs,
    ) -> None:
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
            clip_scale=clip_scale,
            fallback_to_clip=fallback_to_clip,
            fallback_margin=fallback_margin,
            device=device,
        )


__all__ = ["TDA"]
