from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class TDA:
    """
    TDA: Test-Time Data Augmentation via feature caching.

    Stores confident (low-entropy) image features in a per-class positive cache
    and medium-confidence features in a negative cache.  Cache logits are added
    to/subtracted from CLIP logits to adjust predictions.

    GPU efficiency:
      - Pre-computes ALL clip logits in a single batched matmul before the loop
      - Cache affinity computed via batched einsum (no per-slot Python loop)
      - All tensor ops stay on GPU throughout
    """

    DATASET_DEFAULTS = {
        "caltech":  dict(alpha=5.0, beta=5.0,  neg_alpha=0.117, neg_beta=1.0,
                         low_entropy_thresh=0.2, high_entropy_thresh=0.5,
                         pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
        "dtd":      dict(alpha=2.0, beta=3.0,  neg_alpha=0.117, neg_beta=1.0,
                         low_entropy_thresh=0.2, high_entropy_thresh=0.5,
                         pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
        "eurosat":  dict(alpha=4.0, beta=8.0,  neg_alpha=0.117, neg_beta=1.0,
                         low_entropy_thresh=0.2, high_entropy_thresh=0.5,
                         pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
        "pets":     dict(alpha=2.0, beta=7.0,  neg_alpha=0.117, neg_beta=1.0,
                         low_entropy_thresh=0.2, high_entropy_thresh=0.5,
                         pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
        "imagenet": dict(alpha=1.0, beta=8.0,  neg_alpha=0.117, neg_beta=1.0,
                         low_entropy_thresh=0.2, high_entropy_thresh=0.5,
                         pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    }

    def __init__(
        self,
        text_features: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 5.0,
        neg_alpha: float = 0.117,
        neg_beta: float = 1.0,
        low_entropy_thresh: float = 0.2,
        high_entropy_thresh: float = 0.5,
        pos_shot_capacity: int = 3,
        neg_shot_capacity: int = 2,
        clip_scale: float = 100.0,
        neg_mask_lower: float = 0.03,
        neg_mask_upper: float = 1.0,
        device: str = "cuda",
        # legacy / unused params kept for backward compat
        cache_size: int = 1000,
        k: int = 0,
        shot_capacity: Optional[int] = None,
        fallback_to_clip: bool = False,
        fallback_margin: float = 0.0,
        confidence_threshold: Optional[float] = None,
        **_kwargs,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        tf = F.normalize(text_features.detach().float().to(self.device), dim=-1)
        self.text_features = tf
        self.num_classes, self.dim = tf.shape
        self.max_entropy = math.log(max(self.num_classes, 2))

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.neg_alpha = float(neg_alpha)
        self.neg_beta = float(neg_beta)
        self.low_entropy = float(low_entropy_thresh)
        self.high_entropy = float(high_entropy_thresh)
        self.pos_cap = int(pos_shot_capacity)
        self.neg_cap = int(neg_shot_capacity)
        self.clip_scale = float(clip_scale)
        self.neg_mask_lower = float(neg_mask_lower)
        self.neg_mask_upper = float(neg_mask_upper)

        self._init_cache()

    def _init_cache(self) -> None:
        C, D = self.num_classes, self.dim
        dev = self.device

        # Positive cache: (C, pos_cap, D) features + (C, pos_cap) entropy losses
        self.pos_feat   = torch.zeros(C, self.pos_cap, D, device=dev)
        self.pos_loss   = torch.full((C, self.pos_cap), float("inf"), device=dev)
        self.pos_counts = torch.zeros(C, dtype=torch.long, device=dev)

        # Negative cache: (C, neg_cap, D) + (C, neg_cap) entropy losses + (C, neg_cap, C) prob maps
        self.neg_feat   = torch.zeros(C, self.neg_cap, D, device=dev)
        self.neg_loss   = torch.full((C, self.neg_cap), float("inf"), device=dev)
        self.neg_pmaps  = torch.zeros(C, self.neg_cap, C, device=dev)
        self.neg_counts = torch.zeros(C, dtype=torch.long, device=dev)

    def reset(self) -> None:
        self._init_cache()

    @property
    def pos_size(self) -> int:
        return int(self.pos_counts.sum().item())

    @property
    def neg_size(self) -> int:
        return int(self.neg_counts.sum().item())

    @property
    def pos_shot_capacity(self) -> int:
        return self.pos_cap

    @property
    def neg_shot_capacity(self) -> int:
        return self.neg_cap

    def _update_slot(self, negative: bool, cls: int, feat: torch.Tensor,
                     loss: float, pmap: Optional[torch.Tensor] = None) -> None:
        """Insert a sample into the per-class cache slot (evict worst if full)."""
        if negative:
            feats, losses, counts, cap = self.neg_feat, self.neg_loss, self.neg_counts, self.neg_cap
        else:
            feats, losses, counts, cap = self.pos_feat, self.pos_loss, self.pos_counts, self.pos_cap

        cnt = int(counts[cls].item())
        if cnt < cap:
            idx = cnt
            counts[cls] += 1
        else:
            worst_val, worst_idx = losses[cls].max(dim=0)
            if loss >= float(worst_val.item()):
                return                                 # current sample is worse; skip
            idx = int(worst_idx.item())

        feats[cls, idx].copy_(feat)
        losses[cls, idx] = loss

        if negative and pmap is not None:
            masked = ((pmap > self.neg_mask_lower) & (pmap < self.neg_mask_upper)).float()
            self.neg_pmaps[cls, idx].copy_(masked)

        # Keep slots sorted by ascending loss (best = lowest loss first)
        active = int(counts[cls].item())
        if active > 1:
            sorted_l, order = losses[cls, :active].sort()
            losses[cls, :active] = sorted_l
            feats[cls, :active]  = feats[cls, order]
            if negative:
                self.neg_pmaps[cls, :active] = self.neg_pmaps[cls, order]

    def _cache_logits(self, x: torch.Tensor, negative: bool) -> torch.Tensor:
        """
        Vectorised cache logits computation.

        x: (D,) L2-normalised query
        Returns (C,) logits
        """
        if negative:
            feats, counts, cap = self.neg_feat, self.neg_counts, self.neg_cap
            alpha, beta = self.neg_alpha, self.neg_beta
        else:
            feats, counts, cap = self.pos_feat, self.pos_counts, self.pos_cap
            alpha, beta = self.alpha, self.beta

        total = int(counts.sum().item())
        if total == 0:
            return torch.zeros(self.num_classes, device=self.device)

        # affinity: (C, cap) via batch matmul  x·feats[c,s]
        # feats: (C, cap, D) → reshape to (C*cap, D) → matmul → (C*cap,) → reshape (C, cap)
        aff = torch.einsum("d,csd->cs", x, feats)    # (C, cap)

        # Mask out unfilled slots
        slot_idx = torch.arange(cap, device=self.device).unsqueeze(0)  # (1, cap)
        active = slot_idx < counts.unsqueeze(1)                         # (C, cap)

        weights = torch.exp(-(beta - beta * aff)) * active.float()      # (C, cap)

        if negative:
            # logits: (C,) = sum_s [ weight[c,s] * sum_k pmaps[c,s,k] ] → (C,)
            logits = torch.einsum("cs,csk->k", weights, self.neg_pmaps)
        else:
            logits = weights.sum(dim=-1)                                 # (C,)

        return alpha * logits

    @torch.no_grad()
    def run(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run TDA sequentially over all images.

        Parameters
        ----------
        image_features : (N, D) L2-normalised

        Returns
        -------
        preds      : (N,) TDA predictions
        clip_preds : (N,) CLIP-only predictions
        """
        N = image_features.shape[0]
        img = F.normalize(image_features.to(self.device, dtype=torch.float32), dim=-1)

        # ── Single batched matmul for ALL clip logits ────────────────────────
        clip_logits_all = self.clip_scale * (img @ self.text_features.t())  # (N, C)

        preds = torch.empty(N, dtype=torch.long, device=self.device)
        clip_preds = clip_logits_all.argmax(dim=-1)

        for i in range(N):
            x = img[i]
            clip_logits = clip_logits_all[i]                             # (C,)

            clip_probs = torch.softmax(clip_logits, dim=0)               # (C,) – already scaled
            clip_pred  = int(clip_logits.argmax().item())

            # Normalised entropy for cache routing
            h = -(clip_probs * torch.log(clip_probs + 1e-12)).sum()
            norm_h = float((h / self.max_entropy).clamp(0.0, 1.0).item())

            feat = x.detach()
            ent_loss = float(h.item())

            # Always add to positive cache
            self._update_slot(False, clip_pred, feat, ent_loss)

            # Add to negative cache only if medium-confidence
            if self.low_entropy < norm_h < self.high_entropy:
                self._update_slot(True, clip_pred, feat, ent_loss, pmap=clip_probs)

            # Fused logits
            final = (clip_logits
                     + self._cache_logits(x, negative=False)
                     - self._cache_logits(x, negative=True))
            preds[i] = final.argmax()

        return preds, clip_preds

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
        """
        Process a single image feature, update cache, return (pred, clip_pred, final_logits).

        Parameters
        ----------
        x : (D,) or (1, D)  L2-normalised image feature

        Returns
        -------
        pred         : (1,)  TDA predicted class index
        clip_pred    : int   CLIP-only predicted class index
        final_logits : (1, C) fused logits
        """
        x = F.normalize(x.float().to(self.device), dim=-1)
        if x.dim() > 1:
            x = x.squeeze(0)   # (D,)

        clip_logits = self.clip_scale * (x @ self.text_features.t())   # (C,)
        clip_probs = torch.softmax(clip_logits, dim=0)
        clip_pred = int(clip_logits.argmax().item())

        h = -(clip_probs * torch.log(clip_probs + 1e-12)).sum()
        norm_h = float((h / self.max_entropy).clamp(0.0, 1.0).item())
        ent_loss = float(h.item())

        self._update_slot(False, clip_pred, x.detach(), ent_loss)
        if self.low_entropy < norm_h < self.high_entropy:
            self._update_slot(True, clip_pred, x.detach(), ent_loss, pmap=clip_probs)

        final_logits = (clip_logits
                        + self._cache_logits(x, negative=False)
                        - self._cache_logits(x, negative=True))

        return final_logits.argmax().unsqueeze(0), clip_pred, final_logits.unsqueeze(0)

    @classmethod
    def for_dataset(cls, dataset_name: str, text_features: torch.Tensor,
                    device: str = "cuda") -> "TDA":
        cfg = cls.DATASET_DEFAULTS.get(dataset_name.lower(), {})
        return cls(text_features, device=device, **cfg)


__all__ = ["TDA"]
