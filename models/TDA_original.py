from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


CacheItem = Tuple[torch.Tensor, float, torch.Tensor | None]


class TDA:
    """
    Paper-faithful TDA with positive/negative per-class caches.

    Notes:
    - Positive cache is updated for every sample (then pruned by per-class shot capacity).
    - Negative cache is updated only for normalized entropy in (low_entropy_thresh, high_entropy_thresh).
    - CLIP logits are scaled by `clip_scale` (100.0 in the reference code).
    """

    def __init__(
        self,
        text_features: torch.Tensor,
        cache_size: int = 1000,  # legacy compatibility; shot_capacity controls behavior
        k: int = 0,  # optional top-k affinity truncation (0 => use all cache entries)
        alpha: float = 2.0,
        beta: float = 5.0,
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
    ) -> None:
        self.device = torch.device(device)
        self.text_features = F.normalize(text_features, dim=-1).to(self.device, dtype=torch.float32)
        self.num_classes = int(self.text_features.shape[0])
        self.dim = int(self.text_features.shape[1])

        self.cache_size = int(cache_size)
        self.shot_capacity = max(1, int(shot_capacity))
        self.k = max(0, int(k))

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.neg_alpha = float(neg_alpha)
        self.neg_beta = float(neg_beta)

        self.low_entropy = float(low_entropy_thresh)
        self.high_entropy = float(high_entropy_thresh)
        self.neg_mask_lower = float(neg_mask_lower)
        self.neg_mask_upper = float(neg_mask_upper)
        self.clip_scale = float(clip_scale)
        self.fallback_to_clip = bool(fallback_to_clip)
        self.fallback_margin = float(fallback_margin)

        self.max_entropy = math.log2(max(self.num_classes, 2))

        self.pos_cache: Dict[int, List[CacheItem]] = {}
        self.neg_cache: Dict[int, List[CacheItem]] = {}

        self.pos_size = 0
        self.neg_size = 0

    def _cache_len(self, cache: Dict[int, List[CacheItem]]) -> int:
        return sum(len(items) for items in cache.values())

    def _update_cache(
        self,
        cache: Dict[int, List[CacheItem]],
        pred: int,
        feature: torch.Tensor,
        entropy_loss: float,
        prob_map: torch.Tensor | None = None,
    ) -> None:
        item: CacheItem = (feature, entropy_loss, prob_map)
        bucket = cache.get(pred, [])
        if len(bucket) < self.shot_capacity:
            bucket.append(item)
        elif entropy_loss < bucket[-1][1]:
            bucket[-1] = item
        bucket.sort(key=lambda it: it[1])
        cache[pred] = bucket

    def _build_cache_tensors(
        self, cache: Dict[int, List[CacheItem]], negative: bool
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if not cache:
            return None, None

        cache_keys: List[torch.Tensor] = []
        cache_values: List[torch.Tensor] = []

        for class_index in sorted(cache.keys()):
            for feat, _, prob_map in cache[class_index]:
                cache_keys.append(feat)
                if negative:
                    if prob_map is None:
                        continue
                    masked = (
                        (prob_map > self.neg_mask_lower) & (prob_map < self.neg_mask_upper)
                    ).to(torch.float32)
                    cache_values.append(masked)
                else:
                    one_hot = F.one_hot(
                        torch.tensor(class_index, device=self.device),
                        num_classes=self.num_classes,
                    ).to(torch.float32)
                    cache_values.append(one_hot)

        if not cache_keys or not cache_values:
            return None, None

        keys = torch.stack(cache_keys, dim=0).to(self.device, dtype=torch.float32)  # [N, D]
        values = torch.stack(cache_values, dim=0).to(self.device, dtype=torch.float32)  # [N, C]
        return keys, values

    def _compute_cache_logits(
        self,
        x: torch.Tensor,
        cache: Dict[int, List[CacheItem]],
        alpha: float,
        beta: float,
        negative: bool = False,
    ) -> torch.Tensor:
        keys, values = self._build_cache_tensors(cache, negative=negative)
        if keys is None or values is None:
            return torch.zeros((1, self.num_classes), device=self.device, dtype=torch.float32)

        affinity = x @ keys.t()  # [1, N]

        if self.k > 0 and affinity.shape[1] > self.k:
            top_vals, top_idx = torch.topk(affinity, k=self.k, dim=-1)
            affinity = top_vals
            values = values[top_idx[0]]

        cache_logits = torch.exp(-(beta - beta * affinity)) @ values
        return float(alpha) * cache_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.normalize(x.to(self.device, dtype=torch.float32), dim=-1)

        clip_logits = self.clip_scale * (x @ self.text_features.t())
        clip_probs = F.softmax(clip_logits, dim=-1)

        clip_pred = int(torch.argmax(clip_logits, dim=-1).item())
        entropy = -torch.sum(clip_probs * torch.log(clip_probs + 1e-12), dim=-1)
        norm_entropy = float(entropy.item() / self.max_entropy)
        prob_map = clip_probs.squeeze(0)

        feat = x.squeeze(0).detach()
        self._update_cache(self.pos_cache, clip_pred, feat, float(entropy.item()), prob_map=None)
        if self.low_entropy < norm_entropy < self.high_entropy:
            self._update_cache(self.neg_cache, clip_pred, feat, float(entropy.item()), prob_map=prob_map)

        self.pos_size = self._cache_len(self.pos_cache)
        self.neg_size = self._cache_len(self.neg_cache)

        final_logits = clip_logits.clone()
        final_logits += self._compute_cache_logits(
            x,
            cache=self.pos_cache,
            alpha=self.alpha,
            beta=self.beta,
            negative=False,
        )
        final_logits -= self._compute_cache_logits(
            x,
            cache=self.neg_cache,
            alpha=self.neg_alpha,
            beta=self.neg_beta,
            negative=True,
        )

        final_probs = F.softmax(final_logits, dim=-1)
        if self.fallback_to_clip:
            clip_conf = torch.max(clip_probs, dim=-1).values
            final_conf = torch.max(final_probs, dim=-1).values
            if bool((final_conf + self.fallback_margin < clip_conf).item()):
                final_logits = clip_logits
                final_probs = clip_probs

        confidence, pred = torch.max(final_probs, dim=-1)
        return pred, confidence, final_logits
