import math

import torch
import torch.nn.functional as F


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
        pos_shot_capacity: int | None = None,
        neg_shot_capacity: int | None = None,
        clip_scale: float = 100.0,
        fallback_to_clip: bool = False,
        fallback_margin: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.text_features = F.normalize(text_features, dim=-1).to(self.device, dtype=torch.float32)
        self.num_classes = int(self.text_features.shape[0])
        self.dim = int(self.text_features.shape[1])

        self.cache_size = int(cache_size)
        default_capacity = max(1, int(shot_capacity))
        self.pos_shot_capacity = max(1, int(pos_shot_capacity if pos_shot_capacity is not None else default_capacity))
        self.neg_shot_capacity = max(1, int(neg_shot_capacity if neg_shot_capacity is not None else default_capacity))
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

        self.pos_size = 0
        self.neg_size = 0
        self._loss_inf = float("inf")

        # Fixed-capacity tensor caches preserve the same per-class shot capacity
        # without rebuilding flattened cache tensors for every sample.
        self.pos_features = torch.zeros(
            (self.num_classes, self.pos_shot_capacity, self.dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.pos_losses = torch.full(
            (self.num_classes, self.pos_shot_capacity),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        self.pos_counts = torch.zeros(self.num_classes, device=self.device, dtype=torch.long)

        self.neg_features = torch.zeros(
            (self.num_classes, self.neg_shot_capacity, self.dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.neg_losses = torch.full(
            (self.num_classes, self.neg_shot_capacity),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        self.neg_prob_maps = torch.zeros(
            (self.num_classes, self.neg_shot_capacity, self.num_classes),
            device=self.device,
            dtype=torch.float32,
        )
        self.neg_counts = torch.zeros(self.num_classes, device=self.device, dtype=torch.long)

    def _update_cache(
        self,
        negative: bool,
        pred: int,
        feature: torch.Tensor,
        entropy_loss: float,
        prob_map: torch.Tensor | None = None,
    ) -> None:
        if negative:
            features = self.neg_features
            losses = self.neg_losses
            counts = self.neg_counts
        else:
            features = self.pos_features
            losses = self.pos_losses
            counts = self.pos_counts

        class_capacity = self.neg_shot_capacity if negative else self.pos_shot_capacity
        count = int(counts[pred].item())
        if count < class_capacity:
            insert_idx = count
            counts[pred] += 1
            if negative:
                self.neg_size += 1
            else:
                self.pos_size += 1
        else:
            worst_loss, worst_idx = torch.max(losses[pred], dim=0)
            if entropy_loss >= float(worst_loss.item()):
                return
            insert_idx = int(worst_idx.item())

        features[pred, insert_idx].copy_(feature)
        losses[pred, insert_idx] = entropy_loss

        if negative and prob_map is not None:
            masked = ((prob_map > self.neg_mask_lower) & (prob_map < self.neg_mask_upper)).to(torch.float32)
            self.neg_prob_maps[pred, insert_idx].copy_(masked)

        active_count = int(counts[pred].item())
        if active_count <= 1:
            return

        sorted_losses, order = torch.sort(losses[pred, :active_count], dim=0)
        losses[pred, :active_count] = sorted_losses
        features[pred, :active_count] = features[pred, order]
        if negative:
            self.neg_prob_maps[pred, :active_count] = self.neg_prob_maps[pred, order]

    def _compute_cache_logits(
        self,
        x: torch.Tensor,
        alpha: float,
        beta: float,
        negative: bool = False,
    ) -> torch.Tensor:
        counts = self.neg_counts if negative else self.pos_counts
        total = self.neg_size if negative else self.pos_size
        if total == 0:
            return torch.zeros((1, self.num_classes), device=self.device, dtype=torch.float32)

        features = self.neg_features if negative else self.pos_features
        class_capacity = self.neg_shot_capacity if negative else self.pos_shot_capacity
        active_mask = (
            torch.arange(class_capacity, device=self.device).unsqueeze(0)
            < counts.unsqueeze(1)
        )
        active_mask_f = active_mask.to(torch.float32)

        affinity = torch.einsum("bd,csd->bcs", x, features)

        if self.k > 0 and total > self.k:
            flat_affinity = affinity.reshape(affinity.shape[0], -1)
            flat_mask = active_mask.reshape(-1)
            flat_affinity = flat_affinity.masked_fill(~flat_mask.unsqueeze(0), float("-inf"))
            top_vals, top_idx = torch.topk(flat_affinity, k=self.k, dim=-1)
            top_weights = torch.exp(-(beta - beta * top_vals))
            class_idx = torch.div(top_idx[0], class_capacity, rounding_mode="floor")
            slot_idx = torch.remainder(top_idx[0], class_capacity)
            if negative:
                selected_values = self.neg_prob_maps[class_idx, slot_idx]
                cache_logits = top_weights @ selected_values
            else:
                cache_logits = torch.zeros((1, self.num_classes), device=self.device, dtype=torch.float32)
                cache_logits.scatter_add_(1, class_idx.unsqueeze(0), top_weights)
            return float(alpha) * cache_logits

        weights = torch.exp(-(beta - beta * affinity)) * active_mask_f.unsqueeze(0)
        if negative:
            cache_logits = torch.einsum("bcs,csk->bk", weights, self.neg_prob_maps)
        else:
            cache_logits = weights.sum(dim=-1)
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
        entropy_loss = float(entropy.item())
        self._update_cache(False, clip_pred, feat, entropy_loss, prob_map=None)
        if self.low_entropy < norm_entropy < self.high_entropy:
            self._update_cache(True, clip_pred, feat, entropy_loss, prob_map=prob_map)

        final_logits = clip_logits.clone()
        final_logits += self._compute_cache_logits(
            x,
            alpha=self.alpha,
            beta=self.beta,
            negative=False,
        )
        final_logits -= self._compute_cache_logits(
            x,
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
