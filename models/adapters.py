from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.EdgeFreeTTA import EdgeFreeTTA


class AdapterOutput(object):
    def __init__(self, pred, extra):
        self.pred = pred
        self.extra = extra


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-12))


def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + 1e-12) - torch.log(m + 1e-12)))
    kl_qm = torch.sum(q * (torch.log(q + 1e-12) - torch.log(m + 1e-12)))
    return 0.5 * (kl_pm + kl_qm)


class CLIPBaselineAdapter:
    def process_sample(
        self, x: torch.Tensor, clip_probs: torch.Tensor, clip_logits: torch.Tensor
    ) -> AdapterOutput:
        probs = torch.softmax(clip_logits, dim=-1)
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
        ent = float(_entropy(probs).item())
        return AdapterOutput(pred=pred, extra={"pred_confidence": conf, "entropy": ent})

    def state_nbytes(self) -> int:
        return 0


class TDAMemoryAdapter:
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        cache_size: int = 1024,
        top_k: int = 64,
        alpha: float = 0.45,
        temperature: float = 20.0,
        device: str = "cpu",
    ) -> None:
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.cache_size = int(cache_size)
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.update_conf_threshold = 0.60

        self.cache_features = torch.empty((0, self.feature_dim), dtype=torch.float32, device=self.device)
        self.cache_labels = torch.empty((0,), dtype=torch.long, device=self.device)

    def _cache_probs(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.cache_features.shape[0] == 0:
            return None
        similarities = torch.matmul(self.cache_features, x)
        similarities = similarities * self.temperature

        k = min(self.top_k, int(similarities.shape[0]))
        top_vals, top_idx = torch.topk(similarities, k=k, dim=0)
        top_labels = self.cache_labels[top_idx]
        weights = torch.softmax(top_vals, dim=0)

        class_scores = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)
        class_scores.scatter_add_(0, top_labels, weights)
        class_scores = class_scores / class_scores.sum().clamp_min(1e-12)
        return class_scores

    def _append_cache(self, x: torch.Tensor, pred: int) -> None:
        x = x.detach().to(self.device, dtype=torch.float32).view(1, -1)
        y = torch.tensor([pred], dtype=torch.long, device=self.device)
        if self.cache_features.shape[0] < self.cache_size:
            self.cache_features = torch.cat([self.cache_features, x], dim=0)
            self.cache_labels = torch.cat([self.cache_labels, y], dim=0)
            return
        self.cache_features = torch.cat([self.cache_features[1:], x], dim=0)
        self.cache_labels = torch.cat([self.cache_labels[1:], y], dim=0)

    def process_sample(
        self, x: torch.Tensor, clip_probs: torch.Tensor, clip_logits: torch.Tensor
    ) -> AdapterOutput:
        x = F.normalize(x, dim=-1)
        clip_probs = torch.softmax(clip_logits, dim=-1)
        clip_pred = int(torch.argmax(clip_probs).item())
        clip_conf = float(torch.max(clip_probs).item())
        cache_probs = self._cache_probs(x)
        cache_fill_ratio = float(self.cache_features.shape[0] / max(self.cache_size, 1))

        if cache_probs is None:
            fused_probs = clip_probs
            jsd_clip_fused = 0.0
        else:
            blend = max(0.0, min(1.0, self.alpha * cache_fill_ratio))
            fused_probs = (1.0 - blend) * clip_probs + blend * cache_probs
            fused_probs = fused_probs / fused_probs.sum().clamp_min(1e-12)
            jsd_clip_fused = float(_jsd(clip_probs, fused_probs).item())

        pred = int(torch.argmax(fused_probs).item())
        conf = float(fused_probs[pred].item())
        ent = float(_entropy(fused_probs).item())

        # Keep memory clean by caching only confident and non-contradictory pseudo-labels.
        if clip_conf >= self.update_conf_threshold and pred == clip_pred:
            self._append_cache(x, pred)

        cache_fill_ratio = float(self.cache_features.shape[0] / max(self.cache_size, 1))
        return AdapterOutput(
            pred=pred,
            extra={
                "pred_confidence": conf,
                "entropy": ent,
                "cache_fill_ratio": cache_fill_ratio,
                "jsd_clip_fused": jsd_clip_fused,
            },
        )

    def state_nbytes(self) -> int:
        return int(self.cache_features.nbytes + self.cache_labels.nbytes)


class FreeTTAOnlineEMAdapter:
    def __init__(
        self,
        text_features: torch.Tensor,
        alpha: float = 0.15,
        beta: float = 6.0,
        cov_momentum: float = 0.02,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        text = text_features.detach().to(self.device, dtype=torch.float32)
        self.mu = F.normalize(text, dim=-1)
        self.initial_mu = self.mu.detach().clone()
        self.counts = torch.ones(self.mu.shape[0], dtype=torch.float32, device=self.device)
        self.var = torch.full((self.mu.shape[1],), 0.1, dtype=torch.float32, device=self.device)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.cov_momentum = float(cov_momentum)

    def _entropy_weight(self, clip_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ent = _entropy(clip_probs)
        weight = torch.exp(-self.beta * ent)
        return ent, weight

    def _generative_logits(self, x: torch.Tensor) -> torch.Tensor:
        var = self.var.clamp_min(1e-6)
        inv_var = 1.0 / var
        diff = self.mu - x.unsqueeze(0)
        mahal = torch.sum(diff * diff * inv_var.unsqueeze(0), dim=-1)
        log_det = torch.sum(torch.log(var))
        log_prior = torch.log(self.counts / self.counts.sum().clamp_min(1e-12))
        return log_prior - 0.5 * (mahal + log_det)

    def _update_parameters(self, x: torch.Tensor, posterior: torch.Tensor, weight: torch.Tensor) -> float:
        delta = weight * posterior
        new_counts = self.counts + delta

        mu_prev = self.mu
        new_mu = (self.counts.unsqueeze(1) * self.mu + delta.unsqueeze(1) * x.unsqueeze(0)) / (
            new_counts.unsqueeze(1).clamp_min(1e-12)
        )
        self.mu = F.normalize(new_mu, dim=-1)
        self.counts = new_counts

        weighted_residual = torch.sum(delta.unsqueeze(1) * (x.unsqueeze(0) - mu_prev) ** 2, dim=0)
        denom = delta.sum().clamp_min(1e-12)
        target_var = weighted_residual / denom
        self.var = (1.0 - self.cov_momentum) * self.var + self.cov_momentum * target_var
        self.var = self.var.clamp(1e-6, 10.0)

        mu_update_norm = torch.norm(self.mu - mu_prev, dim=1).mean()
        return float(mu_update_norm.item())

    def process_sample(
        self, x: torch.Tensor, clip_probs: torch.Tensor, clip_logits: torch.Tensor
    ) -> AdapterOutput:
        x = F.normalize(x, dim=-1)
        clip_probs = torch.softmax(clip_logits, dim=-1)

        entropy, weight = self._entropy_weight(clip_probs)
        gen_logits = self._generative_logits(x)
        fused_logits = clip_logits + self.alpha * gen_logits
        fused_probs = torch.softmax(fused_logits, dim=-1)

        mu_update_norm = self._update_parameters(x, posterior=fused_probs, weight=weight)

        pred = int(torch.argmax(fused_probs).item())
        conf = float(fused_probs[pred].item())
        mu_drift = float(torch.norm(self.mu - self.initial_mu, dim=1).mean().item())
        weight_f = float(weight.item())

        return AdapterOutput(
            pred=pred,
            extra={
                "pred_confidence": conf,
                "entropy": float(entropy.item()),
                "weight": weight_f,
                "em_weight": weight_f,
                "mu_update_norm": mu_update_norm,
                "mu_drift_from_init": mu_drift,
            },
        )

    def state_nbytes(self) -> int:
        total = self.mu.nbytes + self.initial_mu.nbytes + self.counts.nbytes + self.var.nbytes
        return int(total)

    def get_class_means(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.initial_mu.detach().cpu(), self.mu.detach().cpu()


class EdgeFreeTTALowRankAdapter:
    def __init__(
        self,
        text_features: torch.Tensor,
        rank: int = 8,
        fusion_alpha: float = 0.5,
        learning_rate: float = 1e-2,
        beta: float = 4.5,
        min_confidence: float = 0.65,
        align_weight: float = 0.5,
        residual_weight: float = 0.05,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.model = EdgeFreeTTA(
            text_features=text_features,
            rank=rank,
            fusion_alpha=fusion_alpha,
            learning_rate=learning_rate,
            beta=beta,
            min_confidence=min_confidence,
            align_weight=align_weight,
            residual_weight=residual_weight,
            weight_decay=weight_decay,
            device=device,
        )

    def process_sample(
        self, x: torch.Tensor, clip_probs: torch.Tensor, clip_logits: torch.Tensor
    ) -> AdapterOutput:
        x = F.normalize(x, dim=-1)
        clip_logits = clip_logits.to(self.model.device, dtype=torch.float32)
        pred, probs, stats = self.model.predict_and_adapt(x=x, clip_logits=clip_logits)
        probs = probs.squeeze(0)
        pred_i = int(pred.squeeze(0).item())
        entropy = float(_entropy(probs).item())
        return AdapterOutput(
            pred=pred_i,
            extra={
                "pred_confidence": float(probs[pred_i].item()),
                "entropy": entropy,
                "edge_rank": float(self.model.rank),
                "edge_updated": stats["updated"],
                "edge_update_weight": stats["update_weight"],
                "edge_adapter_norm": stats["adapter_norm"],
                "edge_residual_norm": stats["residual_norm"],
            },
        )

    def state_nbytes(self) -> int:
        return self.model.state_nbytes()
