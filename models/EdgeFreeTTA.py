from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class EdgeFreeTTA:
    """Low-rank feature-space test-time adapter for frozen CLIP features.

    The pretrained image/text features remain frozen. Adaptation happens through
    a compact residual module `delta(x) = B(Ax)` with rank << feature_dim.
    """

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
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        text = text_features.detach().to(self.device, dtype=torch.float32)
        self.text_features = F.normalize(text, dim=-1)
        self.num_classes, self.feature_dim = self.text_features.shape

        self.rank = max(1, min(int(rank), int(self.feature_dim)))
        self.fusion_alpha = float(fusion_alpha)
        self.learning_rate = float(learning_rate)
        self.beta = float(beta)
        self.min_confidence = float(min_confidence)
        self.align_weight = float(align_weight)
        self.residual_weight = float(residual_weight)
        self.weight_decay = float(weight_decay)

        self.down_proj = torch.empty(
            (self.rank, self.feature_dim), dtype=torch.float32, device=self.device
        )
        self.up_proj = torch.zeros(
            (self.feature_dim, self.rank), dtype=torch.float32, device=self.device
        )
        torch.nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

        self.down_proj.requires_grad_(True)
        self.up_proj.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            [self.down_proj, self.up_proj],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

    def _low_rank_residual(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down_proj @ x
        residual = self.up_proj @ hidden
        return residual

    def _forward_logits(self, x: torch.Tensor, clip_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self._low_rank_residual(x)
        adapted_x = F.normalize(x + residual, dim=-1)
        adapted_logits = adapted_x @ self.text_features.t()
        fused_logits = clip_logits + self.fusion_alpha * (adapted_logits - clip_logits)
        return fused_logits, adapted_x

    @torch.no_grad()
    def predict(self, x: torch.Tensor, clip_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 1:
            x = x.reshape(-1)
        if clip_logits.dim() != 1:
            clip_logits = clip_logits.reshape(-1)

        x = F.normalize(x.to(self.device, dtype=torch.float32), dim=-1)
        clip_logits = clip_logits.to(self.device, dtype=torch.float32)
        fused_logits, _ = self._forward_logits(x, clip_logits)
        probs = torch.softmax(fused_logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        return pred.unsqueeze(0), probs.unsqueeze(0)

    def adapt(self, x: torch.Tensor, clip_logits: torch.Tensor) -> Dict[str, float]:
        if x.dim() != 1:
            x = x.reshape(-1)
        if clip_logits.dim() != 1:
            clip_logits = clip_logits.reshape(-1)

        x = F.normalize(x.to(self.device, dtype=torch.float32), dim=-1)
        clip_logits = clip_logits.detach().to(self.device, dtype=torch.float32)
        clip_probs = torch.softmax(clip_logits, dim=-1)
        clip_entropy = self._entropy(clip_probs)
        clip_confidence = torch.max(clip_probs)
        update_weight = torch.exp(-self.beta * clip_entropy)

        if bool(clip_confidence < self.min_confidence):
            return {
                "updated": 0.0,
                "update_weight": float(update_weight.item()),
                "adapter_norm": float(torch.norm(self.up_proj @ self.down_proj).item()),
                "residual_norm": 0.0,
            }

        pseudo_label = int(torch.argmax(clip_probs).item())
        target_text = self.text_features[pseudo_label]

        self.optimizer.zero_grad(set_to_none=True)
        fused_logits, adapted_x = self._forward_logits(x, clip_logits)
        residual = adapted_x - x

        ce_loss = F.cross_entropy(fused_logits.unsqueeze(0), torch.tensor([pseudo_label], device=self.device))
        align_loss = 1.0 - torch.sum(adapted_x * target_text)
        residual_loss = torch.sum(residual * residual)
        loss = update_weight * (ce_loss + self.align_weight * align_loss) + self.residual_weight * residual_loss
        loss.backward()
        self.optimizer.step()

        adapter_matrix = self.up_proj @ self.down_proj
        return {
            "updated": 1.0,
            "update_weight": float(update_weight.item()),
            "adapter_norm": float(torch.norm(adapter_matrix).item()),
            "residual_norm": float(torch.norm(residual).item()),
        }

    def predict_and_adapt(
        self,
        x: torch.Tensor,
        clip_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        stats = self.adapt(x, clip_logits)
        pred, probs = self.predict(x, clip_logits)
        return pred, probs, stats

    def state_nbytes(self) -> int:
        total = int(self.down_proj.nbytes + self.up_proj.nbytes)
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += int(value.nbytes)
        return total
