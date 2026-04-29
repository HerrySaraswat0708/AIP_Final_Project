from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _entropy(probs: torch.Tensor) -> float:
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


def _softmax_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    return _entropy(torch.softmax(logits * temperature, dim=-1))


@dataclass
class SampleTrace:
    """
    Unified per-sample result produced by every method.
    logits: (C,) in SCALED space (clip_scale * cosine_sim basis)
    probs:  (C,) softmax probabilities
    """
    logits: torch.Tensor
    probs: torch.Tensor
    pred: int
    confidence: float       # max(probs)
    entropy: float          # H(probs)
    changed: bool           # prediction differs from CLIP baseline
    extra: dict = field(default_factory=dict)


class MethodBase(ABC):
    """Streaming TTA method interface."""

    name: str = "base"

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def predict_update(
        self,
        x: torch.Tensor,           # (D,)  L2-normalised image feature
        raw_clip_logits: torch.Tensor,  # (C,) raw cosine sims (NOT scaled)
    ) -> SampleTrace: ...


class CLIPBaseline(MethodBase):
    """Frozen CLIP — no adaptation."""

    name = "clip"

    def __init__(self, text_features: torch.Tensor, clip_scale: float = 100.0) -> None:
        self.text_features = F.normalize(text_features.float(), dim=-1)
        self.clip_scale = clip_scale
        self.num_classes = text_features.shape[0]
        self.max_entropy = math.log(max(self.num_classes, 2))

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def predict_update(self, x: torch.Tensor, raw_clip_logits: torch.Tensor) -> SampleTrace:
        logits = raw_clip_logits.squeeze() * self.clip_scale  # (C,)
        probs = torch.softmax(logits, dim=-1)
        pred = int(logits.argmax().item())
        return SampleTrace(
            logits=logits,
            probs=probs,
            pred=pred,
            confidence=float(probs.max().item()),
            entropy=_entropy(probs),
            changed=False,
            extra={},
        )
