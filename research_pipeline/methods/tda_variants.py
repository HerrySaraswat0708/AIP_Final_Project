from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TDA import TDA
from research_pipeline.methods.base import MethodBase, SampleTrace, _entropy


class TDAAdapter(MethodBase):
    """Wraps the TDA model with a unified SampleTrace interface."""

    name = "tda"

    def __init__(self, text_features: torch.Tensor, dataset: str = "dtd", **kwargs) -> None:
        cfg = {**TDA.DATASET_DEFAULTS.get(dataset.lower(), {}), **kwargs}
        self._model = TDA(text_features=text_features, **cfg)
        self._clip_pred: int = 0

    def reset(self) -> None:
        self._model.reset()

    @torch.no_grad()
    def predict_update(self, x: torch.Tensor, raw_clip_logits: torch.Tensor) -> SampleTrace:
        pred_t, clip_pred_int, final_logits = self._model.predict(x)
        self._clip_pred = clip_pred_int
        logits = final_logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred = int(pred_t.item())
        return SampleTrace(
            logits=logits,
            probs=probs,
            pred=pred,
            confidence=float(probs.max().item()),
            entropy=_entropy(probs),
            changed=(pred != clip_pred_int),
            extra={
                "pos_cache_size": self._model.pos_size,
                "neg_cache_size": self._model.neg_size,
                "gate_open": int(
                    self._model.low_entropy
                    < float((
                        -(torch.softmax(raw_clip_logits * self._model.clip_scale, dim=-1)
                          * torch.log(torch.softmax(raw_clip_logits * self._model.clip_scale, dim=-1) + 1e-12)
                         ).sum() / math.log(max(self._model.num_classes, 2))
                    ).clamp(0.0, 1.0).item())
                    < self._model.high_entropy
                ),
            },
        )


class EntropyGatedTDA(MethodBase):
    """
    TDA with an adaptive entropy gate.

    Standard TDA uses a fixed [low, high] entropy band for cache insertion.
    This variant tracks the running p-th percentile of observed entropies and
    dynamically tightens the low threshold, so only the most confident samples
    enter the positive cache.  Result: smaller but higher-quality cache.
    """

    name = "ent_tda"

    def __init__(
        self,
        text_features: torch.Tensor,
        dataset: str = "dtd",
        adaptive_percentile: float = 0.20,   # keep only bottom 20% by entropy
        warmup_steps: int = 50,              # steps before adaptive threshold kicks in
        **kwargs,
    ) -> None:
        cfg = {**TDA.DATASET_DEFAULTS.get(dataset.lower(), {}), **kwargs}
        # Start with a very permissive gate; will be tightened adaptively
        self._orig_low = cfg.get("low_entropy_thresh", 0.2)
        self._orig_high = cfg.get("high_entropy_thresh", 0.5)
        cfg["low_entropy_thresh"] = 0.0       # accept everything initially
        self._model = TDA(text_features=text_features, **cfg)
        self._adaptive_percentile = adaptive_percentile
        self._warmup = warmup_steps
        self._entropy_history: list[float] = []
        self._step = 0
        self._current_low = 0.0
        self._max_entropy = math.log(max(text_features.shape[0], 2))

    def reset(self) -> None:
        self._model.reset()
        self._entropy_history.clear()
        self._step = 0
        self._current_low = 0.0
        self._model.low_entropy = 0.0

    @torch.no_grad()
    def predict_update(self, x: torch.Tensor, raw_clip_logits: torch.Tensor) -> SampleTrace:
        # Compute normalized entropy of CLIP prediction
        scaled_probs = torch.softmax(raw_clip_logits * self._model.clip_scale, dim=-1)
        norm_ent = float(
            (-(scaled_probs * torch.log(scaled_probs + 1e-12)).sum()
             / self._max_entropy
            ).clamp(0.0, 1.0).item()
        )
        self._entropy_history.append(norm_ent)
        self._step += 1

        # Update adaptive threshold after warmup
        if self._step > self._warmup:
            sorted_ents = sorted(self._entropy_history)
            idx = max(0, int(len(sorted_ents) * self._adaptive_percentile) - 1)
            self._current_low = sorted_ents[idx]
            self._model.low_entropy = self._current_low
            self._model.high_entropy = min(self._orig_high, self._current_low + 0.15)

        pred_t, clip_pred_int, final_logits = self._model.predict(x)
        logits = final_logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred = int(pred_t.item())
        return SampleTrace(
            logits=logits,
            probs=probs,
            pred=pred,
            confidence=float(probs.max().item()),
            entropy=_entropy(probs),
            changed=(pred != clip_pred_int),
            extra={
                "pos_cache_size": self._model.pos_size,
                "neg_cache_size": self._model.neg_size,
                "adaptive_low_thresh": self._current_low,
                "gate_open": int(
                    self._model.low_entropy < norm_ent < self._model.high_entropy
                ),
            },
        )
