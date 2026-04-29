from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TDA import TDA
from models.FreeTTA import FreeTTA
from research_pipeline.methods.base import MethodBase, SampleTrace, _entropy


class HybridTDAFreeTTA(MethodBase):
    """
    Hybrid method: TDA local cache + FreeTTA global statistics.

    Architecture:
      z_hybrid = z_clip + w_local * (z_tda_adj) + w_global * (z_freetta_adj)

    where:
      z_tda_adj    = TDA's cache-based logit adjustment (pos_cache - neg_cache)
      z_freetta_adj = FreeTTA's generative logit adjustment (alpha * x @ mu.T)

    Both sub-adapters update simultaneously.  The relative weighting is
    controlled by w_local and w_global.

    Hypothesis: local cache covers short-range correlation (visually similar
    samples seen recently), while global statistics handle domain shift.
    Together they should be complementary on datasets with both within-class
    variation (handled by cache) and global domain shift (handled by FreeTTA).
    """

    name = "hybrid"

    def __init__(
        self,
        text_features: torch.Tensor,
        dataset: str = "dtd",
        w_local: float = 0.5,    # weight on TDA cache adjustment
        w_global: float = 0.5,   # weight on FreeTTA mean adjustment
        **kwargs,
    ) -> None:
        tda_cfg = dict(TDA.DATASET_DEFAULTS.get(dataset.lower(), {}))
        ftta_cfg = dict(FreeTTA.DATASET_DEFAULTS.get(dataset.lower(), {}))
        # Apply any overrides
        for k, v in kwargs.items():
            if k in ("alpha", "beta"):
                ftta_cfg[k] = v
            else:
                tda_cfg[k] = v

        self._tda = TDA(text_features=text_features, **tda_cfg)
        self._ftta = FreeTTA(text_features=text_features, **ftta_cfg)
        self._w_local = float(w_local)
        self._w_global = float(w_global)
        self._clip_scale = float(tda_cfg.get("clip_scale", 100.0))
        self._alpha = float(ftta_cfg.get("alpha", 0.3))
        self._initial_mu = self._ftta.mu.detach().clone()

    def reset(self) -> None:
        self._tda.reset()
        self._ftta.reset()
        self._initial_mu = self._ftta.mu.detach().clone()

    @torch.no_grad()
    def predict_update(self, x: torch.Tensor, raw_clip_logits: torch.Tensor) -> SampleTrace:
        clip_pred_int = int(raw_clip_logits.argmax().item())

        # ── CLIP base logits ──────────────────────────────────────────────────
        x_n = F.normalize(x.float().to(self._tda.device), dim=-1)
        if x_n.dim() > 1:
            x_n = x_n.squeeze(0)
        cl = raw_clip_logits.float().to(self._tda.device).squeeze()
        z_clip = cl * self._clip_scale  # (C,)

        # ── TDA cache adjustment ──────────────────────────────────────────────
        z_pos = self._tda._cache_logits(x_n, negative=False)   # (C,)
        z_neg = self._tda._cache_logits(x_n, negative=True)    # (C,)
        z_tda_adj = z_pos - z_neg                               # (C,)

        # ── FreeTTA generative adjustment ─────────────────────────────────────
        z_gen = x_n @ self._ftta.mu.t()   # (C,)  x · adapted_mu
        z_freetta_adj = self._alpha * z_gen * self._clip_scale  # (C,)

        # ── Fused prediction ─────────────────────────────────────────────────
        z_hybrid = (
            z_clip
            + self._w_local * z_tda_adj
            + self._w_global * z_freetta_adj
        )
        probs = torch.softmax(z_hybrid, dim=-1)
        pred = int(z_hybrid.argmax().item())

        # ── Update both sub-adapters ──────────────────────────────────────────
        # TDA update (cache insertion)
        clip_probs_raw = torch.softmax(cl, dim=-1)  # for TDA entropy routing
        h_raw = -(clip_probs_raw * torch.log(clip_probs_raw + 1e-12)).sum()
        norm_h_raw = float(
            (h_raw / math.log(max(self._tda.num_classes, 2))).clamp(0.0, 1.0).item()
        )
        ent_loss = float(h_raw.item())
        tda_clip_pred = int(cl.argmax().item())
        self._tda._update_slot(False, tda_clip_pred, x_n.detach(), ent_loss)
        if self._tda.low_entropy < norm_h_raw < self._tda.high_entropy:
            self._tda._update_slot(
                True, tda_clip_pred, x_n.detach(), ent_loss, pmap=clip_probs_raw
            )

        # FreeTTA M-step update
        scaled_probs = torch.softmax(cl * self._ftta.clip_scale, dim=-1)
        h_scaled = -(scaled_probs * torch.log(scaled_probs + 1e-8)).sum()
        norm_h_scaled = (h_scaled / self._ftta.max_entropy).clamp(0.0, 1.0)
        weight = torch.exp(-self._ftta.beta * norm_h_scaled)
        delta = weight * scaled_probs
        Ny_new = self._ftta.Ny + delta
        self._ftta.mu = (
            self._ftta.Ny.unsqueeze(1) * self._ftta.mu
            + delta.unsqueeze(1) * x_n.unsqueeze(0)
        ) / (Ny_new.unsqueeze(1) + 1e-8)
        if self._ftta.normalize_mu:
            self._ftta.mu = F.normalize(self._ftta.mu, dim=-1)
        self._ftta.Ny = Ny_new
        self._ftta.t += 1

        mu_drift = float(
            torch.norm(self._ftta.mu - self._initial_mu, dim=1).mean().item()
        )

        return SampleTrace(
            logits=z_hybrid,
            probs=probs,
            pred=pred,
            confidence=float(probs.max().item()),
            entropy=_entropy(probs),
            changed=(pred != clip_pred_int),
            extra={
                "tda_pos_cache_size": self._tda.pos_size,
                "tda_neg_cache_size": self._tda.neg_size,
                "ftta_mu_drift": mu_drift,
                "z_tda_adj_norm": float(torch.norm(z_tda_adj).item()),
                "z_freetta_adj_norm": float(torch.norm(z_freetta_adj).item()),
            },
        )
