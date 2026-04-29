from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FreeTTA import FreeTTA
from research_pipeline.methods.base import MethodBase, SampleTrace, _entropy


class FreeTTAAdapter(MethodBase):
    """Wraps the FreeTTA model with a unified SampleTrace interface."""

    name = "freetta"

    def __init__(self, text_features: torch.Tensor, dataset: str = "dtd", **kwargs) -> None:
        cfg = {**FreeTTA.DATASET_DEFAULTS.get(dataset.lower(), {}), **kwargs}
        self._model = FreeTTA(text_features=text_features, **cfg)
        self._initial_mu = self._model.mu.detach().clone()

    def reset(self) -> None:
        self._model.reset()
        self._initial_mu = self._model.mu.detach().clone()

    @torch.no_grad()
    def predict_update(self, x: torch.Tensor, raw_clip_logits: torch.Tensor) -> SampleTrace:
        clip_pred_int = int(raw_clip_logits.argmax().item())

        # Match FreeTTA's internal prior-entropy and em_weight computation
        scaled_probs = torch.softmax(
            raw_clip_logits * self._model.clip_scale, dim=-1
        )
        norm_ent = float(
            (-(scaled_probs * torch.log(scaled_probs + 1e-12)).sum()
             / self._model.max_entropy
            ).clamp(0.0, 1.0).item()
        )
        em_weight = math.exp(-self._model.beta * norm_ent)
        mu_before = self._model.mu.detach().clone()

        pred_t, final_probs = self._model.predict(x, raw_clip_logits)
        final_probs = final_probs.squeeze(0)
        pred = int(pred_t.item())

        # Convert probs to log-scale logits for consistent storage
        logits = torch.log(final_probs + 1e-12)

        # Drift stats
        mu_drift_per_class = torch.norm(self._model.mu - self._initial_mu, dim=1)
        mu_drift = float(mu_drift_per_class.mean().item())
        mu_update_norm = float(torch.norm(self._model.mu - mu_before, dim=1).mean().item())
        priors = (self._model.Ny / (self._model.t + 1e-8)).clamp_min(1e-12)
        prior_entropy = float((-(priors * torch.log(priors))).sum().item())
        sigma_trace = float(
            torch.sum(self._model.mu * self._model.mu0).item()
            / max(self._model.num_classes, 1)
        )

        return SampleTrace(
            logits=logits,
            probs=final_probs,
            pred=pred,
            confidence=float(final_probs.max().item()),
            entropy=_entropy(final_probs),
            changed=(pred != clip_pred_int),
            extra={
                "em_weight": em_weight,
                "mu_update_norm": mu_update_norm,
                "mu_drift": mu_drift,
                "mu_drift_per_class": mu_drift_per_class.detach().cpu().numpy(),
                "prior_entropy": prior_entropy,
                "sigma_trace": sigma_trace,
            },
        )


class ConfGatedFreeTTA(MethodBase):
    """
    FreeTTA with a confidence gate on the M-step.

    Standard FreeTTA updates class means for every sample, weighted by entropy.
    This variant adds a hard confidence threshold: the M-step is skipped when
    max(CLIP softmax) < conf_threshold.  This prevents corrupt pseudo-labels
    from highly uncertain samples from drifting class means.

    Expected behavior: smaller but more accurate mean updates on uncertain
    datasets (DTD, EuroSAT). Minimal effect where CLIP is already confident.
    """

    name = "conf_ftta"

    def __init__(
        self,
        text_features: torch.Tensor,
        dataset: str = "dtd",
        conf_threshold: float = 0.50,    # skip M-step below this CLIP confidence
        **kwargs,
    ) -> None:
        cfg = {**FreeTTA.DATASET_DEFAULTS.get(dataset.lower(), {}), **kwargs}
        self._model = FreeTTA(text_features=text_features, **cfg)
        self._initial_mu = self._model.mu.detach().clone()
        self._conf_threshold = float(conf_threshold)
        self._skipped = 0
        self._total = 0

    def reset(self) -> None:
        self._model.reset()
        self._initial_mu = self._model.mu.detach().clone()
        self._skipped = 0
        self._total = 0

    @torch.no_grad()
    def predict_update(self, x: torch.Tensor, raw_clip_logits: torch.Tensor) -> SampleTrace:
        clip_pred_int = int(raw_clip_logits.argmax().item())
        self._total += 1

        clip_probs = torch.softmax(raw_clip_logits * self._model.clip_scale, dim=-1)
        clip_conf = float(clip_probs.max().item())

        # E-step is always computed for prediction
        x_n = F.normalize(x.float().to(self._model.device), dim=-1)
        if x_n.dim() > 1:
            x_n = x_n.squeeze(0)
        cl = raw_clip_logits.float().to(self._model.device).squeeze()

        clip_probs_dev = torch.softmax(cl * self._model.clip_scale, dim=0)
        h = -(clip_probs_dev * torch.log(clip_probs_dev + 1e-8)).sum()
        norm_h = (h / self._model.max_entropy).clamp(0.0, 1.0)
        weight = torch.exp(-self._model.beta * norm_h)

        gen_logits = x_n @ self._model.mu.t()
        fused_logits = (cl + self._model.alpha * gen_logits) * self._model.clip_scale
        pred = int(fused_logits.argmax().item())
        final_probs = torch.softmax(fused_logits, dim=0)
        logits = torch.log(final_probs + 1e-12)

        # Conditional M-step
        if clip_conf >= self._conf_threshold:
            delta = weight * clip_probs_dev
            Ny_new = self._model.Ny + delta
            self._model.mu = (
                self._model.Ny.unsqueeze(1) * self._model.mu
                + delta.unsqueeze(1) * x_n.unsqueeze(0)
            ) / (Ny_new.unsqueeze(1) + 1e-8)
            if self._model.normalize_mu:
                self._model.mu = F.normalize(self._model.mu, dim=-1)
            self._model.Ny = Ny_new
        else:
            self._skipped += 1

        self._model.t += 1

        mu_drift = float(
            torch.norm(self._model.mu - self._initial_mu, dim=1).mean().item()
        )
        priors = (self._model.Ny / (self._model.t + 1e-8)).clamp_min(1e-12)
        prior_entropy = float((-(priors * torch.log(priors))).sum().item())

        return SampleTrace(
            logits=logits,
            probs=final_probs,
            pred=pred,
            confidence=float(final_probs.max().item()),
            entropy=_entropy(final_probs),
            changed=(pred != clip_pred_int),
            extra={
                "m_step_skipped": int(clip_conf < self._conf_threshold),
                "skip_rate": self._skipped / max(self._total, 1),
                "mu_drift": mu_drift,
                "prior_entropy": prior_entropy,
            },
        )
