from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class FreeTTA:
    """
    FreeTTA: Online EM test-time adaptation using CLIP features.

    Algorithm (per test sample, in order):
      E-step: compute fused prediction from CLIP + adapted class means
      M-step: update adapted class means weighted by CLIP confidence

    GPU efficiency:
      - Pre-computes ALL clip_logits in a single batched matmul before the loop
      - All inner-loop ops stay on GPU (no .item() / CPU transfers in hot path)
      - Vectorized mean update (no per-class inner loop)
    """

    DATASET_DEFAULTS = {
        "caltech":  dict(alpha=0.02,  beta=3.0),
        "dtd":      dict(alpha=0.1,   beta=3.0),
        "eurosat":  dict(alpha=0.8,   beta=3.0),
        "pets":     dict(alpha=0.25,  beta=4.0),
        "imagenet": dict(alpha=0.05,  beta=4.0),
    }

    def __init__(
        self,
        text_features: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 3.0,
        clip_scale: float = 100.0,
        normalize_mu: bool = True,
        device: str = "cuda",
        # legacy ignored params
        **_kwargs,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        tf = F.normalize(text_features.detach().float().to(self.device), dim=-1)
        self.text_features = tf                         # (C, D) fixed CLIP text features
        self.mu = tf.clone()                            # (C, D) adapted means, init = text features
        self.Ny = torch.ones(tf.shape[0], device=self.device) / tf.shape[0]  # (C,) soft counts

        self.num_classes, self.dim = tf.shape
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.clip_scale = float(clip_scale)
        self.normalize_mu = bool(normalize_mu)
        self.max_entropy = math.log(max(self.num_classes, 2))
        self.mu0 = tf.clone()   # fixed copy of initial text features for drift tracking
        self.t = 0              # samples processed counter

    def reset(self) -> None:
        """Reset adapted state back to CLIP text features."""
        self.mu = self.text_features.clone()
        self.Ny = torch.ones(self.num_classes, device=self.device) / self.num_classes
        self.t = 0

    @torch.no_grad()
    def run(self, image_features: torch.Tensor,
            batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run TTA over all images, updating class means after each sample.

        Parameters
        ----------
        image_features : (N, D)  L2-normalised CLIP image features
        batch_size     : int     1 = exact sequential (default); >1 = batch-EM
                                 approximation for fast GPU tuning (~30× speedup)

        Returns
        -------
        preds      : (N,)  predicted class indices
        clip_preds : (N,)  CLIP-only predicted class indices (for accuracy comparison)
        """
        N = image_features.shape[0]
        img = F.normalize(image_features.to(self.device, dtype=torch.float32), dim=-1)

        # ── Single batched matmul for ALL clip logits ─────────────────────────
        clip_logits_all = img @ self.text_features.t()   # (N, C)

        preds = torch.empty(N, dtype=torch.long, device=self.device)
        clip_preds = clip_logits_all.argmax(dim=-1)

        if batch_size <= 1:
            # ── Exact sequential (accurate, slower on GPU for small C) ────────
            for i in range(N):
                x = img[i]                              # (D,)
                clip_logits = clip_logits_all[i]        # (C,)

                # E-step
                clip_probs = torch.softmax(clip_logits * self.clip_scale, dim=0)
                h = -(clip_probs * torch.log(clip_probs + 1e-8)).sum()
                norm_h = (h / self.max_entropy).clamp(0.0, 1.0)
                weight = torch.exp(-self.beta * norm_h)

                gen_logits = x @ self.mu.t()            # (C,)
                fused_logits = (clip_logits + self.alpha * gen_logits) * self.clip_scale
                preds[i] = fused_logits.argmax()

                # M-step
                delta = weight * clip_probs             # (C,)
                Ny_new = self.Ny + delta
                self.mu = (
                    self.Ny.unsqueeze(1) * self.mu
                    + delta.unsqueeze(1) * x.unsqueeze(0)
                ) / (Ny_new.unsqueeze(1) + 1e-8)
                if self.normalize_mu:
                    self.mu = F.normalize(self.mu, dim=-1)
                self.Ny = Ny_new
        else:
            # ── Batch-EM (fast GPU mode): E-step uses mu at start of mini-batch;
            #    M-step aggregates batch updates in one fused op.
            #    Accuracy is very close to sequential for batch_size ≤ 32.  ──────
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_b = img[start:end]                   # (B, D)
                cl_b = clip_logits_all[start:end]       # (B, C)

                # E-step (batch)
                cp_b = torch.softmax(cl_b * self.clip_scale, dim=-1)   # (B, C)
                h_b = -(cp_b * torch.log(cp_b + 1e-8)).sum(dim=-1)     # (B,)
                nh_b = (h_b / self.max_entropy).clamp(0.0, 1.0)
                w_b = torch.exp(-self.beta * nh_b)                      # (B,)

                gl_b = x_b @ self.mu.t()                                # (B, C)
                fl_b = (cl_b + self.alpha * gl_b) * self.clip_scale     # (B, C)
                preds[start:end] = fl_b.argmax(dim=-1)

                # M-step (batch): Σ_k delta_k ⊗ x_k = delta_b.T @ x_b
                delta_b = w_b.unsqueeze(-1) * cp_b     # (B, C)
                delta_sum = delta_b.sum(dim=0)          # (C,)
                Ny_new = self.Ny + delta_sum
                mu_update = delta_b.t() @ x_b          # (C, D)
                self.mu = (
                    self.Ny.unsqueeze(1) * self.mu + mu_update
                ) / (Ny_new.unsqueeze(1) + 1e-8)
                if self.normalize_mu:
                    self.mu = F.normalize(self.mu, dim=-1)
                self.Ny = Ny_new
                self.t += 1

        return preds, clip_preds

    @torch.no_grad()
    def predict(self, x: torch.Tensor, clip_logits: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single image feature, update state, return (pred, final_probs).

        Parameters
        ----------
        x            : (D,) or (1, D)  L2-normalised image feature
        clip_logits  : (C,) or (1, C)  pre-computed CLIP logits (optional)

        Returns
        -------
        pred         : (1,)  predicted class index
        final_probs  : (1, C)  softmax probabilities of fused logits
        """
        x = F.normalize(x.float().to(self.device), dim=-1)
        if x.dim() > 1:
            x = x.squeeze(0)   # (D,)

        if clip_logits is None:
            clip_logits = x @ self.text_features.t()   # (C,)
        else:
            clip_logits = clip_logits.float().to(self.device)
            if clip_logits.dim() > 1:
                clip_logits = clip_logits.squeeze(0)   # (C,)

        # E-step
        clip_probs = torch.softmax(clip_logits * self.clip_scale, dim=0)   # (C,)
        h = -(clip_probs * torch.log(clip_probs + 1e-8)).sum()
        norm_h = (h / self.max_entropy).clamp(0.0, 1.0)
        weight = torch.exp(-self.beta * norm_h)

        gen_logits = x @ self.mu.t()   # (C,)
        fused_logits = (clip_logits + self.alpha * gen_logits) * self.clip_scale
        pred = fused_logits.argmax().unsqueeze(0)                          # (1,)
        final_probs = torch.softmax(fused_logits, dim=0).unsqueeze(0)     # (1, C)

        # M-step
        delta = weight * clip_probs   # (C,)
        Ny_new = self.Ny + delta
        self.mu = (
            self.Ny.unsqueeze(1) * self.mu
            + delta.unsqueeze(1) * x.unsqueeze(0)
        ) / (Ny_new.unsqueeze(1) + 1e-8)
        if self.normalize_mu:
            self.mu = F.normalize(self.mu, dim=-1)
        self.Ny = Ny_new
        self.t += 1

        return pred, final_probs

    @classmethod
    def for_dataset(cls, dataset_name: str, text_features: torch.Tensor,
                    device: str = "cuda") -> "FreeTTA":
        cfg = cls.DATASET_DEFAULTS.get(dataset_name.lower(), {})
        return cls(text_features, device=device, **cfg)


__all__ = ["FreeTTA"]
