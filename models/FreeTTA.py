from typing import Tuple, Union

import torch
import torch.nn.functional as F


class FreeTTA:
    """Online EM-style FreeTTA adapter with entropy-weighted updates."""

    def __init__(
        self,
        text_features,
        alpha=0.2,
        beta=4.5,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.mu = F.normalize(text_features.detach().float().to(self.device), dim=-1)
        self.num_classes, self.dim = self.mu.shape

        # Paper initialization: N_y=1/K, t=1, Sigma=I.
        self.Ny = torch.ones(self.num_classes, device=self.device) / float(self.num_classes)
        self.t = torch.tensor(1.0, device=self.device)
        self.sigma = torch.eye(self.dim, device=self.device)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eye = torch.eye(self.dim, device=self.device)

    @torch.no_grad()
    def predict(self, x, clip_logits):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if clip_logits.dim() == 1:
            clip_logits = clip_logits.unsqueeze(0)

        x = F.normalize(x.to(self.device, dtype=torch.float32), dim=-1)
        clip_logits = clip_logits.to(self.device, dtype=torch.float32)
        clip_probs = F.softmax(clip_logits, dim=-1)

        entropy = -torch.sum(clip_probs * torch.log(clip_probs + 1e-8), dim=-1)
        weight = torch.exp(-self.beta * entropy).item()

        sigma = self.sigma + 1e-4 * self.eye
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
            inv_sigma_mu = torch.linalg.solve(sigma, self.mu.t())
        else:
            inv_sigma_mu = torch.solve(self.mu.t(), sigma)[0]

        term1 = x @ inv_sigma_mu
        term2 = 0.5 * torch.sum(self.mu.t() * inv_sigma_mu, dim=0)
        priors = self.Ny / (self.t + 1e-8)
        gen_logits = term1 - term2 + torch.log(priors + 1e-8)

        gamma = F.softmax(gen_logits, dim=-1).squeeze(0)

        Ny_old = self.Ny.clone()
        t_old = self.t.clone()
        sigma_old = self.sigma.clone()
        mu_old = self.mu.clone()

        delta = weight * gamma
        self.Ny = self.Ny + delta
        self.t = self.t + weight

        self.mu = (Ny_old.unsqueeze(1) * mu_old + delta.unsqueeze(1) * x.squeeze(0)) / (
            self.Ny.unsqueeze(1) + 1e-8
        )
        self.mu = F.normalize(self.mu, dim=-1)

        # Eq. (12)-style online covariance update.
        if float(self.t.item()) > 1.0:
            diff = x.squeeze(0).unsqueeze(0) - self.mu
            outer = diff.unsqueeze(2) @ diff.unsqueeze(1)
            weighted_cov = (gamma.unsqueeze(1).unsqueeze(2) * outer).sum(dim=0)
            self.sigma = sigma_old + (weight * weighted_cov) / (self.t - 1 + 1e-8)

        self.sigma = self.sigma + 1e-6 * self.eye

        # Paper-style fusion in logit space.
        final_logits = clip_logits + self.alpha * gen_logits
        final_probs = F.softmax(final_logits, dim=-1)
        pred = torch.argmax(final_probs, dim=-1)
        return pred, final_probs
