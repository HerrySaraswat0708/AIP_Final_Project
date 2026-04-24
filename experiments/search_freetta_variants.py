from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_freetta import load_freetta_dataset


@dataclass
class VariantConfig:
    alpha: float
    beta: float
    stream_seed: int
    clip_scale: float
    normalize_mu: bool
    use_paper_cov_update: bool
    use_reg_inverse: bool
    normalize_entropy: bool
    use_fused_posterior: bool


class FreeTTAVariant:
    def __init__(self, text_features: torch.Tensor, config: VariantConfig, device: torch.device) -> None:
        self.device = device
        self.cfg = config
        self.mu = F.normalize(text_features.detach().float().to(device), dim=-1)
        self.num_classes, self.dim = self.mu.shape
        self.Ny = torch.ones(self.num_classes, device=device) / float(self.num_classes)
        self.t = torch.tensor(1.0, device=device)
        self.sigma = torch.eye(self.dim, device=device)
        self.eye = torch.eye(self.dim, device=device)

    def _inv_sigma(self) -> torch.Tensor:
        sigma = self.sigma + 1e-6 * self.eye
        if self.cfg.use_reg_inverse:
            reg = (self.t - 1.0).clamp_min(0.0) * sigma + torch.trace(sigma) * self.eye
            sigma = self.dim * reg
        return torch.linalg.inv(sigma)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, clip_logits: torch.Tensor) -> tuple[int, torch.Tensor]:
        x = F.normalize(x.to(self.device, dtype=torch.float32), dim=-1)
        clip_logits = clip_logits.to(self.device, dtype=torch.float32) * self.cfg.clip_scale
        clip_probs = torch.softmax(clip_logits, dim=-1)

        entropy = -(clip_probs * torch.log(clip_probs + 1e-12)).sum()
        if self.cfg.normalize_entropy:
            entropy = entropy / torch.log(torch.tensor(float(self.num_classes), device=self.device))
        weight = torch.exp(-self.cfg.beta * entropy)

        inv_sigma = self._inv_sigma()
        w = self.mu @ inv_sigma
        b = torch.log((self.Ny / self.t).clamp_min(1e-12)) - 0.5 * torch.sum(w * self.mu, dim=-1)
        gen_logits = w @ x + b
        gen_probs = torch.softmax(gen_logits, dim=-1)

        fused_logits = clip_logits + self.cfg.alpha * gen_logits
        fused_probs = torch.softmax(fused_logits, dim=-1)
        posterior = fused_probs if self.cfg.use_fused_posterior else gen_probs

        Ny_old = self.Ny.clone()
        t_old = self.t.clone()
        sigma_old = self.sigma.clone()
        mu_old = self.mu.clone()

        delta = weight * posterior
        self.Ny = self.Ny + delta
        self.t = self.t + weight

        self.mu = (Ny_old.unsqueeze(1) * mu_old + delta.unsqueeze(1) * x.unsqueeze(0)) / (
            self.Ny.unsqueeze(1).clamp_min(1e-12)
        )
        if self.cfg.normalize_mu:
            self.mu = F.normalize(self.mu, dim=-1)

        diff = x.unsqueeze(0) - self.mu
        weighted_cov = (posterior.unsqueeze(1).unsqueeze(2) * (diff.unsqueeze(2) @ diff.unsqueeze(1))).sum(dim=0)
        if self.cfg.use_paper_cov_update:
            num = (self.t - 1.0) * sigma_old + weight * weighted_cov
            den = (self.t - 1.0).clamp_min(1e-12)
            self.sigma = num / den
        else:
            self.sigma = sigma_old + (weight * weighted_cov) / ((self.t - 1.0).clamp_min(1e-12))

        self.sigma = 0.5 * (self.sigma + self.sigma.t()) + 1e-6 * self.eye
        pred = int(torch.argmax(fused_probs).item())
        return pred, fused_probs


def evaluate_variant(payload: dict, config: VariantConfig, device: torch.device) -> float:
    image_features = payload["image_features"]
    text_features = payload["text_features"]
    labels = payload["labels"]
    total = int(payload["num_samples"])

    model = FreeTTAVariant(text_features=text_features, config=config, device=device)
    generator = torch.Generator()
    generator.manual_seed(int(config.stream_seed))
    order = torch.randperm(total, generator=generator).to(labels.device)

    correct = 0
    with torch.no_grad():
        for idx in order:
            x = image_features[idx]
            clip_logits = x @ text_features.t()
            pred, _ = model.predict(x, clip_logits)
            correct += int(pred == int(labels[idx].item()))
    return float(correct / max(total, 1))


def parse_bool_list(raw: str) -> list[bool]:
    return [item.strip().lower() in {"1", "true", "yes", "y"} for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search FreeTTA implementation variants")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--betas", default="1.0,2.0,4.5,6.0,8.0")
    parser.add_argument("--stream-seeds", default="1,2,3,4,5")
    parser.add_argument("--shuffle-stream", action="store_true")
    parser.add_argument("--clip-scales", default="1.0,100.0")
    parser.add_argument("--normalize-mu", default="true,false")
    parser.add_argument("--paper-cov", default="true,false")
    parser.add_argument("--reg-inverse", default="true,false")
    parser.add_argument("--normalize-entropy", default="false,true")
    parser.add_argument("--fused-posterior", default="false,true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = load_freetta_dataset(dataset=args.dataset, device=device, features_dir=args.features_dir)

    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    seeds = parse_int_list(args.stream_seeds)
    clip_scales = parse_float_list(args.clip_scales)
    normalize_mu_list = parse_bool_list(args.normalize_mu)
    paper_cov_list = parse_bool_list(args.paper_cov)
    reg_inverse_list = parse_bool_list(args.reg_inverse)
    normalize_entropy_list = parse_bool_list(args.normalize_entropy)
    fused_posterior_list = parse_bool_list(args.fused_posterior)

    rows = []
    best = None
    for combo in itertools.product(
        alphas,
        betas,
        seeds,
        clip_scales,
        normalize_mu_list,
        paper_cov_list,
        reg_inverse_list,
        normalize_entropy_list,
        fused_posterior_list,
    ):
        cfg = VariantConfig(*combo)
        if args.shuffle_stream:
            acc = evaluate_variant(payload=payload, config=cfg, device=device)
        else:
            cfg = VariantConfig(
                alpha=cfg.alpha,
                beta=cfg.beta,
                stream_seed=1,
                clip_scale=cfg.clip_scale,
                normalize_mu=cfg.normalize_mu,
                use_paper_cov_update=cfg.use_paper_cov_update,
                use_reg_inverse=cfg.use_reg_inverse,
                normalize_entropy=cfg.normalize_entropy,
                use_fused_posterior=cfg.use_fused_posterior,
            )
            image_features = payload["image_features"]
            text_features = payload["text_features"]
            labels = payload["labels"]
            total = int(payload["num_samples"])
            model = FreeTTAVariant(text_features=text_features, config=cfg, device=device)
            correct = 0
            with torch.no_grad():
                for idx in range(total):
                    x = image_features[idx]
                    clip_logits = x @ text_features.t()
                    pred, _ = model.predict(x, clip_logits)
                    correct += int(pred == int(labels[idx].item()))
            acc = float(correct / max(total, 1))
        row = asdict(cfg)
        row["accuracy"] = acc
        row["shuffle_stream"] = bool(args.shuffle_stream)
        rows.append(row)
        if best is None or acc > best["accuracy"]:
            best = row
        print(json.dumps(row), flush=True)

    result = {"dataset": args.dataset, "best": best, "rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
