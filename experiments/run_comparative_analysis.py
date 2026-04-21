from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_freetta import load_freetta_dataset
from experiments.evaluate_tda import load_tda_dataset
from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.paper_configs import DEFAULT_FREETTA_PARAMS, PAPER_TDA_DEFAULTS
from src.paper_setup import EXPECTED_TEST_SPLIT_SIZES


DEFAULT_DATASETS = ("caltech", "dtd", "eurosat", "pets", "imagenet")


@dataclass
class DatasetPayload:
    dataset: str
    image_features: torch.Tensor
    text_features: torch.Tensor
    labels: torch.Tensor
    raw_clip_logits: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.text_features.shape[0])


@dataclass
class GeometryMetrics:
    oracle_centroid_acc: float
    oracle_1nn_acc: float
    geometry_alignment_score: float


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=dim)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=dim)


def top2_margin(logits: torch.Tensor) -> float:
    values = torch.topk(logits.reshape(-1), k=min(2, logits.numel())).values
    if values.numel() < 2:
        return float("nan")
    return float((values[0] - values[1]).item())


def get_order(num_samples: int, device: torch.device, seed: int | None) -> torch.Tensor:
    if seed is None:
        return torch.arange(num_samples, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_samples, generator=generator).to(device)


def validate_payload_size(dataset: str, num_samples: int) -> None:
    expected = EXPECTED_TEST_SPLIT_SIZES.get(dataset)
    if expected is not None and int(num_samples) != int(expected):
        raise ValueError(
            f"Dataset '{dataset}' has {num_samples} samples, expected {expected} from the official split."
        )


def load_payload(dataset: str, device: torch.device, features_dir: str) -> DatasetPayload:
    tda_payload = load_tda_dataset(dataset, device=device, features_dir=features_dir)
    image_features = F.normalize(tda_payload["image_features"], dim=-1)
    text_features = F.normalize(tda_payload["text_features"], dim=-1)
    labels = tda_payload["labels"]
    dataset_key = str(tda_payload["dataset"]).lower()
    validate_payload_size(dataset_key, int(tda_payload["num_samples"]))
    raw_clip_logits = image_features @ text_features.t()
    return DatasetPayload(
        dataset=dataset_key,
        image_features=image_features,
        text_features=text_features,
        labels=labels,
        raw_clip_logits=raw_clip_logits,
    )


def centroid_oracle_accuracy(payload: DatasetPayload) -> float:
    labels = payload.labels
    features = payload.image_features
    centroids = []
    label_to_pos = {}
    unique = torch.unique(labels)
    for pos, label in enumerate(unique):
        mask = labels == label
        centroid = features[mask].mean(dim=0)
        centroid = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)
        centroids.append(centroid)
        label_to_pos[int(label.item())] = pos
    centroid_tensor = torch.stack(centroids, dim=0)
    pred = torch.argmax(features @ centroid_tensor.t(), dim=-1).detach().cpu()
    target = torch.tensor([label_to_pos[int(x.item())] for x in labels.detach().cpu()], dtype=torch.long)
    return float((pred == target).float().mean().item())


def leave_one_out_1nn_accuracy(payload: DatasetPayload, chunk_size: int = 1024) -> float:
    total = payload.num_samples
    feats = payload.image_features
    labels = payload.labels
    correct = 0
    with torch.inference_mode():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            sims = feats[start:end] @ feats.t()
            row_idx = torch.arange(start, end, device=feats.device)
            sims[torch.arange(end - start, device=feats.device), row_idx] = -1e9
            nearest = torch.argmax(sims, dim=-1)
            correct += int((labels[nearest] == labels[start:end]).sum().item())
    return correct / max(total, 1)


def compute_geometry_metrics(payload: DatasetPayload) -> GeometryMetrics:
    centroid = centroid_oracle_accuracy(payload)
    one_nn = leave_one_out_1nn_accuracy(payload)
    return GeometryMetrics(
        oracle_centroid_acc=centroid,
        oracle_1nn_acc=one_nn,
        geometry_alignment_score=centroid - one_nn,
    )


def run_tda_stream(payload: DatasetPayload, order: torch.Tensor, params: dict) -> tuple[dict, pd.DataFrame]:
    model = TDA(text_features=payload.text_features, device=payload.image_features.device, **params)
    rows: list[dict] = []
    clip_correct = 0
    tda_correct = 0
    total = payload.num_samples
    max_entropy = math.log(max(payload.num_classes, 2))

    with torch.inference_mode():
        for stream_step, idx_t in enumerate(order):
            idx = int(idx_t.item())
            x = payload.image_features[idx]
            y = int(payload.labels[idx].item())
            clip_logits = payload.raw_clip_logits[idx].unsqueeze(0)
            clip_probs = torch.softmax(clip_logits, dim=-1)
            clip_pred = int(torch.argmax(clip_logits, dim=-1).item())
            clip_is_correct = int(clip_pred == y)
            clip_correct += clip_is_correct
            clip_conf = float(torch.max(clip_probs, dim=-1).values.item())
            clip_entropy = float(entropy_from_logits(clip_logits, dim=-1).item())
            clip_margin = top2_margin(clip_logits)

            pred, conf, final_logits = model.predict(x)
            final_probs = torch.softmax(final_logits, dim=-1)
            tda_pred = int(pred.item())
            tda_is_correct = int(tda_pred == y)
            tda_correct += tda_is_correct
            tda_conf = float(conf.item())
            tda_entropy = float(entropy_from_logits(final_logits, dim=-1).item())
            tda_margin = top2_margin(final_logits)
            changed = int(tda_pred != clip_pred)
            beneficial_flip = int(changed and clip_is_correct == 0 and tda_is_correct == 1)
            harmful_flip = int(changed and clip_is_correct == 1 and tda_is_correct == 0)
            norm_entropy = clip_entropy / max(max_entropy, 1e-12)

            rows.append(
                {
                    "dataset": payload.dataset,
                    "sample_index": idx,
                    "stream_step": stream_step,
                    "label": y,
                    "clip_pred": clip_pred,
                    "clip_correct": clip_is_correct,
                    "clip_confidence": clip_conf,
                    "clip_entropy": clip_entropy,
                    "clip_margin": clip_margin,
                    "tda_pred": tda_pred,
                    "tda_correct": tda_is_correct,
                    "tda_confidence": tda_conf,
                    "tda_entropy": tda_entropy,
                    "tda_margin": tda_margin,
                    "tda_changed_prediction": changed,
                    "tda_beneficial_flip": beneficial_flip,
                    "tda_harmful_flip": harmful_flip,
                    "tda_positive_cache_size": int(model.pos_size),
                    "tda_negative_cache_size": int(model.neg_size),
                    "tda_negative_gate_open": int(model.low_entropy < norm_entropy < model.high_entropy),
                }
            )

    summary = {
        "dataset": payload.dataset,
        "samples": payload.num_samples,
        "clip_acc": clip_correct / max(total, 1),
        "tda_acc": tda_correct / max(total, 1),
        "tda_gain_vs_clip": (tda_correct - clip_correct) / max(total, 1),
        "tda_final_positive_cache_size": int(model.pos_size),
        "tda_final_negative_cache_size": int(model.neg_size),
        "tda_total_cache_slots": int(model.num_classes * (model.pos_shot_capacity + model.neg_shot_capacity)),
    }
    return summary, pd.DataFrame(rows)


def run_freetta_stream(payload: DatasetPayload, order: torch.Tensor, params: dict) -> tuple[dict, pd.DataFrame]:
    model = FreeTTA(text_features=payload.text_features, device=payload.image_features.device, **params)
    initial_mu = model.mu.detach().clone()
    rows: list[dict] = []
    clip_correct = 0
    freetta_correct = 0
    total = payload.num_samples

    with torch.inference_mode():
        for stream_step, idx_t in enumerate(order):
            idx = int(idx_t.item())
            x = payload.image_features[idx]
            y = int(payload.labels[idx].item())
            clip_logits = payload.raw_clip_logits[idx].unsqueeze(0)
            clip_probs = torch.softmax(clip_logits, dim=-1)
            clip_pred = int(torch.argmax(clip_logits, dim=-1).item())
            clip_is_correct = int(clip_pred == y)
            clip_correct += clip_is_correct
            clip_conf = float(torch.max(clip_probs, dim=-1).values.item())
            clip_entropy = float(entropy_from_logits(clip_logits, dim=-1).item())
            clip_margin = top2_margin(clip_logits)
            em_weight = float(torch.exp(-model.beta * torch.tensor(clip_entropy, device=model.device)).item())
            mu_before = model.mu.detach().clone()

            pred, final_probs = model.predict(x, clip_logits)
            freetta_pred = int(pred.squeeze(0).item())
            freetta_is_correct = int(freetta_pred == y)
            freetta_correct += freetta_is_correct
            freetta_conf = float(torch.max(final_probs, dim=-1).values.item())
            freetta_entropy = float(entropy_from_logits(torch.log(final_probs + 1e-12), dim=-1).item())
            freetta_margin = top2_margin(torch.log(final_probs + 1e-12))
            changed = int(freetta_pred != clip_pred)
            beneficial_flip = int(changed and clip_is_correct == 0 and freetta_is_correct == 1)
            harmful_flip = int(changed and clip_is_correct == 1 and freetta_is_correct == 0)
            mu_update_norm = float(torch.norm(model.mu - mu_before, dim=1).mean().item())
            mu_drift = float(torch.norm(model.mu - initial_mu, dim=1).mean().item())
            priors = (model.Ny / (model.t + 1e-8)).clamp_min(1e-12)
            prior_entropy = float((-(priors * torch.log(priors))).sum().item())
            sigma_trace = float(torch.trace(model.sigma).item())

            rows.append(
                {
                    "dataset": payload.dataset,
                    "sample_index": idx,
                    "stream_step": stream_step,
                    "label": y,
                    "clip_pred": clip_pred,
                    "clip_correct": clip_is_correct,
                    "clip_confidence": clip_conf,
                    "clip_entropy": clip_entropy,
                    "clip_margin": clip_margin,
                    "freetta_pred": freetta_pred,
                    "freetta_correct": freetta_is_correct,
                    "freetta_confidence": freetta_conf,
                    "freetta_entropy": freetta_entropy,
                    "freetta_margin": freetta_margin,
                    "freetta_changed_prediction": changed,
                    "freetta_beneficial_flip": beneficial_flip,
                    "freetta_harmful_flip": harmful_flip,
                    "freetta_em_weight": em_weight,
                    "freetta_mu_update_norm": mu_update_norm,
                    "freetta_mu_drift": mu_drift,
                    "freetta_prior_entropy": prior_entropy,
                    "freetta_sigma_trace": sigma_trace,
                }
            )

    summary = {
        "dataset": payload.dataset,
        "samples": payload.num_samples,
        "clip_acc": clip_correct / max(total, 1),
        "freetta_acc": freetta_correct / max(total, 1),
        "freetta_gain_vs_clip": (freetta_correct - clip_correct) / max(total, 1),
        "freetta_final_mu_drift": float(torch.norm(model.mu - initial_mu, dim=1).mean().item()),
        "freetta_final_prior_entropy": float((-(priors * torch.log(priors))).sum().item()),
        "freetta_final_sigma_trace": float(torch.trace(model.sigma).item()),
    }
    return summary, pd.DataFrame(rows)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def compute_flip_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        for method in ("tda", "freetta"):
            changed = group[f"{method}_changed_prediction"] == 1
            beneficial = group[f"{method}_beneficial_flip"] == 1
            harmful = group[f"{method}_harmful_flip"] == 1
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "change_rate": float(changed.mean()),
                    "beneficial_flip_count": int(beneficial.sum()),
                    "harmful_flip_count": int(harmful.sum()),
                    "beneficial_flip_precision": float(beneficial.sum() / max(int(changed.sum()), 1)),
                    "harmful_flip_rate_on_clip_correct": float(
                        harmful.sum() / max(int((group["clip_correct"] == 1).sum()), 1)
                    ),
                    "avg_entropy_after_beneficial_flip": safe_mean(group.loc[beneficial, f"{method}_entropy"]),
                    "avg_entropy_after_harmful_flip": safe_mean(group.loc[harmful, f"{method}_entropy"]),
                }
            )
    return pd.DataFrame(rows)


def compute_disagreement_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        disagreement = group[group["tda_pred"] != group["freetta_pred"]]
        rows.append(
            {
                "dataset": dataset,
                "disagreement_rate": float(len(disagreement) / max(len(group), 1)),
                "tda_acc_on_disagreement": float(disagreement["tda_correct"].mean()) if len(disagreement) else 0.0,
                "freetta_acc_on_disagreement": float(disagreement["freetta_correct"].mean()) if len(disagreement) else 0.0,
                "avg_clip_entropy_on_disagreement": float(disagreement["clip_entropy"].mean()) if len(disagreement) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_latency_metrics(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    rolling_rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        group = group.sort_values("stream_step").reset_index(drop=True)
        total = len(group)
        window = max(25, total // 20)
        clip_roll = group["clip_correct"].astype(float).rolling(window, min_periods=window).mean()
        tda_roll = group["tda_correct"].astype(float).rolling(window, min_periods=window).mean()
        freetta_roll = group["freetta_correct"].astype(float).rolling(window, min_periods=window).mean()
        tda_vs_clip = tda_roll - clip_roll
        freetta_vs_clip = freetta_roll - clip_roll
        freetta_vs_tda = freetta_roll - tda_roll

        def first_positive(series: pd.Series) -> int | None:
            for idx, value in enumerate(series.tolist(), start=1):
                if pd.notna(value) and value > 0:
                    return idx
            return None

        tda_be = first_positive(tda_vs_clip)
        freetta_be = first_positive(freetta_vs_clip)
        freetta_tda_be = first_positive(freetta_vs_tda)

        summary_rows.append(
            {
                "dataset": dataset,
                "window": window,
                "tda_break_even_vs_clip": tda_be,
                "freetta_break_even_vs_clip": freetta_be,
                "freetta_break_even_vs_tda": freetta_tda_be,
                "tda_break_even_ratio": (tda_be / total) if tda_be is not None else float("nan"),
                "freetta_break_even_ratio": (freetta_be / total) if freetta_be is not None else float("nan"),
                "freetta_vs_tda_break_even_ratio": (freetta_tda_be / total) if freetta_tda_be is not None else float("nan"),
            }
        )

        for idx in range(total):
            rolling_rows.append(
                {
                    "dataset": dataset,
                    "stream_step": idx + 1,
                    "progress_ratio": (idx + 1) / total,
                    "rolling_tda_vs_clip": float(tda_vs_clip.iloc[idx]) if pd.notna(tda_vs_clip.iloc[idx]) else float("nan"),
                    "rolling_freetta_vs_clip": float(freetta_vs_clip.iloc[idx]) if pd.notna(freetta_vs_clip.iloc[idx]) else float("nan"),
                    "rolling_freetta_vs_tda": float(freetta_vs_tda.iloc[idx]) if pd.notna(freetta_vs_tda.iloc[idx]) else float("nan"),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(rolling_rows)


def compute_difficulty_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    ranked = merged["clip_entropy"].rank(method="first")
    bins = pd.qcut(ranked, q=3, labels=["easy", "medium", "hard"])
    tmp = merged.copy()
    tmp["difficulty_bin"] = bins
    rows: list[dict] = []
    for (dataset, difficulty), group in tmp.groupby(["dataset", "difficulty_bin"], observed=True):
        rows.append(
            {
                "dataset": dataset,
                "difficulty_bin": str(difficulty),
                "samples": int(len(group)),
                "clip_acc": float(group["clip_correct"].mean()),
                "tda_acc": float(group["tda_correct"].mean()),
                "freetta_acc": float(group["freetta_correct"].mean()),
                "freetta_minus_tda": float(group["freetta_correct"].mean() - group["tda_correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_internal_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        geometry_alignment = float(group["geometry_alignment_score"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "tda_mean_positive_cache_size": float(group["tda_positive_cache_size"].mean()),
                "tda_mean_negative_cache_size": float(group["tda_negative_cache_size"].mean()),
                "tda_negative_gate_rate": float(group["tda_negative_gate_open"].mean()),
                "tda_cache_pressure_ratio": float(group["samples"].iloc[0] / max(group["tda_total_cache_slots"].iloc[0], 1)),
                "freetta_mean_em_weight": float(group["freetta_em_weight"].mean()),
                "freetta_mean_mu_update_norm": float(group["freetta_mu_update_norm"].mean()),
                "freetta_final_mu_drift": float(group["freetta_mu_drift"].iloc[-1]),
                "freetta_final_prior_entropy": float(group["freetta_prior_entropy"].iloc[-1]),
                "freetta_final_sigma_trace": float(group["freetta_sigma_trace"].iloc[-1]),
                "geometry_alignment_score": geometry_alignment,
                "oracle_centroid_acc": float(group["oracle_centroid_acc"].iloc[0]),
                "oracle_1nn_acc": float(group["oracle_1nn_acc"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_path: Path,
    summary_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    flip_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
    internal_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Comparative Analysis Outputs")
    lines.append("")
    lines.append("## Accuracy Summary")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Geometry Probe")
    lines.append(geometry_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Flip Efficiency")
    lines.append(flip_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Adaptation Latency")
    lines.append(latency_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Difficulty-Conditioned Comparison")
    lines.append(difficulty_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Disagreement Comparison")
    lines.append(disagreement_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Internal Mechanism Metrics")
    lines.append(internal_df.to_markdown(index=False))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single consolidated TDA vs FreeTTA analysis pipeline")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/comparative_analysis")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stream-seed", type=int, default=None, help="Use None for natural order, or set a seed for shuffled order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    summary_rows: list[dict] = []
    geometry_rows: list[dict] = []
    merged_frames: list[pd.DataFrame] = []

    for dataset in [str(x).lower() for x in args.datasets]:
        payload = load_payload(dataset, device=device, features_dir=args.features_dir)
        order = get_order(payload.num_samples, device=device, seed=args.stream_seed)
        tda_summary, tda_rows = run_tda_stream(payload, order, PAPER_TDA_DEFAULTS[payload.dataset])
        freetta_summary, freetta_rows = run_freetta_stream(payload, order, DEFAULT_FREETTA_PARAMS[payload.dataset])
        geometry = compute_geometry_metrics(payload)
        geometry_row = {
            "dataset": payload.dataset,
            "oracle_centroid_acc": geometry.oracle_centroid_acc,
            "oracle_1nn_acc": geometry.oracle_1nn_acc,
            "geometry_alignment_score": geometry.geometry_alignment_score,
        }
        geometry_rows.append(geometry_row)

        merged = tda_rows.merge(
            freetta_rows[
                [
                    "sample_index",
                    "freetta_pred",
                    "freetta_correct",
                    "freetta_confidence",
                    "freetta_entropy",
                    "freetta_margin",
                    "freetta_changed_prediction",
                    "freetta_beneficial_flip",
                    "freetta_harmful_flip",
                    "freetta_em_weight",
                    "freetta_mu_update_norm",
                    "freetta_mu_drift",
                    "freetta_prior_entropy",
                    "freetta_sigma_trace",
                ]
            ],
            on="sample_index",
            how="inner",
        )
        merged["samples"] = payload.num_samples
        merged["tda_total_cache_slots"] = tda_summary["tda_total_cache_slots"]
        for key, value in geometry_row.items():
            if key != "dataset":
                merged[key] = value
        merged_frames.append(merged)

        summary_rows.append(
            {
                "dataset": payload.dataset,
                "samples": payload.num_samples,
                "clip_acc": float(tda_summary["clip_acc"]),
                "tda_acc": float(tda_summary["tda_acc"]),
                "freetta_acc": float(freetta_summary["freetta_acc"]),
                "tda_gain_vs_clip": float(tda_summary["tda_gain_vs_clip"]),
                "freetta_gain_vs_clip": float(freetta_summary["freetta_gain_vs_clip"]),
                "freetta_minus_tda": float(freetta_summary["freetta_acc"] - tda_summary["tda_acc"]),
                "tda_final_positive_cache_size": int(tda_summary["tda_final_positive_cache_size"]),
                "tda_final_negative_cache_size": int(tda_summary["tda_final_negative_cache_size"]),
                "freetta_final_mu_drift": float(freetta_summary["freetta_final_mu_drift"]),
                "freetta_final_prior_entropy": float(freetta_summary["freetta_final_prior_entropy"]),
                "freetta_final_sigma_trace": float(freetta_summary["freetta_final_sigma_trace"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    geometry_df = pd.DataFrame(geometry_rows).sort_values("dataset").reset_index(drop=True)
    merged_df = pd.concat(merged_frames, ignore_index=True)
    flip_df = compute_flip_metrics(merged_df)
    disagreement_df = compute_disagreement_metrics(merged_df)
    latency_df, latency_curve_df = compute_latency_metrics(merged_df)
    difficulty_df = compute_difficulty_metrics(merged_df)
    internal_df = compute_internal_metrics(merged_df)

    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    geometry_df.to_csv(output_dir / "geometry_metrics.csv", index=False)
    merged_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    flip_df.to_csv(output_dir / "flip_metrics.csv", index=False)
    disagreement_df.to_csv(output_dir / "disagreement_metrics.csv", index=False)
    latency_df.to_csv(output_dir / "latency_metrics.csv", index=False)
    latency_curve_df.to_csv(output_dir / "latency_curves.csv", index=False)
    difficulty_df.to_csv(output_dir / "difficulty_metrics.csv", index=False)
    internal_df.to_csv(output_dir / "internal_metrics.csv", index=False)

    report = {
        "device": str(device),
        "datasets": [str(x).lower() for x in args.datasets],
        "stream_seed": args.stream_seed,
        "summary_metrics_csv": str(output_dir / "summary_metrics.csv"),
        "geometry_metrics_csv": str(output_dir / "geometry_metrics.csv"),
        "per_sample_metrics_csv": str(output_dir / "per_sample_metrics.csv"),
        "flip_metrics_csv": str(output_dir / "flip_metrics.csv"),
        "disagreement_metrics_csv": str(output_dir / "disagreement_metrics.csv"),
        "latency_metrics_csv": str(output_dir / "latency_metrics.csv"),
        "difficulty_metrics_csv": str(output_dir / "difficulty_metrics.csv"),
        "internal_metrics_csv": str(output_dir / "internal_metrics.csv"),
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_summary(
        output_dir / "analysis_summary.md",
        summary_df,
        geometry_df,
        flip_df,
        latency_df,
        difficulty_df,
        disagreement_df,
        internal_df,
    )

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
