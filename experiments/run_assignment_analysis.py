from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.feature_store import load_dataset_features
from src.paper_configs import DEFAULT_FREETTA_PARAMS, PAPER_TDA_DEFAULTS


BEST_TDA_PARAMS = {key: dict(value) for key, value in PAPER_TDA_DEFAULTS.items()}
BEST_FREETTA_PARAMS = {key: dict(value) for key, value in DEFAULT_FREETTA_PARAMS.items()}


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


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=dim)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=dim)


def get_order(num_samples: int, seed: int | None, device: torch.device) -> torch.Tensor:
    if seed is None:
        return torch.arange(num_samples, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_samples, generator=generator).to(device)


def load_payload(features_dir: Path, dataset: str, device: torch.device) -> DatasetPayload:
    raw = load_dataset_features(features_dir, dataset)
    image_features = torch.as_tensor(raw["image_features"], dtype=torch.float32, device=device)
    text_features = torch.as_tensor(raw["text_features"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(raw["labels"], dtype=torch.long, device=device)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    raw_clip_logits = image_features @ text_features.t()

    return DatasetPayload(
        dataset=str(raw["dataset_key"]).lower(),
        image_features=image_features,
        text_features=text_features,
        labels=labels,
        raw_clip_logits=raw_clip_logits,
    )


def phase_label(step: int, total: int) -> str:
    frac = (step + 1) / max(total, 1)
    if frac <= 0.25:
        return "early"
    if frac <= 0.75:
        return "middle"
    return "late"


def difficulty_bins(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=3, labels=["easy", "medium", "hard"])


def run_tda_stream(
    payload: DatasetPayload,
    params: dict,
    order: torch.Tensor,
    collect_rows: bool = True,
) -> tuple[dict, pd.DataFrame]:
    model = TDA(text_features=payload.text_features, device=payload.image_features.device, **params)
    total = payload.num_samples
    num_classes = int(payload.text_features.shape[0])
    max_entropy = math.log2(max(num_classes, 2))

    correct = 0
    clip_correct = 0
    rows: list[dict] = []

    with torch.inference_mode():
        for stream_step, idx_t in enumerate(order):
            idx = int(idx_t.item())
            x = payload.image_features[idx]
            y = int(payload.labels[idx].item())
            raw_clip_logits = payload.raw_clip_logits[idx]
            shared_clip_probs = torch.softmax(raw_clip_logits, dim=-1)
            shared_entropy = float(entropy_from_logits(raw_clip_logits).item())
            shared_margin = float(
                (torch.topk(raw_clip_logits, k=2).values[0] - torch.topk(raw_clip_logits, k=2).values[1]).item()
            )

            scaled_clip_logits = model.clip_scale * raw_clip_logits.unsqueeze(0)
            scaled_clip_probs = torch.softmax(scaled_clip_logits, dim=-1)
            clip_pred = int(torch.argmax(scaled_clip_logits, dim=-1).item())
            clip_conf = float(torch.max(scaled_clip_probs, dim=-1).values.item())
            clip_entropy = float(entropy_from_logits(scaled_clip_logits, dim=-1).item())
            norm_entropy = clip_entropy / max(max_entropy, 1e-12)

            pred, final_conf_tensor, returned_logits = model.predict(x)
            final_pred = int(pred.item())
            final_conf = float(final_conf_tensor.item())

            x_2d = x.unsqueeze(0)
            raw_fused_logits = scaled_clip_logits.clone()
            raw_fused_logits += model._compute_cache_logits(  # noqa: SLF001
                x_2d,
                cache=model.pos_cache,
                alpha=model.alpha,
                beta=model.beta,
                negative=False,
            )
            raw_fused_logits -= model._compute_cache_logits(  # noqa: SLF001
                x_2d,
                cache=model.neg_cache,
                alpha=model.neg_alpha,
                beta=model.neg_beta,
                negative=True,
            )
            raw_fused_probs = torch.softmax(raw_fused_logits, dim=-1)
            fused_conf_pre_fallback = float(torch.max(raw_fused_probs, dim=-1).values.item())
            fallback_used = bool(
                model.fallback_to_clip and (fused_conf_pre_fallback + model.fallback_margin < clip_conf)
            )

            clip_is_correct = int(clip_pred == y)
            final_is_correct = int(final_pred == y)
            clip_correct += clip_is_correct
            correct += final_is_correct

            if collect_rows:
                rows.append(
                    {
                        "dataset": payload.dataset,
                        "sample_index": idx,
                        "stream_step": stream_step,
                        "phase": phase_label(stream_step, total),
                        "label": y,
                        "shared_clip_entropy": shared_entropy,
                        "shared_clip_margin": shared_margin,
                        "clip_pred": clip_pred,
                        "clip_correct": clip_is_correct,
                        "tda_pred": final_pred,
                        "tda_correct": final_is_correct,
                        "tda_changed_prediction": int(final_pred != clip_pred),
                        "tda_clip_confidence": clip_conf,
                        "tda_final_confidence": final_conf,
                        "tda_fused_confidence_pre_fallback": fused_conf_pre_fallback,
                        "tda_entropy_scaled": clip_entropy,
                        "tda_norm_entropy_scaled": norm_entropy,
                        "tda_positive_cache_size": int(model.pos_size),
                        "tda_negative_cache_size": int(model.neg_size),
                        "tda_negative_cache_active": int(model.neg_size > 0),
                        "tda_fallback_used": int(fallback_used),
                        "tda_negative_gate_open": int(model.low_entropy < norm_entropy < model.high_entropy),
                    }
                )

    summary = {
        "dataset": payload.dataset,
        "samples": total,
        "clip_acc": clip_correct / max(total, 1),
        "tda_acc": correct / max(total, 1),
        "tda_gain_vs_clip": (correct - clip_correct) / max(total, 1),
        "tda_final_positive_cache_size": int(model.pos_size),
        "tda_final_negative_cache_size": int(model.neg_size),
    }
    return summary, pd.DataFrame(rows)


def run_freetta_stream(
    payload: DatasetPayload,
    params: dict,
    order: torch.Tensor,
    collect_rows: bool = True,
) -> tuple[dict, pd.DataFrame]:
    model = FreeTTA(text_features=payload.text_features, device=payload.image_features.device, **params)
    initial_mu = model.mu.detach().clone()
    total = payload.num_samples

    clip_correct = 0
    correct = 0
    rows: list[dict] = []

    with torch.inference_mode():
        for stream_step, idx_t in enumerate(order):
            idx = int(idx_t.item())
            x = payload.image_features[idx]
            y = int(payload.labels[idx].item())
            raw_clip_logits = payload.raw_clip_logits[idx]
            clip_logits = raw_clip_logits.unsqueeze(0)
            clip_probs = torch.softmax(clip_logits, dim=-1)
            clip_pred = int(torch.argmax(clip_logits, dim=-1).item())
            clip_conf = float(torch.max(clip_probs, dim=-1).values.item())
            clip_entropy = float(entropy_from_logits(clip_logits, dim=-1).item())
            shared_margin = float(
                (torch.topk(raw_clip_logits, k=2).values[0] - torch.topk(raw_clip_logits, k=2).values[1]).item()
            )
            weight = float(torch.exp(-model.beta * torch.tensor(clip_entropy, device=model.device)).item())

            mu_before = model.mu.detach().clone()
            pred, final_probs = model.predict(x, clip_logits)
            final_pred = int(pred.squeeze(0).item())
            final_conf = float(torch.max(final_probs, dim=-1).values.item())
            posterior_entropy = float(entropy_from_logits(torch.log(final_probs + 1e-12), dim=-1).item())
            mu_update_norm = float(torch.norm(model.mu - mu_before, dim=1).mean().item())
            mu_drift = float(torch.norm(model.mu - initial_mu, dim=1).mean().item())
            priors = (model.Ny / (model.t + 1e-8)).clamp_min(1e-12)
            prior_entropy = float((-(priors * torch.log(priors))).sum().item())

            clip_is_correct = int(clip_pred == y)
            final_is_correct = int(final_pred == y)
            clip_correct += clip_is_correct
            correct += final_is_correct

            if collect_rows:
                rows.append(
                    {
                        "dataset": payload.dataset,
                        "sample_index": idx,
                        "stream_step": stream_step,
                        "phase": phase_label(stream_step, total),
                        "label": y,
                        "shared_clip_entropy": clip_entropy,
                        "shared_clip_margin": shared_margin,
                        "clip_pred": clip_pred,
                        "clip_correct": clip_is_correct,
                        "freetta_pred": final_pred,
                        "freetta_correct": final_is_correct,
                        "freetta_changed_prediction": int(final_pred != clip_pred),
                        "freetta_clip_confidence": clip_conf,
                        "freetta_final_confidence": final_conf,
                        "freetta_entropy": clip_entropy,
                        "freetta_em_weight": weight,
                        "freetta_mu_update_norm": mu_update_norm,
                        "freetta_mu_drift": mu_drift,
                        "freetta_prior_entropy": prior_entropy,
                        "freetta_posterior_entropy": posterior_entropy,
                    }
                )

    summary = {
        "dataset": payload.dataset,
        "samples": total,
        "clip_acc": clip_correct / max(total, 1),
        "freetta_acc": correct / max(total, 1),
        "freetta_gain_vs_clip": (correct - clip_correct) / max(total, 1),
        "freetta_final_mu_drift": float(torch.norm(model.mu - initial_mu, dim=1).mean().item()),
        "freetta_final_prior_entropy": float(
            (-(model.Ny / (model.t + 1e-8)).clamp_min(1e-12) * torch.log((model.Ny / (model.t + 1e-8)).clamp_min(1e-12)))
            .sum()
            .item()
        ),
    }
    return summary, pd.DataFrame(rows)


def summarize_conditioned_metrics(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entropy_df = merged.copy()
    entropy_df["difficulty_bin"] = difficulty_bins(entropy_df["shared_clip_entropy"])

    entropy_rows: list[dict] = []
    for (dataset, difficulty_bin), group in entropy_df.groupby(["dataset", "difficulty_bin"], observed=True):
        entropy_rows.append(
            {
                "dataset": dataset,
                "difficulty_bin": str(difficulty_bin),
                "samples": int(len(group)),
                "clip_acc": float(group["clip_correct"].mean()),
                "tda_acc": float(group["tda_correct"].mean()),
                "freetta_acc": float(group["freetta_correct"].mean()),
                "freetta_minus_tda": float(group["freetta_correct"].mean() - group["tda_correct"].mean()),
            }
        )

    phase_rows: list[dict] = []
    for (dataset, phase), group in merged.groupby(["dataset", "phase"], observed=True):
        phase_rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "samples": int(len(group)),
                "clip_acc": float(group["clip_correct"].mean()),
                "tda_acc": float(group["tda_correct"].mean()),
                "freetta_acc": float(group["freetta_correct"].mean()),
                "freetta_minus_tda": float(group["freetta_correct"].mean() - group["tda_correct"].mean()),
            }
        )

    disagreement_rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        disagreement = group[group["tda_pred"] != group["freetta_pred"]]
        size = int(len(disagreement))
        disagreement_rows.append(
            {
                "dataset": dataset,
                "disagreement_samples": size,
                "disagreement_rate": float(size / max(len(group), 1)),
                "tda_acc_on_disagreement": float(disagreement["tda_correct"].mean()) if size else 0.0,
                "freetta_acc_on_disagreement": float(disagreement["freetta_correct"].mean()) if size else 0.0,
                "freetta_wins_on_disagreement": int(
                    ((disagreement["freetta_correct"] == 1) & (disagreement["tda_correct"] == 0)).sum()
                )
                if size
                else 0,
                "tda_wins_on_disagreement": int(
                    ((disagreement["tda_correct"] == 1) & (disagreement["freetta_correct"] == 0)).sum()
                )
                if size
                else 0,
            }
        )

    return pd.DataFrame(entropy_rows), pd.DataFrame(phase_rows), pd.DataFrame(disagreement_rows)


def summarize_internal_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        tda_correct_group = group[group["tda_correct"] == 1]
        tda_wrong_group = group[group["tda_correct"] == 0]
        freetta_correct_group = group[group["freetta_correct"] == 1]
        freetta_wrong_group = group[group["freetta_correct"] == 0]

        rows.append(
            {
                "dataset": dataset,
                "tda_mean_positive_cache_size": float(group["tda_positive_cache_size"].mean()),
                "tda_mean_negative_cache_size": float(group["tda_negative_cache_size"].mean()),
                "tda_fallback_rate": float(group["tda_fallback_used"].mean()),
                "tda_negative_gate_rate": float(group["tda_negative_gate_open"].mean()),
                "tda_accuracy_when_negative_cache_active": float(
                    group.loc[group["tda_negative_cache_active"] == 1, "tda_correct"].mean()
                ),
                "tda_clip_conf_correct": float(tda_correct_group["tda_clip_confidence"].mean()),
                "tda_clip_conf_wrong": float(tda_wrong_group["tda_clip_confidence"].mean())
                if len(tda_wrong_group)
                else float("nan"),
                "freetta_mean_em_weight": float(group["freetta_em_weight"].mean()),
                "freetta_mean_mu_update_norm": float(group["freetta_mu_update_norm"].mean()),
                "freetta_final_mu_drift": float(group["freetta_mu_drift"].iloc[-1]),
                "freetta_weight_correct": float(freetta_correct_group["freetta_em_weight"].mean()),
                "freetta_weight_wrong": float(freetta_wrong_group["freetta_em_weight"].mean())
                if len(freetta_wrong_group)
                else float("nan"),
                "freetta_mu_update_correct": float(freetta_correct_group["freetta_mu_update_norm"].mean()),
                "freetta_mu_update_wrong": float(freetta_wrong_group["freetta_mu_update_norm"].mean())
                if len(freetta_wrong_group)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def centroid_oracle_accuracy(payload: DatasetPayload) -> float:
    labels = payload.labels
    unique_labels = torch.unique(labels)
    centroids = []
    label_lookup = {}
    for class_pos, label in enumerate(unique_labels):
        class_mask = labels == label
        centroid = payload.image_features[class_mask].mean(dim=0)
        centroid = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)
        centroids.append(centroid)
        label_lookup[int(label.item())] = class_pos

    centroid_tensor = torch.stack(centroids, dim=0)
    logits = payload.image_features @ centroid_tensor.t()
    pred_positions = torch.argmax(logits, dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    target_positions = torch.tensor([label_lookup[int(x.item())] for x in labels_cpu], dtype=torch.long)
    return float((pred_positions == target_positions).float().mean().item())


def leave_one_out_1nn_accuracy(payload: DatasetPayload, chunk_size: int = 1024) -> float:
    total = payload.num_samples
    correct = 0
    features = payload.image_features
    labels = payload.labels

    with torch.inference_mode():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            sims = features[start:end] @ features.t()
            row_indices = torch.arange(start, end, device=features.device)
            sims[torch.arange(end - start, device=features.device), row_indices] = -1e9
            nearest = torch.argmax(sims, dim=-1)
            correct += int((labels[nearest] == labels[start:end]).sum().item())

    return correct / max(total, 1)


def geometry_probe(payload: DatasetPayload) -> dict:
    return {
        "dataset": payload.dataset,
        "oracle_centroid_acc": centroid_oracle_accuracy(payload),
        "oracle_leave_one_out_1nn_acc": leave_one_out_1nn_accuracy(payload),
    }


def run_order_sensitivity(
    payload: DatasetPayload,
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in seeds:
        order = get_order(payload.num_samples, int(seed), payload.image_features.device)
        tda_summary, _ = run_tda_stream(payload, BEST_TDA_PARAMS[payload.dataset], order, collect_rows=False)
        freetta_summary, _ = run_freetta_stream(payload, BEST_FREETTA_PARAMS[payload.dataset], order, collect_rows=False)
        rows.append(
            {
                "dataset": payload.dataset,
                "seed": int(seed),
                "tda_acc": float(tda_summary["tda_acc"]),
                "freetta_acc": float(freetta_summary["freetta_acc"]),
                "freetta_minus_tda": float(freetta_summary["freetta_acc"] - tda_summary["tda_acc"]),
            }
        )
    return pd.DataFrame(rows)


def write_plot_accuracy(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(summary_df))
    width = 0.24

    ax.bar([i - width for i in x], summary_df["clip_acc"] * 100.0, width=width, label="CLIP")
    ax.bar(x, summary_df["tda_acc"] * 100.0, width=width, label="TDA")
    ax.bar([i + width for i in x], summary_df["freetta_acc"] * 100.0, width=width, label="FreeTTA")

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["dataset"].tolist())
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison Across Datasets")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_plot_order_sensitivity(order_df: pd.DataFrame, output_path: Path) -> None:
    agg = (
        order_df.groupby("dataset", observed=True)
        .agg(
            tda_mean=("tda_acc", "mean"),
            tda_std=("tda_acc", "std"),
            freetta_mean=("freetta_acc", "mean"),
            freetta_std=("freetta_acc", "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(agg))
    width = 0.28

    ax.bar(
        [i - width / 2 for i in x],
        agg["tda_mean"] * 100.0,
        yerr=agg["tda_std"].fillna(0.0) * 100.0,
        width=width,
        capsize=4,
        label="TDA",
    )
    ax.bar(
        [i + width / 2 for i in x],
        agg["freetta_mean"] * 100.0,
        yerr=agg["freetta_std"].fillna(0.0) * 100.0,
        width=width,
        capsize=4,
        label="FreeTTA",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["dataset"].tolist())
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Order Sensitivity Across Random Stream Seeds")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate assignment-ready TDA vs FreeTTA analysis.")
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/assignment_analysis"))
    parser.add_argument("--datasets", nargs="*", default=["dtd", "caltech", "eurosat", "pets", "imagenet"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--order-seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    geometry_rows: list[dict] = []
    merged_datasets: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []

    for dataset in [str(x).lower() for x in args.datasets]:
        payload = load_payload(args.features_dir, dataset, device)
        print(f"[Run] dataset={payload.dataset} samples={payload.num_samples} device={device}")

        natural_order = get_order(payload.num_samples, None, device)
        tda_summary, tda_rows = run_tda_stream(payload, BEST_TDA_PARAMS[payload.dataset], natural_order, collect_rows=True)
        freetta_summary, freetta_rows = run_freetta_stream(
            payload,
            BEST_FREETTA_PARAMS[payload.dataset],
            natural_order,
            collect_rows=True,
        )

        merged = tda_rows.merge(
            freetta_rows[
                [
                    "sample_index",
                    "freetta_pred",
                    "freetta_correct",
                    "freetta_changed_prediction",
                    "freetta_clip_confidence",
                    "freetta_final_confidence",
                    "freetta_em_weight",
                    "freetta_mu_update_norm",
                    "freetta_mu_drift",
                    "freetta_prior_entropy",
                    "freetta_posterior_entropy",
                ]
            ],
            on="sample_index",
            how="inner",
        )
        merged["dataset"] = payload.dataset
        merged_datasets.append(merged)

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
            }
        )

        geometry_rows.append(geometry_probe(payload))
        if args.order_seeds:
            order_frames.append(run_order_sensitivity(payload, args.order_seeds))

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    geometry_df = pd.DataFrame(geometry_rows).sort_values("dataset").reset_index(drop=True)
    merged_df = pd.concat(merged_datasets, ignore_index=True)
    order_df = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    entropy_df, phase_df, disagreement_df = summarize_conditioned_metrics(merged_df)
    internal_df = summarize_internal_metrics(merged_df)

    if order_df.empty:
        order_summary_df = pd.DataFrame(
            columns=[
                "dataset",
                "tda_mean",
                "tda_std",
                "freetta_mean",
                "freetta_std",
                "freetta_better_seeds",
                "tda_better_seeds",
            ]
        )
    else:
        order_summary_df = (
            order_df.groupby("dataset", observed=True)
            .agg(
                tda_mean=("tda_acc", "mean"),
                tda_std=("tda_acc", "std"),
                freetta_mean=("freetta_acc", "mean"),
                freetta_std=("freetta_acc", "std"),
                freetta_better_seeds=("freetta_minus_tda", lambda s: int((s > 0).sum())),
                tda_better_seeds=("freetta_minus_tda", lambda s: int((s < 0).sum())),
            )
            .reset_index()
        )

    summary_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    geometry_df.to_csv(args.output_dir / "geometry_probe.csv", index=False)
    merged_df.to_csv(args.output_dir / "per_sample_metrics.csv", index=False)
    entropy_df.to_csv(args.output_dir / "entropy_conditioned_metrics.csv", index=False)
    phase_df.to_csv(args.output_dir / "stream_phase_metrics.csv", index=False)
    disagreement_df.to_csv(args.output_dir / "disagreement_metrics.csv", index=False)
    internal_df.to_csv(args.output_dir / "internal_metrics.csv", index=False)
    order_df.to_csv(args.output_dir / "order_sensitivity_runs.csv", index=False)
    order_summary_df.to_csv(args.output_dir / "order_sensitivity_summary.csv", index=False)

    write_plot_accuracy(summary_df, args.output_dir / "accuracy_overview.png")
    if not order_df.empty:
        write_plot_order_sensitivity(order_df, args.output_dir / "order_sensitivity.png")

    run_report = {
        "device": str(device),
        "datasets": [str(x).lower() for x in args.datasets],
        "summary_metrics_csv": str(args.output_dir / "summary_metrics.csv"),
        "geometry_probe_csv": str(args.output_dir / "geometry_probe.csv"),
        "entropy_conditioned_csv": str(args.output_dir / "entropy_conditioned_metrics.csv"),
        "stream_phase_csv": str(args.output_dir / "stream_phase_metrics.csv"),
        "disagreement_csv": str(args.output_dir / "disagreement_metrics.csv"),
        "internal_metrics_csv": str(args.output_dir / "internal_metrics.csv"),
        "order_sensitivity_csv": str(args.output_dir / "order_sensitivity_summary.csv"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print("\nSaved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
