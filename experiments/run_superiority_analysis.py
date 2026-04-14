from __future__ import annotations

import argparse
import gc
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.feature_store import list_available_datasets, load_dataset_features
from src.paper_configs import DEFAULT_DATASETS, DEFAULT_FREETTA_PARAMS, PAPER_TDA_DEFAULTS


BEST_TDA_PARAMS = PAPER_TDA_DEFAULTS
BEST_FREETTA_PARAMS = DEFAULT_FREETTA_PARAMS
FREETTA_ALPHA_GRID = (0.05, 0.1, 0.2, 0.3)
FREETTA_BETA_GRID = (0.1, 0.5, 1.0, 2.0, 4.5)
TDA_SHOT_GRID = (1, 2, 3, 5, 10)


@dataclass
class DatasetPayload:
    dataset: str
    image_features: torch.Tensor
    text_features: torch.Tensor
    labels: torch.Tensor
    raw_clip_logits: torch.Tensor
    raw_clip_probs: torch.Tensor
    raw_clip_entropy: torch.Tensor
    original_num_samples: int

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.text_features.shape[0])


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def should_log_progress(step: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    if step in (1, total):
        return True
    return interval > 0 and step % interval == 0


def release_gpu_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def make_axes_grid(num_panels: int, ncols: int = 2, figsize_per_panel: tuple[float, float] = (5.2, 3.9)):
    ncols = max(1, min(ncols, num_panels))
    nrows = max(1, math.ceil(num_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    return fig, axes


def entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-12)).sum(dim=dim)


def safe_mean(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def top2_margin(logits: torch.Tensor) -> float:
    flat = logits.squeeze(0) if logits.dim() == 2 else logits
    top_vals = torch.topk(flat, k=min(2, flat.numel())).values
    if top_vals.numel() < 2:
        return float("nan")
    return float((top_vals[0] - top_vals[1]).item())


def choose_subset_indices(
    labels: np.ndarray,
    max_samples: int | None,
    subset_mode: str,
    sample_seed: int,
) -> np.ndarray:
    total = int(labels.shape[0])
    if max_samples is None or max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(int(sample_seed))
    if subset_mode == "head":
        return np.arange(max_samples, dtype=np.int64)
    if subset_mode == "random":
        return np.sort(rng.choice(total, size=max_samples, replace=False).astype(np.int64))

    unique_labels = np.unique(labels)
    buckets: list[np.ndarray] = []
    for label in unique_labels:
        bucket = np.where(labels == label)[0].astype(np.int64)
        rng.shuffle(bucket)
        buckets.append(bucket)

    selected: list[int] = []
    depth = 0
    while len(selected) < max_samples:
        progressed = False
        for bucket in buckets:
            if depth < len(bucket):
                selected.append(int(bucket[depth]))
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        depth += 1

    return np.sort(np.asarray(selected, dtype=np.int64))


def load_payload(
    features_dir: Path,
    dataset: str,
    device: torch.device,
    max_samples: int | None = None,
    subset_mode: str = "stratified",
    sample_seed: int = 1,
) -> DatasetPayload:
    raw = load_dataset_features(features_dir, dataset)
    raw_image_features = raw["image_features"]
    raw_text_features = raw["text_features"]
    raw_labels = raw["labels"]
    original_num_samples = int(raw_labels.shape[0])

    subset_indices = choose_subset_indices(
        labels=raw_labels,
        max_samples=max_samples,
        subset_mode=subset_mode,
        sample_seed=sample_seed,
    )

    image_features = torch.as_tensor(raw_image_features[subset_indices], dtype=torch.float32, device=device)
    text_features = torch.as_tensor(raw_text_features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(raw_labels[subset_indices], dtype=torch.long, device=device)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    raw_clip_logits = image_features @ text_features.t()
    raw_clip_probs = torch.softmax(raw_clip_logits, dim=-1)
    raw_clip_entropy = entropy_from_probs(raw_clip_probs, dim=-1)

    return DatasetPayload(
        dataset=str(raw["dataset_key"]).lower(),
        image_features=image_features,
        text_features=text_features,
        labels=labels,
        raw_clip_logits=raw_clip_logits,
        raw_clip_probs=raw_clip_probs,
        raw_clip_entropy=raw_clip_entropy,
        original_num_samples=original_num_samples,
    )


def get_random_order(num_samples: int, device: torch.device, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_samples, generator=generator).to(device)


def get_round_robin_order(labels: torch.Tensor) -> torch.Tensor:
    labels_cpu = labels.detach().cpu().numpy()
    unique_labels = sorted(np.unique(labels_cpu).tolist())
    buckets = [np.where(labels_cpu == label)[0].tolist() for label in unique_labels]
    max_len = max(len(bucket) for bucket in buckets)
    order: list[int] = []
    for idx in range(max_len):
        for bucket in buckets:
            if idx < len(bucket):
                order.append(int(bucket[idx]))
    return torch.as_tensor(order, dtype=torch.long, device=labels.device)


def build_orders(payload: DatasetPayload) -> dict[str, torch.Tensor]:
    natural = torch.arange(payload.num_samples, device=payload.labels.device)
    random = get_random_order(payload.num_samples, payload.labels.device, seed=1)
    round_robin = get_round_robin_order(payload.labels)
    easy_to_hard = torch.argsort(payload.raw_clip_entropy, descending=False)
    hard_to_easy = torch.argsort(payload.raw_clip_entropy, descending=True)
    class_blocked = torch.argsort(payload.labels, descending=False, stable=True)
    return {
        "natural": natural,
        "random": random,
        "round_robin": round_robin,
        "easy_to_hard": easy_to_hard,
        "hard_to_easy": hard_to_easy,
        "class_blocked": class_blocked,
    }


def prefix_class_balance(order: torch.Tensor, labels: torch.Tensor, num_classes: int, prefix: int = 100) -> dict[str, float]:
    limit = min(prefix, int(order.numel()))
    prefix_indices = order[:limit].detach().cpu()
    prefix_labels = labels[prefix_indices].detach().cpu()
    counts = prefix_labels.bincount(minlength=num_classes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = float((-(probs[probs > 0] * torch.log(probs[probs > 0]))).sum().item())
    normalized_entropy = entropy / math.log(max(num_classes, 2))
    unique_classes = int((counts > 0).sum().item())
    return {
        "prefix_size": limit,
        "prefix_unique_classes": unique_classes,
        "prefix_unique_class_ratio": unique_classes / max(num_classes, 1),
        "prefix_class_entropy_ratio": normalized_entropy,
    }


def evaluate_tda_accuracy(
    payload: DatasetPayload,
    params: dict,
    order: torch.Tensor,
    progress_label: str | None = None,
    progress_interval: int = 1000,
) -> float:
    model = TDA(text_features=payload.text_features, device=payload.image_features.device, **params)
    correct = 0
    order_list = order.detach().cpu().tolist()
    total = len(order_list)
    with torch.inference_mode():
        for step, idx in enumerate(order_list, start=1):
            pred, _, _ = model.predict(payload.image_features[idx])
            correct += int(pred.item() == payload.labels[idx].item())
            if progress_label is not None and should_log_progress(step, total, progress_interval):
                log(f"{progress_label}: {step}/{total} samples")
    return correct / max(payload.num_samples, 1)


def evaluate_freetta_accuracy(
    payload: DatasetPayload,
    params: dict,
    order: torch.Tensor,
    progress_label: str | None = None,
    progress_interval: int = 1000,
) -> float:
    model = FreeTTA(text_features=payload.text_features, device=payload.image_features.device, **params)
    correct = 0
    order_list = order.detach().cpu().tolist()
    total = len(order_list)
    with torch.inference_mode():
        for step, idx in enumerate(order_list, start=1):
            pred, _ = model.predict(payload.image_features[idx], payload.raw_clip_logits[idx].unsqueeze(0))
            correct += int(pred.item() == payload.labels[idx].item())
            if progress_label is not None and should_log_progress(step, total, progress_interval):
                log(f"{progress_label}: {step}/{total} samples")
    return correct / max(payload.num_samples, 1)


def run_pair_detailed(
    payload: DatasetPayload,
    order: torch.Tensor,
    progress_label: str | None = None,
    progress_interval: int = 1000,
) -> tuple[dict, pd.DataFrame]:
    tda_params = BEST_TDA_PARAMS[payload.dataset]
    freetta_params = BEST_FREETTA_PARAMS[payload.dataset]

    tda = TDA(text_features=payload.text_features, device=payload.image_features.device, **tda_params)
    freetta = FreeTTA(text_features=payload.text_features, device=payload.image_features.device, **freetta_params)
    initial_mu = freetta.mu.detach().clone()

    max_entropy_tda = math.log(max(payload.num_classes, 2))
    rows: list[dict] = []
    clip_correct = 0
    tda_correct = 0
    freetta_correct = 0

    order_list = order.detach().cpu().tolist()
    total = len(order_list)
    with torch.inference_mode():
        for stream_step, idx in enumerate(order_list):
            x = payload.image_features[idx]
            y = int(payload.labels[idx].item())

            raw_clip_logits = payload.raw_clip_logits[idx]
            raw_clip_probs = payload.raw_clip_probs[idx]
            clip_pred = int(torch.argmax(raw_clip_logits).item())
            clip_is_correct = int(clip_pred == y)
            clip_correct += clip_is_correct

            clip_conf = float(torch.max(raw_clip_probs).item())
            clip_entropy = float(payload.raw_clip_entropy[idx].item())
            clip_margin = top2_margin(raw_clip_logits)

            scaled_clip_logits = tda.clip_scale * raw_clip_logits.unsqueeze(0)
            scaled_clip_probs = torch.softmax(scaled_clip_logits, dim=-1)
            tda_input_conf = float(torch.max(scaled_clip_probs, dim=-1).values.item())
            tda_input_entropy = float(entropy_from_probs(scaled_clip_probs, dim=-1).item())
            tda_norm_entropy_scaled = tda_input_entropy / max(max_entropy_tda, 1e-12)

            tda_pred_tensor, tda_final_conf_tensor, tda_final_logits = tda.predict(x)
            tda_pred = int(tda_pred_tensor.item())
            tda_is_correct = int(tda_pred == y)
            tda_correct += tda_is_correct
            tda_final_probs = torch.softmax(tda_final_logits, dim=-1)
            tda_final_conf = float(tda_final_conf_tensor.item())
            tda_final_entropy = float(entropy_from_probs(tda_final_probs, dim=-1).item())
            tda_true_class_prob = float(tda_final_probs[0, y].item())
            tda_final_margin = top2_margin(tda_final_logits)
            tda_changed = int(tda_pred != clip_pred)

            x_2d = x.unsqueeze(0)
            raw_fused_logits = scaled_clip_logits.clone()
            raw_fused_logits += tda._compute_cache_logits(  # noqa: SLF001
                x_2d,
                cache=tda.pos_cache,
                alpha=tda.alpha,
                beta=tda.beta,
                negative=False,
            )
            raw_fused_logits -= tda._compute_cache_logits(  # noqa: SLF001
                x_2d,
                cache=tda.neg_cache,
                alpha=tda.neg_alpha,
                beta=tda.neg_beta,
                negative=True,
            )
            raw_fused_probs = torch.softmax(raw_fused_logits, dim=-1)
            fused_conf_pre_fallback = float(torch.max(raw_fused_probs, dim=-1).values.item())
            fallback_used = int(
                tda.fallback_to_clip and (fused_conf_pre_fallback + tda.fallback_margin < tda_input_conf)
            )
            tda_beneficial_flip = int((clip_is_correct == 0) and (tda_is_correct == 1) and (tda_changed == 1))
            tda_harmful_flip = int((clip_is_correct == 1) and (tda_is_correct == 0) and (tda_changed == 1))

            freetta_input_entropy = clip_entropy
            freetta_input_conf = clip_conf
            freetta_weight = float(torch.exp(-freetta.beta * torch.tensor(clip_entropy, device=freetta.device)).item())
            mu_before = freetta.mu.detach().clone()
            freetta_pred_tensor, freetta_final_probs = freetta.predict(x, raw_clip_logits.unsqueeze(0))
            freetta_pred = int(freetta_pred_tensor.item())
            freetta_is_correct = int(freetta_pred == y)
            freetta_correct += freetta_is_correct
            freetta_final_conf = float(torch.max(freetta_final_probs, dim=-1).values.item())
            freetta_final_entropy = float(entropy_from_probs(freetta_final_probs, dim=-1).item())
            freetta_true_class_prob = float(freetta_final_probs[0, y].item())
            freetta_final_margin = top2_margin(torch.log(freetta_final_probs + 1e-12))
            freetta_changed = int(freetta_pred != clip_pred)
            freetta_beneficial_flip = int((clip_is_correct == 0) and (freetta_is_correct == 1) and (freetta_changed == 1))
            freetta_harmful_flip = int((clip_is_correct == 1) and (freetta_is_correct == 0) and (freetta_changed == 1))
            freetta_mu_update_norm = float(torch.norm(freetta.mu - mu_before, dim=1).mean().item())
            freetta_mu_drift = float(torch.norm(freetta.mu - initial_mu, dim=1).mean().item())
            priors = (freetta.Ny / (freetta.t + 1e-8)).clamp_min(1e-12)
            freetta_prior_entropy = float((-(priors * torch.log(priors))).sum().item())

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
                    "tda_changed_prediction": tda_changed,
                    "tda_beneficial_flip": tda_beneficial_flip,
                    "tda_harmful_flip": tda_harmful_flip,
                    "tda_input_confidence": tda_input_conf,
                    "tda_input_entropy": tda_input_entropy,
                    "tda_norm_entropy_scaled": tda_norm_entropy_scaled,
                    "tda_final_confidence": tda_final_conf,
                    "tda_final_entropy": tda_final_entropy,
                    "tda_true_class_prob": tda_true_class_prob,
                    "tda_final_margin": tda_final_margin,
                    "tda_entropy_drop": tda_input_entropy - tda_final_entropy,
                    "tda_positive_cache_size": int(tda.pos_size),
                    "tda_negative_cache_size": int(tda.neg_size),
                    "tda_negative_gate_open": int(tda.low_entropy < tda_norm_entropy_scaled < tda.high_entropy),
                    "tda_fallback_used": fallback_used,
                    "freetta_pred": freetta_pred,
                    "freetta_correct": freetta_is_correct,
                    "freetta_changed_prediction": freetta_changed,
                    "freetta_beneficial_flip": freetta_beneficial_flip,
                    "freetta_harmful_flip": freetta_harmful_flip,
                    "freetta_input_confidence": freetta_input_conf,
                    "freetta_input_entropy": freetta_input_entropy,
                    "freetta_final_confidence": freetta_final_conf,
                    "freetta_final_entropy": freetta_final_entropy,
                    "freetta_true_class_prob": freetta_true_class_prob,
                    "freetta_final_margin": freetta_final_margin,
                    "freetta_entropy_drop": freetta_input_entropy - freetta_final_entropy,
                    "freetta_em_weight": freetta_weight,
                    "freetta_mu_update_norm": freetta_mu_update_norm,
                    "freetta_mu_drift": freetta_mu_drift,
                    "freetta_prior_entropy": freetta_prior_entropy,
                }
            )

            step = stream_step + 1
            if progress_label is not None and should_log_progress(step, total, progress_interval):
                log(f"{progress_label}: {step}/{total} samples")

    summary = {
        "dataset": payload.dataset,
        "samples": payload.num_samples,
        "original_samples": payload.original_num_samples,
        "sample_fraction": payload.num_samples / max(payload.original_num_samples, 1),
        "clip_acc": clip_correct / max(payload.num_samples, 1),
        "tda_acc": tda_correct / max(payload.num_samples, 1),
        "freetta_acc": freetta_correct / max(payload.num_samples, 1),
        "freetta_minus_tda": (freetta_correct - tda_correct) / max(payload.num_samples, 1),
    }
    return summary, pd.DataFrame(rows)


def compute_ece(confidence: pd.Series, correctness: pd.Series, num_bins: int = 10) -> float:
    conf = confidence.to_numpy(dtype=float)
    corr = correctness.to_numpy(dtype=float)
    if conf.size == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = conf.size
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        bin_conf = conf[mask].mean()
        bin_acc = corr[mask].mean()
        ece += float(np.abs(bin_conf - bin_acc) * (mask.sum() / total))
    return ece


def summarize_mechanism_metrics(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in details.groupby("dataset", observed=True):
        for method in ("tda", "freetta"):
            correct = group[f"{method}_correct"]
            changed = group[f"{method}_changed_prediction"] == 1
            entropy_drop = group[f"{method}_entropy_drop"]
            beneficial = group[f"{method}_beneficial_flip"]
            harmful = group[f"{method}_harmful_flip"]

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "accuracy": float(correct.mean()),
                    "ece": compute_ece(group[f"{method}_final_confidence"], correct),
                    "avg_confidence": float(group[f"{method}_final_confidence"].mean()),
                    "avg_true_class_prob": float(group[f"{method}_true_class_prob"].mean()),
                    "avg_final_entropy": float(group[f"{method}_final_entropy"].mean()),
                    "entropy_drop_correct": safe_mean(entropy_drop[correct == 1]),
                    "entropy_drop_wrong": safe_mean(entropy_drop[correct == 0]),
                    "beneficial_entropy_drop": safe_mean(entropy_drop[beneficial == 1]),
                    "harmful_entropy_drop": safe_mean(entropy_drop[harmful == 1]),
                    "selective_entropy_gap": safe_mean(entropy_drop[correct == 1]) - safe_mean(entropy_drop[correct == 0]),
                    "change_rate": float(changed.mean()),
                    "beneficial_flips": int(beneficial.sum()),
                    "harmful_flips": int(harmful.sum()),
                    "beneficial_precision": float(beneficial.sum() / max(changed.sum(), 1)),
                    "harmful_rate_on_clip_correct": float(
                        harmful.sum() / max(int((group["clip_correct"] == 1).sum()), 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_break_even(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    running_rows: list[dict] = []

    for dataset, group in details.groupby("dataset", observed=True):
        group = group.sort_values("stream_step").reset_index(drop=True)
        total = len(group)
        window = max(50, total // 20)

        clip_correct = group["clip_correct"].astype(float)
        tda_correct = group["tda_correct"].astype(float)
        freetta_correct = group["freetta_correct"].astype(float)

        tda_vs_clip = tda_correct.rolling(window, min_periods=window).mean() - clip_correct.rolling(window, min_periods=window).mean()
        freetta_vs_clip = freetta_correct.rolling(window, min_periods=window).mean() - clip_correct.rolling(window, min_periods=window).mean()
        freetta_vs_tda = freetta_correct.rolling(window, min_periods=window).mean() - tda_correct.rolling(window, min_periods=window).mean()

        break_tda = next((int(i) + 1 for i, v in enumerate(tda_vs_clip) if pd.notna(v) and v > 0), None)
        break_freetta = next((int(i) + 1 for i, v in enumerate(freetta_vs_clip) if pd.notna(v) and v > 0), None)
        break_freetta_vs_tda = next((int(i) + 1 for i, v in enumerate(freetta_vs_tda) if pd.notna(v) and v > 0), None)

        summary_rows.append(
            {
                "dataset": dataset,
                "samples": total,
                "window": window,
                "tda_break_even_vs_clip": break_tda,
                "freetta_break_even_vs_clip": break_freetta,
                "freetta_break_even_vs_tda": break_freetta_vs_tda,
                "tda_break_even_ratio": (break_tda / total) if break_tda is not None else float("nan"),
                "freetta_break_even_ratio": (break_freetta / total) if break_freetta is not None else float("nan"),
                "freetta_vs_tda_break_even_ratio": (break_freetta_vs_tda / total)
                if break_freetta_vs_tda is not None
                else float("nan"),
            }
        )

        for idx in range(total):
            running_rows.append(
                {
                    "dataset": dataset,
                    "stream_step": idx + 1,
                    "progress_ratio": (idx + 1) / total,
                    "rolling_tda_vs_clip": float(tda_vs_clip.iloc[idx]) if pd.notna(tda_vs_clip.iloc[idx]) else float("nan"),
                    "rolling_freetta_vs_clip": float(freetta_vs_clip.iloc[idx])
                    if pd.notna(freetta_vs_clip.iloc[idx])
                    else float("nan"),
                    "rolling_freetta_vs_tda": float(freetta_vs_tda.iloc[idx])
                    if pd.notna(freetta_vs_tda.iloc[idx])
                    else float("nan"),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(running_rows)


def run_order_stress_test(
    features_dir: Path,
    datasets: list[str],
    device: torch.device,
    max_samples: int | None = None,
    subset_mode: str = "stratified",
    sample_seed: int = 1,
    progress_interval: int = 1000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    total_datasets = len(datasets)
    for dataset_idx, dataset in enumerate(datasets, start=1):
        log(f"[Order Stress {dataset_idx}/{total_datasets}] Loading {dataset}")
        payload = load_payload(
            features_dir,
            dataset,
            device,
            max_samples=max_samples,
            subset_mode=subset_mode,
            sample_seed=sample_seed,
        )
        try:
            clip_acc = float((torch.argmax(payload.raw_clip_logits, dim=-1) == payload.labels).float().mean().item())
            orders = build_orders(payload)
            total_schemes = len(orders)
            for scheme_idx, (scheme, order) in enumerate(orders.items(), start=1):
                label_prefix = f"[Order Stress {dataset} {scheme_idx}/{total_schemes}]"
                log(f"{label_prefix} Evaluating TDA and FreeTTA")
                balance = prefix_class_balance(order, payload.labels, payload.num_classes)
                tda_acc = evaluate_tda_accuracy(
                    payload,
                    BEST_TDA_PARAMS[dataset],
                    order,
                    progress_label=f"{label_prefix} TDA",
                    progress_interval=progress_interval,
                )
                freetta_acc = evaluate_freetta_accuracy(
                    payload,
                    BEST_FREETTA_PARAMS[dataset],
                    order,
                    progress_label=f"{label_prefix} FreeTTA",
                    progress_interval=progress_interval,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "scheme": scheme,
                        "clip_acc": clip_acc,
                        "tda_acc": tda_acc,
                        "freetta_acc": freetta_acc,
                        "freetta_minus_tda": freetta_acc - tda_acc,
                        **balance,
                    }
                )
                if output_path is not None:
                    pd.DataFrame(rows).to_csv(output_path, index=False)
        finally:
            del payload
            release_gpu_memory(device)
    return pd.DataFrame(rows)


def run_tda_cache_ablation(
    features_dir: Path,
    datasets: list[str],
    device: torch.device,
    max_samples: int | None = None,
    subset_mode: str = "stratified",
    sample_seed: int = 1,
    progress_interval: int = 1000,
    variant_output_path: Path | None = None,
    shot_output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_rows: list[dict] = []
    shot_rows: list[dict] = []

    total_datasets = len(datasets)
    for dataset_idx, dataset in enumerate(datasets, start=1):
        log(f"[TDA Ablation {dataset_idx}/{total_datasets}] Loading {dataset}")
        payload = load_payload(
            features_dir,
            dataset,
            device,
            max_samples=max_samples,
            subset_mode=subset_mode,
            sample_seed=sample_seed,
        )
        try:
            natural = build_orders(payload)["natural"]
            base_params = dict(BEST_TDA_PARAMS[dataset])

            for variant in ("both", "positive_only", "negative_only"):
                params = dict(base_params)
                if variant == "positive_only":
                    params["neg_alpha"] = 0.0
                elif variant == "negative_only":
                    params["alpha"] = 0.0
                log(f"[TDA Ablation {dataset}] Variant {variant}")
                acc = evaluate_tda_accuracy(
                    payload,
                    params,
                    natural,
                    progress_label=f"[TDA Ablation {dataset} {variant}]",
                    progress_interval=progress_interval,
                )
                variant_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "tda_acc": acc,
                        "clip_acc": float((torch.argmax(payload.raw_clip_logits, dim=-1) == payload.labels).float().mean().item()),
                    }
                )
                if variant_output_path is not None:
                    pd.DataFrame(variant_rows).to_csv(variant_output_path, index=False)

            for shot_capacity in TDA_SHOT_GRID:
                params = dict(base_params)
                params["shot_capacity"] = int(shot_capacity)
                log(f"[TDA Ablation {dataset}] Shot capacity {shot_capacity}")
                acc = evaluate_tda_accuracy(
                    payload,
                    params,
                    natural,
                    progress_label=f"[TDA Ablation {dataset} shots={shot_capacity}]",
                    progress_interval=progress_interval,
                )
                shot_rows.append(
                    {
                        "dataset": dataset,
                        "shot_capacity": int(shot_capacity),
                        "effective_slots_per_cache": int(shot_capacity) * payload.num_classes,
                        "tda_acc": acc,
                    }
                )
                if shot_output_path is not None:
                    pd.DataFrame(shot_rows).to_csv(shot_output_path, index=False)
        finally:
            del payload
            release_gpu_memory(device)

    return pd.DataFrame(variant_rows), pd.DataFrame(shot_rows)


def run_freetta_param_sweep(
    features_dir: Path,
    datasets: list[str],
    device: torch.device,
    max_samples: int | None = None,
    subset_mode: str = "stratified",
    sample_seed: int = 1,
    progress_interval: int = 1000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    total_datasets = len(datasets)
    total_combos = len(FREETTA_ALPHA_GRID) * len(FREETTA_BETA_GRID)
    for dataset_idx, dataset in enumerate(datasets, start=1):
        log(f"[FreeTTA Sweep {dataset_idx}/{total_datasets}] Loading {dataset}")
        payload = load_payload(
            features_dir,
            dataset,
            device,
            max_samples=max_samples,
            subset_mode=subset_mode,
            sample_seed=sample_seed,
        )
        try:
            natural = build_orders(payload)["natural"]
            log(f"[FreeTTA Sweep {dataset}] Computing TDA reference")
            tda_reference = evaluate_tda_accuracy(
                payload,
                BEST_TDA_PARAMS[dataset],
                natural,
                progress_label=f"[FreeTTA Sweep {dataset} TDA reference]",
                progress_interval=progress_interval,
            )
            combo_idx = 0
            for alpha in FREETTA_ALPHA_GRID:
                for beta in FREETTA_BETA_GRID:
                    combo_idx += 1
                    params = {"alpha": float(alpha), "beta": float(beta)}
                    combo_label = f"[FreeTTA Sweep {dataset} {combo_idx}/{total_combos}] alpha={alpha}, beta={beta}"
                    log(combo_label)
                    acc = evaluate_freetta_accuracy(
                        payload,
                        params,
                        natural,
                        progress_label=combo_label,
                        progress_interval=progress_interval,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "freetta_acc": acc,
                            "freetta_minus_tda": acc - tda_reference,
                        }
                    )
                    if output_path is not None:
                        pd.DataFrame(rows).to_csv(output_path, index=False)
        finally:
            del payload
            release_gpu_memory(device)
    return pd.DataFrame(rows)


def plot_calibration(mechanism_df: pd.DataFrame, output_path: Path) -> None:
    datasets = mechanism_df["dataset"].unique().tolist()
    x = np.arange(len(datasets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))

    tda = mechanism_df[mechanism_df["method"] == "tda"].set_index("dataset").loc[datasets]
    freetta = mechanism_df[mechanism_df["method"] == "freetta"].set_index("dataset").loc[datasets]

    ax.bar(x - width / 2, tda["ece"] * 100.0, width, label="TDA")
    ax.bar(x + width / 2, freetta["ece"] * 100.0, width, label="FreeTTA")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("ECE (%)")
    ax.set_title("Confidence Calibration Error")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_entropy_quality(mechanism_df: pd.DataFrame, output_path: Path) -> None:
    datasets = mechanism_df["dataset"].unique().tolist()
    x = np.arange(len(datasets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.2))

    tda = mechanism_df[mechanism_df["method"] == "tda"].set_index("dataset").loc[datasets]
    freetta = mechanism_df[mechanism_df["method"] == "freetta"].set_index("dataset").loc[datasets]

    ax.bar(x - 1.5 * width, tda["entropy_drop_correct"], width, label="TDA: correct")
    ax.bar(x - 0.5 * width, tda["entropy_drop_wrong"], width, label="TDA: wrong")
    ax.bar(x + 0.5 * width, freetta["entropy_drop_correct"], width, label="FreeTTA: correct")
    ax.bar(x + 1.5 * width, freetta["entropy_drop_wrong"], width, label="FreeTTA: wrong")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean entropy drop")
    ax.set_title("Selective Entropy Reduction")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_flip_precision(mechanism_df: pd.DataFrame, output_path: Path) -> None:
    datasets = mechanism_df["dataset"].unique().tolist()
    x = np.arange(len(datasets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))

    tda = mechanism_df[mechanism_df["method"] == "tda"].set_index("dataset").loc[datasets]
    freetta = mechanism_df[mechanism_df["method"] == "freetta"].set_index("dataset").loc[datasets]

    ax.bar(x - width / 2, tda["beneficial_precision"] * 100.0, width, label="TDA")
    ax.bar(x + width / 2, freetta["beneficial_precision"] * 100.0, width, label="FreeTTA")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Beneficial flip precision (%)")
    ax.set_title("How Often Prediction Changes Help")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_running_gains(running_df: pd.DataFrame, break_even_df: pd.DataFrame, output_path: Path) -> None:
    datasets = running_df["dataset"].unique().tolist()
    fig, axes = make_axes_grid(len(datasets), ncols=2)

    for ax, dataset in zip(axes, datasets):
        group = running_df[running_df["dataset"] == dataset]
        ax.plot(group["progress_ratio"], group["rolling_freetta_vs_tda"] * 100.0, color="#1f77b4")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        be_row = break_even_df[break_even_df["dataset"] == dataset].iloc[0]
        if pd.notna(be_row["freetta_vs_tda_break_even_ratio"]):
            ax.axvline(float(be_row["freetta_vs_tda_break_even_ratio"]), color="#d62728", linestyle="--", alpha=0.8)
        ax.set_title(dataset)
        ax.grid(alpha=0.25)

    for ax in axes[: len(datasets)]:
        ax.set_xlabel("Stream progress")
        ax.set_ylabel("Rolling FreeTTA - TDA accuracy (pp)")
    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

    fig.suptitle("When FreeTTA Starts Beating TDA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_order_heatmaps(order_df: pd.DataFrame, output_path: Path) -> None:
    datasets = sorted(order_df["dataset"].unique().tolist())
    schemes = ["natural", "class_blocked", "round_robin", "random", "easy_to_hard", "hard_to_easy"]

    tda_mat = (
        order_df.pivot(index="dataset", columns="scheme", values="tda_acc")
        .reindex(index=datasets, columns=schemes)
        .to_numpy()
        * 100.0
    )
    freetta_mat = (
        order_df.pivot(index="dataset", columns="scheme", values="freetta_acc")
        .reindex(index=datasets, columns=schemes)
        .to_numpy()
        * 100.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, mat, title in zip(axes, (tda_mat, freetta_mat), ("TDA", "FreeTTA")):
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(np.arange(len(schemes)))
        ax.set_xticklabels(schemes, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_title(f"{title} accuracy by stream order")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_tda_ablation(variant_df: pd.DataFrame, shot_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = sorted(variant_df["dataset"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    variant_order = ["both", "positive_only", "negative_only"]
    x = np.arange(len(datasets))
    width = 0.24
    for offset, variant in zip((-width, 0.0, width), variant_order):
        values = variant_df[variant_df["variant"] == variant].set_index("dataset").loc[datasets]["tda_acc"] * 100.0
        axes[0].bar(x + offset, values, width=width, label=variant)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("TDA dual-cache ablation")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    summary_lookup = summary_df.set_index("dataset")
    for dataset in datasets:
        group = shot_df[shot_df["dataset"] == dataset].sort_values("shot_capacity")
        axes[1].plot(group["shot_capacity"], group["tda_acc"] * 100.0, marker="o", label=dataset)
        axes[1].axhline(summary_lookup.loc[dataset, "freetta_acc"] * 100.0, linestyle="--", linewidth=0.8, alpha=0.35)
    axes[1].set_xlabel("Shot capacity per class")
    axes[1].set_ylabel("TDA accuracy (%)")
    axes[1].set_title("TDA effective cache-size sweep")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_freetta_heatmaps(param_df: pd.DataFrame, output_path: Path) -> None:
    datasets = sorted(param_df["dataset"].unique().tolist())
    fig, axes = make_axes_grid(len(datasets), ncols=2)

    alpha_labels = list(FREETTA_ALPHA_GRID)
    beta_labels = list(FREETTA_BETA_GRID)
    global_min = float(param_df["freetta_minus_tda"].min() * 100.0)
    global_max = float(param_df["freetta_minus_tda"].max() * 100.0)

    for ax, dataset in zip(axes, datasets):
        pivot = (
            param_df[param_df["dataset"] == dataset]
            .pivot(index="alpha", columns="beta", values="freetta_minus_tda")
            .reindex(index=alpha_labels, columns=beta_labels)
        )
        mat = pivot.to_numpy() * 100.0
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=global_min, vmax=global_max)
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(beta_labels)))
        ax.set_xticklabels(beta_labels)
        ax.set_yticks(np.arange(len(alpha_labels)))
        ax.set_yticklabels(alpha_labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)

    for ax in axes[: len(datasets)]:
        ax.set_xlabel("beta")
        ax.set_ylabel("alpha")
    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

    fig.colorbar(im, ax=axes[: len(datasets)].tolist(), fraction=0.03, pad=0.02, label="FreeTTA - TDA (pp)")
    fig.suptitle("FreeTTA sensitivity to alpha and beta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report_notes(
    summary_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
    break_even_df: pd.DataFrame,
    order_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    freetta_sweep_df: pd.DataFrame,
    output_path: Path,
) -> None:
    summary = summary_df.set_index("dataset")
    mech = mechanism_df.set_index(["dataset", "method"])
    order_worst = order_df.sort_values(["dataset", "freetta_minus_tda"]).groupby("dataset", as_index=False).first()
    cache_best = variant_df.sort_values(["dataset", "tda_acc"], ascending=[True, False]).groupby("dataset", as_index=False).first()
    sweep_best = freetta_sweep_df.sort_values(["dataset", "freetta_minus_tda"], ascending=[True, False]).groupby("dataset", as_index=False).first()

    lines = [
        "# Superiority Analysis Notes",
        "",
        "## Headline Findings",
    ]
    for dataset in summary.index:
        lines.append(
            f"- `{dataset}`: FreeTTA - TDA = {summary.loc[dataset, 'freetta_minus_tda'] * 100.0:.2f} pp, "
            f"ECE(TDA/FreeTTA) = {mech.loc[(dataset, 'tda'), 'ece'] * 100.0:.2f}/{mech.loc[(dataset, 'freetta'), 'ece'] * 100.0:.2f}, "
            f"break-even ratio vs TDA = {break_even_df.set_index('dataset').loc[dataset, 'freetta_vs_tda_break_even_ratio']:.3f}."
        )

    lines.extend(["", "## Order Robustness"])
    for _, row in order_worst.iterrows():
        lines.append(
            f"- `{row['dataset']}` worst FreeTTA-vs-TDA scheme: `{row['scheme']}` "
            f"({row['freetta_minus_tda'] * 100.0:.2f} pp) with prefix class entropy ratio {row['prefix_class_entropy_ratio']:.3f}."
        )

    lines.extend(["", "## TDA Cache Observations"])
    for _, row in cache_best.iterrows():
        lines.append(f"- `{row['dataset']}` best TDA cache variant under natural order: `{row['variant']}` ({row['tda_acc'] * 100.0:.2f}%).")

    lines.extend(["", "## FreeTTA Parameter Sweet Spots"])
    for _, row in sweep_best.iterrows():
        lines.append(
            f"- `{row['dataset']}` best tested alpha/beta = ({row['alpha']}, {row['beta']}) "
            f"with gain {row['freetta_minus_tda'] * 100.0:.2f} pp over TDA."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workshop-style TDA vs FreeTTA superiority analysis")
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/workshop_superiority"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--subset-mode", choices=["stratified", "random", "head"], default="stratified")
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--skip-order-stress", action="store_true")
    parser.add_argument("--skip-tda-ablation", action="store_true")
    parser.add_argument("--skip-freetta-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    available = set(list_available_datasets(args.features_dir))
    requested = [str(dataset).lower() for dataset in args.datasets]
    datasets = [dataset for dataset in requested if dataset in available]
    skipped = [dataset for dataset in requested if dataset not in available]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(
        "Starting superiority analysis "
        f"on {len(datasets)} dataset(s) with device={device.type}, "
        f"max_samples_per_dataset={args.max_samples_per_dataset}, subset_mode={args.subset_mode}, "
        f"progress_interval={args.progress_interval}"
    )
    if skipped:
        log(f"Skipping datasets without features: {skipped}")
    if not datasets:
        raise RuntimeError(f"No requested datasets were available in {args.features_dir}")

    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    total_datasets = len(datasets)
    for dataset_idx, dataset in enumerate(datasets, start=1):
        log(f"[Natural Pass {dataset_idx}/{total_datasets}] Loading {dataset}")
        payload = load_payload(
            args.features_dir,
            dataset,
            device,
            max_samples=args.max_samples_per_dataset,
            subset_mode=args.subset_mode,
            sample_seed=args.sample_seed,
        )
        try:
            if payload.num_samples != payload.original_num_samples:
                log(
                    f"[Natural Pass {dataset_idx}/{total_datasets}] Using subset for {dataset}: "
                    f"{payload.num_samples}/{payload.original_num_samples} samples"
                )
            natural = build_orders(payload)["natural"]
            summary, detail_df = run_pair_detailed(
                payload,
                natural,
                progress_label=f"[Natural Pass {dataset}]",
                progress_interval=args.progress_interval,
            )
            summary_rows.append(summary)
            detail_frames.append(detail_df)
            pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True).to_csv(
                args.output_dir / "natural_order_summary.csv", index=False
            )
            pd.concat(detail_frames, ignore_index=True).to_csv(args.output_dir / "natural_order_per_sample.csv", index=False)
            log(f"[Natural Pass {dataset_idx}/{total_datasets}] Saved intermediate natural-order outputs for {dataset}")
        finally:
            del payload
            release_gpu_memory(device)

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    details_df = pd.concat(detail_frames, ignore_index=True)
    log("Summarizing mechanism and break-even metrics")
    mechanism_df = summarize_mechanism_metrics(details_df)
    break_even_df, running_df = summarize_break_even(details_df)
    summary_df.to_csv(args.output_dir / "natural_order_summary.csv", index=False)
    details_df.to_csv(args.output_dir / "natural_order_per_sample.csv", index=False)
    mechanism_df.to_csv(args.output_dir / "mechanism_metrics.csv", index=False)
    break_even_df.to_csv(args.output_dir / "break_even_metrics.csv", index=False)
    running_df.to_csv(args.output_dir / "running_gain_metrics.csv", index=False)

    if not args.skip_order_stress:
        log("Starting order stress test")
        order_df = run_order_stress_test(
            features_dir=args.features_dir,
            datasets=list(datasets),
            device=device,
            max_samples=args.max_samples_per_dataset,
            subset_mode=args.subset_mode,
            sample_seed=args.sample_seed,
            progress_interval=args.progress_interval,
            output_path=args.output_dir / "order_stress_test.csv",
        )
    else:
        order_df = pd.DataFrame()
    if not args.skip_tda_ablation:
        log("Starting TDA cache ablation")
        variant_df, shot_df = run_tda_cache_ablation(
            features_dir=args.features_dir,
            datasets=list(datasets),
            device=device,
            max_samples=args.max_samples_per_dataset,
            subset_mode=args.subset_mode,
            sample_seed=args.sample_seed,
            progress_interval=args.progress_interval,
            variant_output_path=args.output_dir / "tda_cache_variants.csv",
            shot_output_path=args.output_dir / "tda_shot_capacity_sweep.csv",
        )
    else:
        variant_df, shot_df = pd.DataFrame(), pd.DataFrame()
    if not args.skip_freetta_sweep:
        log("Starting FreeTTA parameter sweep")
        freetta_sweep_df = run_freetta_param_sweep(
            features_dir=args.features_dir,
            datasets=list(datasets),
            device=device,
            max_samples=args.max_samples_per_dataset,
            subset_mode=args.subset_mode,
            sample_seed=args.sample_seed,
            progress_interval=args.progress_interval,
            output_path=args.output_dir / "freetta_param_sweep.csv",
        )
    else:
        freetta_sweep_df = pd.DataFrame()

    if not order_df.empty:
        order_df.to_csv(args.output_dir / "order_stress_test.csv", index=False)
    if not variant_df.empty:
        variant_df.to_csv(args.output_dir / "tda_cache_variants.csv", index=False)
    if not shot_df.empty:
        shot_df.to_csv(args.output_dir / "tda_shot_capacity_sweep.csv", index=False)
    if not freetta_sweep_df.empty:
        freetta_sweep_df.to_csv(args.output_dir / "freetta_param_sweep.csv", index=False)

    log("Rendering plots")
    plot_calibration(mechanism_df, args.output_dir / "calibration_ece.png")
    plot_entropy_quality(mechanism_df, args.output_dir / "entropy_quality.png")
    plot_flip_precision(mechanism_df, args.output_dir / "flip_precision.png")
    plot_running_gains(running_df, break_even_df, args.output_dir / "running_gains.png")
    if not order_df.empty:
        plot_order_heatmaps(order_df, args.output_dir / "order_heatmaps.png")
    if not variant_df.empty and not shot_df.empty:
        plot_tda_ablation(variant_df, shot_df, summary_df, args.output_dir / "tda_cache_ablation.png")
    if not freetta_sweep_df.empty:
        plot_freetta_heatmaps(freetta_sweep_df, args.output_dir / "freetta_param_heatmaps.png")

    if not order_df.empty and not variant_df.empty and not freetta_sweep_df.empty:
        log("Writing report notes")
        write_report_notes(
            summary_df=summary_df,
            mechanism_df=mechanism_df,
            break_even_df=break_even_df,
            order_df=order_df,
            variant_df=variant_df,
            freetta_sweep_df=freetta_sweep_df,
            output_path=args.output_dir / "report_notes.md",
        )

    log(f"Superiority analysis complete. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
