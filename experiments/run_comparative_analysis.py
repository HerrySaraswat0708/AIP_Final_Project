from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    PCA = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_tda import load_tda_dataset
from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.imagenet_loader import ensure_imagenetv2
from src.paper_configs import DEFAULT_FREETTA_PARAMS, PAPER_TDA_DEFAULTS
from src.paper_setup import EXPECTED_TEST_SPLIT_SIZES
from src.pet_loader import resolve_pet_image_root


DEFAULT_DATASETS = ("caltech", "dtd", "eurosat", "pets", "imagenet")
FAILURE_BUCKETS = {
    "clip_wrong_tda_wrong_freetta_correct": lambda df: (
        (df["clip_correct"] == 0) & (df["tda_correct"] == 0) & (df["freetta_correct"] == 1)
    ),
    "clip_wrong_tda_correct_freetta_wrong": lambda df: (
        (df["clip_correct"] == 0) & (df["tda_correct"] == 1) & (df["freetta_correct"] == 0)
    ),
    "clip_correct_tda_wrong_freetta_correct": lambda df: (
        (df["clip_correct"] == 1) & (df["tda_correct"] == 0) & (df["freetta_correct"] == 1)
    ),
    "clip_correct_tda_correct_freetta_wrong": lambda df: (
        (df["clip_correct"] == 1) & (df["tda_correct"] == 1) & (df["freetta_correct"] == 0)
    ),
    "all_wrong": lambda df: (
        (df["clip_correct"] == 0) & (df["tda_correct"] == 0) & (df["freetta_correct"] == 0)
    ),
}


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


def load_best_tda_params(config_path: str | None) -> dict[str, dict]:
    defaults = {key: dict(value) for key, value in PAPER_TDA_DEFAULTS.items()}
    if not config_path:
        return defaults
    path = Path(config_path)
    if not path.exists():
        return defaults
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("results", []):
        dataset = str(item.get("dataset", "")).lower()
        params = item.get("params")
        if dataset in defaults and isinstance(params, dict):
            defaults[dataset] = dict(params)
    return defaults


def load_best_freetta_params(config_path: str | None) -> dict[str, dict]:
    defaults = {key: dict(value) for key, value in DEFAULT_FREETTA_PARAMS.items()}
    if not config_path:
        return defaults
    path = Path(config_path)
    if not path.exists():
        return defaults
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset, params in payload.items():
        dataset_key = str(dataset).lower()
        if dataset_key in defaults and isinstance(params, dict):
            alpha = params.get("alpha")
            beta = params.get("beta")
            if alpha is not None and beta is not None:
                defaults[dataset_key] = {"alpha": float(alpha), "beta": float(beta)}
    return defaults


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


def entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
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


def run_tda_stream(payload: DatasetPayload, order: torch.Tensor, params: dict) -> tuple[dict, pd.DataFrame, np.ndarray]:
    model = TDA(text_features=payload.text_features, device=payload.image_features.device, **params)
    rows: list[dict] = []
    tda_logits_rows: list[np.ndarray] = []
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

            pred, _, final_logits = model.predict(x)
            final_probs = torch.softmax(final_logits, dim=-1)
            tda_pred = int(pred.item())
            tda_is_correct = int(tda_pred == y)
            tda_correct += tda_is_correct
            tda_conf = float(torch.max(final_probs, dim=-1).values.item())
            tda_entropy = float(entropy_from_logits(final_logits, dim=-1).item())
            tda_margin = top2_margin(final_logits)
            changed = int(tda_pred != clip_pred)
            beneficial_flip = int(changed and clip_is_correct == 0 and tda_is_correct == 1)
            harmful_flip = int(changed and clip_is_correct == 1 and tda_is_correct == 0)
            other_changed_wrong = int(changed and clip_is_correct == 0 and tda_is_correct == 0)
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
                    "tda_other_changed_wrong": other_changed_wrong,
                    "tda_positive_cache_size": int(model.pos_size),
                    "tda_negative_cache_size": int(model.neg_size),
                    "tda_negative_gate_open": int(model.low_entropy < norm_entropy < model.high_entropy),
                }
            )
            tda_logits_rows.append(final_logits.squeeze(0).detach().cpu().numpy().astype(np.float32))

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
    return summary, pd.DataFrame(rows), np.stack(tda_logits_rows, axis=0)


def run_freetta_stream(
    payload: DatasetPayload,
    order: torch.Tensor,
    params: dict,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    model = FreeTTA(text_features=payload.text_features, device=payload.image_features.device, **params)
    initial_mu = model.mu.detach().clone()
    rows: list[dict] = []
    freetta_logits_rows: list[np.ndarray] = []
    mu_drift_by_class_rows: list[np.ndarray] = []
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
            em_weight = math.exp(-float(model.beta) * clip_entropy)
            mu_before = model.mu.detach().clone()

            pred, final_probs = model.predict(x, clip_logits)
            final_logits = torch.log(final_probs + 1e-12)
            freetta_pred = int(pred.squeeze(0).item())
            freetta_is_correct = int(freetta_pred == y)
            freetta_correct += freetta_is_correct
            freetta_conf = float(torch.max(final_probs, dim=-1).values.item())
            freetta_entropy = float(entropy_from_probs(final_probs, dim=-1).item())
            freetta_margin = top2_margin(final_logits)
            changed = int(freetta_pred != clip_pred)
            beneficial_flip = int(changed and clip_is_correct == 0 and freetta_is_correct == 1)
            harmful_flip = int(changed and clip_is_correct == 1 and freetta_is_correct == 0)
            other_changed_wrong = int(changed and clip_is_correct == 0 and freetta_is_correct == 0)
            mu_update_norm = float(torch.norm(model.mu - mu_before, dim=1).mean().item())
            mu_drift_by_class = torch.norm(model.mu - initial_mu, dim=1)
            mu_drift = float(mu_drift_by_class.mean().item())
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
                    "freetta_other_changed_wrong": other_changed_wrong,
                    "freetta_em_weight": em_weight,
                    "freetta_mu_update_norm": mu_update_norm,
                    "freetta_mu_drift": mu_drift,
                    "freetta_prior_entropy": prior_entropy,
                    "freetta_sigma_trace": sigma_trace,
                }
            )
            freetta_logits_rows.append(final_logits.squeeze(0).detach().cpu().numpy().astype(np.float32))
            mu_drift_by_class_rows.append(mu_drift_by_class.detach().cpu().numpy().astype(np.float32))

    summary = {
        "dataset": payload.dataset,
        "samples": payload.num_samples,
        "clip_acc": clip_correct / max(total, 1),
        "freetta_acc": freetta_correct / max(total, 1),
        "freetta_gain_vs_clip": (freetta_correct - clip_correct) / max(total, 1),
        "freetta_final_mu_drift": float(rows[-1]["freetta_mu_drift"]) if rows else float("nan"),
        "freetta_final_prior_entropy": float(rows[-1]["freetta_prior_entropy"]) if rows else float("nan"),
        "freetta_final_sigma_trace": float(rows[-1]["freetta_sigma_trace"]) if rows else float("nan"),
    }
    return (
        summary,
        pd.DataFrame(rows),
        np.stack(freetta_logits_rows, axis=0),
        np.stack(mu_drift_by_class_rows, axis=0),
    )


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def resolve_dataset_sample_paths(dataset: str, expected_count: int) -> list[str]:
    dataset = str(dataset).lower()

    if dataset == "caltech":
        split_path = PROJECT_ROOT / "data" / "splits" / "split_zhou_Caltech101.json"
        root_candidates = [
            PROJECT_ROOT / "data" / "raw" / "CALTECH_clean" / "caltech101" / "101_ObjectCategories",
            PROJECT_ROOT / "data" / "raw" / "CALTECH_fresh" / "caltech101" / "101_ObjectCategories",
            PROJECT_ROOT / "data" / "raw" / "CALTECH" / "caltech101" / "101_ObjectCategories",
        ]
        root_dir = next((candidate for candidate in root_candidates if candidate.exists()), None)
        if not split_path.exists() or root_dir is None:
            return []
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        paths = []
        for rel_path, *_ in payload.get("test", []):
            image_path = root_dir / str(rel_path)
            if image_path.exists():
                paths.append(str(image_path))
        return paths if len(paths) == expected_count else []

    if dataset == "dtd":
        split_candidates = [
            PROJECT_ROOT / "data" / "splits" / "split_zhou_DescribableTextures.json",
            PROJECT_ROOT / "data" / "splits" / "split_zhou_DescribableTextures_tda.json",
        ]
        split_path = next((candidate for candidate in split_candidates if candidate.exists()), None)
        root_candidates = [
            PROJECT_ROOT / "data" / "raw" / "DTD_fresh" / "dtd",
            PROJECT_ROOT / "data" / "raw" / "DTD" / "dtd",
            PROJECT_ROOT / "data" / "raw" / "DTD" / "dtd" / "dtd",
        ]
        root_dir = next((candidate for candidate in root_candidates if (candidate / "images").exists()), None)
        if split_path is None or root_dir is None:
            return []
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        image_dir = root_dir / "images"
        paths = []
        for rel_path, *_ in payload.get("test", []):
            image_path = image_dir / str(rel_path)
            if image_path.exists():
                paths.append(str(image_path))
        return paths if len(paths) == expected_count else []

    if dataset == "eurosat":
        split_path = PROJECT_ROOT / "data" / "splits" / "split_zhou_EuroSAT.json"
        root_candidates = [
            PROJECT_ROOT / "data" / "raw" / "EUROSAT" / "eurosat" / "2750",
            PROJECT_ROOT / "data" / "raw" / "EUROSAT_fresh" / "eurosat" / "2750",
        ]
        root_dir = next((candidate for candidate in root_candidates if candidate.exists()), None)
        if not split_path.exists() or root_dir is None:
            return []
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        paths = []
        for rel_path, *_ in payload.get("test", []):
            image_path = root_dir / str(rel_path)
            if image_path.exists():
                paths.append(str(image_path))
        return paths if len(paths) == expected_count else []

    if dataset == "pets":
        split_path = PROJECT_ROOT / "data" / "splits" / "split_zhou_OxfordPets.json"
        if not split_path.exists():
            return []
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        test_entries = payload.get("test", [])
        root_dir = resolve_pet_image_root(split_entries=test_entries)
        if root_dir is None:
            return []
        paths = []
        for rel_name, *_ in test_entries:
            image_path = root_dir / str(rel_name)
            try:
                exists = image_path.exists()
            except OSError:
                exists = False
            if exists:
                paths.append(str(image_path))
        return paths if len(paths) == expected_count else []

    if dataset == "imagenet":
        image_dir = ensure_imagenetv2()
        paths = []
        class_dirs = sorted(
            (path for path in image_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        for class_dir in class_dirs:
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file():
                    paths.append(str(image_path))
        return paths if len(paths) == expected_count else []

    return []


def compute_prediction_change_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        for method in ("tda", "freetta"):
            changed = group[f"{method}_changed_prediction"] == 1
            beneficial = group[f"{method}_beneficial_flip"] == 1
            harmful = group[f"{method}_harmful_flip"] == 1
            other_changed_wrong = group[f"{method}_other_changed_wrong"] == 1
            unchanged_correct = (group["clip_correct"] == 1) & (~changed)
            unchanged_wrong = (group["clip_correct"] == 0) & (~changed)
            total = max(len(group), 1)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "samples": int(len(group)),
                    "unchanged_correct_count": int(unchanged_correct.sum()),
                    "unchanged_wrong_count": int(unchanged_wrong.sum()),
                    "beneficial_flip_count": int(beneficial.sum()),
                    "harmful_flip_count": int(harmful.sum()),
                    "other_changed_wrong_count": int(other_changed_wrong.sum()),
                    "unchanged_correct_rate": float(unchanged_correct.mean()),
                    "unchanged_wrong_rate": float(unchanged_wrong.mean()),
                    "change_rate": float(changed.mean()),
                    "beneficial_flip_precision": float(beneficial.sum() / max(int(changed.sum()), 1)),
                    "harmful_flip_rate_on_clip_correct": float(
                        harmful.sum() / max(int((group["clip_correct"] == 1).sum()), 1)
                    ),
                    "net_correction_score": int(beneficial.sum() - harmful.sum()),
                    "net_correction_rate": float((beneficial.sum() - harmful.sum()) / total),
                    "avg_entropy_after_beneficial_flip": safe_mean(group.loc[beneficial, f"{method}_entropy"]),
                    "avg_entropy_after_harmful_flip": safe_mean(group.loc[harmful, f"{method}_entropy"]),
                }
            )
    return pd.DataFrame(rows)


def compute_entropy_confidence_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        for method in ("clip", "tda", "freetta"):
            for subset_name, mask in (
                ("all", np.ones(len(group), dtype=bool)),
                ("correct", (group[f"{method}_correct"] == 1).to_numpy()),
                ("wrong", (group[f"{method}_correct"] == 0).to_numpy()),
            ):
                subset = group.loc[mask]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "subset": subset_name,
                        "samples": int(len(subset)),
                        "mean_entropy": safe_mean(subset[f"{method}_entropy"]),
                        "median_entropy": float(subset[f"{method}_entropy"].median()) if len(subset) else float("nan"),
                        "std_entropy": float(subset[f"{method}_entropy"].std(ddof=0)) if len(subset) else float("nan"),
                        "mean_confidence": safe_mean(subset[f"{method}_confidence"]),
                        "median_confidence": float(subset[f"{method}_confidence"].median()) if len(subset) else float("nan"),
                        "std_confidence": float(subset[f"{method}_confidence"].std(ddof=0)) if len(subset) else float("nan"),
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


def build_trajectory_metrics(group: pd.DataFrame, window: int) -> tuple[pd.DataFrame, dict]:
    group = group.sort_values("stream_step").reset_index(drop=True)
    total = len(group)
    use_window = min(max(window, 5), max(total, 5))

    def rolling(series: pd.Series) -> pd.Series:
        return series.astype(float).rolling(use_window, min_periods=1).mean()

    curve_df = pd.DataFrame(
        {
            "dataset": group["dataset"],
            "sample_index": group["sample_index"],
            "stream_step": group["stream_step"],
            "progress_ratio": (group["stream_step"] + 1) / max(total, 1),
            "rolling_clip_acc": rolling(group["clip_correct"]),
            "rolling_tda_acc": rolling(group["tda_correct"]),
            "rolling_freetta_acc": rolling(group["freetta_correct"]),
            "rolling_clip_confidence": rolling(group["clip_confidence"]),
            "rolling_tda_confidence": rolling(group["tda_confidence"]),
            "rolling_freetta_confidence": rolling(group["freetta_confidence"]),
            "rolling_clip_entropy": rolling(group["clip_entropy"]),
            "rolling_tda_entropy": rolling(group["tda_entropy"]),
            "rolling_freetta_entropy": rolling(group["freetta_entropy"]),
        }
    )
    curve_df["rolling_tda_vs_clip"] = curve_df["rolling_tda_acc"] - curve_df["rolling_clip_acc"]
    curve_df["rolling_freetta_vs_clip"] = curve_df["rolling_freetta_acc"] - curve_df["rolling_clip_acc"]
    curve_df["rolling_freetta_vs_tda"] = curve_df["rolling_freetta_acc"] - curve_df["rolling_tda_acc"]

    def first_positive(series: pd.Series) -> int | None:
        positive = np.flatnonzero(series.to_numpy() > 0)
        return int(positive[0] + 1) if len(positive) else None

    tda_be = first_positive(curve_df["rolling_tda_vs_clip"])
    freetta_be = first_positive(curve_df["rolling_freetta_vs_clip"])
    freetta_tda_be = first_positive(curve_df["rolling_freetta_vs_tda"])
    latency_row = {
        "dataset": str(group["dataset"].iloc[0]),
        "window": int(use_window),
        "tda_break_even_vs_clip": tda_be,
        "freetta_break_even_vs_clip": freetta_be,
        "freetta_break_even_vs_tda": freetta_tda_be,
        "tda_break_even_ratio": (tda_be / total) if tda_be is not None else float("nan"),
        "freetta_break_even_ratio": (freetta_be / total) if freetta_be is not None else float("nan"),
        "freetta_vs_tda_break_even_ratio": (freetta_tda_be / total) if freetta_tda_be is not None else float("nan"),
    }
    return curve_df, latency_row


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


def compute_failure_bucket_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in merged.groupby("dataset", observed=True):
        for bucket_name, matcher in FAILURE_BUCKETS.items():
            mask = matcher(group)
            rows.append(
                {
                    "dataset": dataset,
                    "bucket": bucket_name,
                    "count": int(mask.sum()),
                    "rate": float(mask.mean()),
                }
            )
    return pd.DataFrame(rows)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_prediction_change_analysis(dataset_dir: Path, prediction_df: pd.DataFrame) -> None:
    methods = prediction_df["method"].tolist()
    stacked_labels = [
        ("unchanged_correct_rate", "Unchanged Correct", "#66c2a5"),
        ("unchanged_wrong_rate", "Unchanged Wrong", "#bdbdbd"),
        ("beneficial_flip_count", "Beneficial Flip", "#4daf4a"),
        ("harmful_flip_count", "Harmful Flip", "#e41a1c"),
        ("other_changed_wrong_count", "Changed Wrong->Wrong", "#ffb347"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bottoms = np.zeros(len(methods))
    total_samples = prediction_df["samples"].iloc[0]
    for column, label, color in stacked_labels:
        if column.endswith("_rate"):
            values = prediction_df[column].to_numpy(dtype=float)
        else:
            values = prediction_df[column].to_numpy(dtype=float) / max(float(total_samples), 1.0)
        axes[0].bar(methods, values, bottom=bottoms, label=label, color=color)
        bottoms += values
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fraction of Samples")
    axes[0].set_title("Prediction Change Breakdown")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    x = np.arange(len(methods))
    width = 0.35
    axes[1].bar(x - width / 2, prediction_df["change_rate"], width=width, label="Change Rate", color="#6baed6")
    axes[1].bar(x + width / 2, prediction_df["net_correction_rate"], width=width, label="Net Correction Rate", color="#31a354")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Change Rate and Net Correction")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(dataset_dir / "prediction_change_analysis.png", dpi=180)
    plt.close(fig)


def plot_entropy_confidence_analysis(dataset_dir: Path, group: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ("clip", "tda", "freetta")
    correct_data = [group.loc[group[f"{method}_correct"] == 1, f"{method}_entropy"].to_numpy() for method in methods]
    wrong_data = [group.loc[group[f"{method}_correct"] == 0, f"{method}_entropy"].to_numpy() for method in methods]

    axes[0].boxplot(correct_data, tick_labels=[method.upper() for method in methods], showfliers=False)
    axes[0].set_title("Entropy on Correct Predictions")
    axes[0].set_ylabel("Entropy")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(wrong_data, tick_labels=[method.upper() for method in methods], showfliers=False)
    axes[1].set_title("Entropy on Wrong Predictions")
    axes[1].grid(axis="y", alpha=0.25)

    bins = np.linspace(0.0, 1.0, 25)
    for method, color in zip(methods, ("#4c72b0", "#dd8452", "#55a868"), strict=False):
        axes[2].hist(
            group[f"{method}_confidence"].to_numpy(),
            bins=bins,
            alpha=0.45,
            density=True,
            label=method.upper(),
            color=color,
        )
    axes[2].set_title("Confidence Distribution")
    axes[2].set_xlabel("Max Softmax Probability")
    axes[2].set_ylabel("Density")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(dataset_dir / "entropy_confidence_analysis.png", dpi=180)
    plt.close(fig)


def plot_trajectory_analysis(dataset_dir: Path, trajectory_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x = trajectory_df["stream_step"].to_numpy() + 1

    axes[0].plot(x, trajectory_df["rolling_clip_acc"], label="CLIP", linewidth=2)
    axes[0].plot(x, trajectory_df["rolling_tda_acc"], label="TDA", linewidth=2)
    axes[0].plot(x, trajectory_df["rolling_freetta_acc"], label="FreeTTA", linewidth=2)
    axes[0].set_ylabel("Rolling Accuracy")
    axes[0].set_title("Trajectory Analysis")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, trajectory_df["rolling_clip_confidence"], label="CLIP", linewidth=2)
    axes[1].plot(x, trajectory_df["rolling_tda_confidence"], label="TDA", linewidth=2)
    axes[1].plot(x, trajectory_df["rolling_freetta_confidence"], label="FreeTTA", linewidth=2)
    axes[1].set_ylabel("Rolling Confidence")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, trajectory_df["rolling_clip_entropy"], label="CLIP", linewidth=2)
    axes[2].plot(x, trajectory_df["rolling_tda_entropy"], label="TDA", linewidth=2)
    axes[2].plot(x, trajectory_df["rolling_freetta_entropy"], label="FreeTTA", linewidth=2)
    axes[2].set_ylabel("Rolling Entropy")
    axes[2].set_xlabel("Stream Step")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(dataset_dir / "trajectory_analysis.png", dpi=180)
    plt.close(fig)


def plot_freetta_internal_analysis(dataset_dir: Path, group: pd.DataFrame, mu_drift_by_class: np.ndarray) -> None:
    x = group["stream_step"].to_numpy() + 1
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    mean_drift = mu_drift_by_class.mean(axis=1)
    final_drift = mu_drift_by_class[-1]
    top_k = min(5, mu_drift_by_class.shape[1])
    top_indices = np.argsort(final_drift)[-top_k:]
    axes[0].plot(x, mean_drift, label="Mean Drift", linewidth=2, color="#111111")
    for class_idx in top_indices:
        axes[0].plot(x, mu_drift_by_class[:, class_idx], linewidth=1.2, alpha=0.8, label=f"class {class_idx}")
    axes[0].set_ylabel("||mu_y(t) - mu_y(0)||")
    axes[0].set_title("FreeTTA Internal Statistics")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncols=2)

    axes[1].plot(x, group["freetta_prior_entropy"], linewidth=2, color="#4c72b0")
    axes[1].set_ylabel("Prior Entropy")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, group["freetta_sigma_trace"], linewidth=2, color="#55a868")
    axes[2].set_ylabel("Covariance Trace")
    axes[2].set_xlabel("Stream Step")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(dataset_dir / "freetta_internal_analysis.png", dpi=180)
    plt.close(fig)


def plot_tda_internal_analysis(dataset_dir: Path, group: pd.DataFrame, window: int) -> None:
    x = group["stream_step"].to_numpy() + 1
    gate_window = min(max(window, 5), len(group))
    gate_rate = group["tda_negative_gate_open"].astype(float).rolling(gate_window, min_periods=1).mean()
    cumulative_gate = group["tda_negative_gate_open"].astype(float).expanding().mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, group["tda_positive_cache_size"], label="Positive Cache", linewidth=2)
    axes[0].plot(x, group["tda_negative_cache_size"], label="Negative Cache", linewidth=2)
    axes[0].set_ylabel("Cache Size")
    axes[0].set_title("TDA Internal Statistics")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, gate_rate, label="Rolling Gate Activation", linewidth=2)
    axes[1].plot(x, cumulative_gate, label="Cumulative Gate Activation", linewidth=2, linestyle="--")
    axes[1].set_ylabel("Gate Rate")
    axes[1].set_xlabel("Stream Step")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(dataset_dir / "tda_internal_analysis.png", dpi=180)
    plt.close(fig)


def compute_pca_projection(
    clip_logits: np.ndarray,
    tda_logits: np.ndarray,
    freetta_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_logits = np.concatenate([clip_logits, tda_logits, freetta_logits], axis=0).astype(np.float32, copy=False)
    if PCA is not None:
        projector = PCA(n_components=2, svd_solver="randomized", random_state=0)
        projection = projector.fit_transform(all_logits)
    else:  # pragma: no cover - fallback path
        centered = all_logits - all_logits.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        projection = (u[:, :2] * s[:2]).astype(np.float32, copy=False)
    n = clip_logits.shape[0]
    return projection[:n], projection[n : 2 * n], projection[2 * n :]


def plot_pca_logit_visualization(
    dataset_dir: Path,
    group: pd.DataFrame,
    clip_2d: np.ndarray,
    tda_2d: np.ndarray,
    freetta_2d: np.ndarray,
    max_arrows: int = 400,
) -> pd.DataFrame:
    same_prediction_all = (
        (group["clip_pred"] == group["tda_pred"]) & (group["clip_pred"] == group["freetta_pred"])
    ).to_numpy()
    tda_wins = ((group["tda_correct"] == 1) & (group["freetta_correct"] == 0)).to_numpy()
    freetta_wins = ((group["freetta_correct"] == 1) & (group["tda_correct"] == 0)).to_numpy()
    both_wrong = ((group["tda_correct"] == 0) & (group["freetta_correct"] == 0)).to_numpy()
    different_from_clip = (
        (group["tda_pred"] != group["clip_pred"]) | (group["freetta_pred"] != group["clip_pred"])
    ).to_numpy()

    category = np.full(len(group), "different_from_clip", dtype=object)
    category[same_prediction_all] = "same_prediction_all"
    category[both_wrong] = "both_wrong"
    category[tda_wins] = "tda_correct_freetta_wrong"
    category[freetta_wins] = "freetta_correct_tda_wrong"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    markers = {"clip": "o", "tda": "s", "freetta": "^"}
    colors = {True: "#2ca25f", False: "#de2d26"}

    for method, coords in (("clip", clip_2d), ("tda", tda_2d), ("freetta", freetta_2d)):
        correct = group[f"{method}_correct"].astype(bool).to_numpy()
        for is_correct in (True, False):
            mask = correct == is_correct
            axes[0, 0].scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                alpha=0.18,
                marker=markers[method],
                color=colors[is_correct],
                label=f"{method.upper()} {'correct' if is_correct else 'wrong'}",
            )
    axes[0, 0].set_title("PCA of CLIP / TDA / FreeTTA Logits")
    axes[0, 0].grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0, 0].legend(dedup.values(), dedup.keys(), fontsize=8, ncols=2)

    axes[0, 1].scatter(clip_2d[:, 0], clip_2d[:, 1], s=8, alpha=0.08, color="#7f7f7f", label="CLIP")
    arrow_scores = np.maximum(group["clip_entropy"].to_numpy(), group["freetta_entropy"].to_numpy())
    candidate_mask = different_from_clip
    candidate_idx = np.flatnonzero(candidate_mask)
    if len(candidate_idx):
        ordered = candidate_idx[np.argsort(arrow_scores[candidate_idx])[::-1]]
        chosen = ordered[:max_arrows]
        for idx in chosen:
            if bool(group["tda_changed_prediction"].iloc[idx]):
                axes[0, 1].annotate(
                    "",
                    xy=(tda_2d[idx, 0], tda_2d[idx, 1]),
                    xytext=(clip_2d[idx, 0], clip_2d[idx, 1]),
                    arrowprops={"arrowstyle": "->", "color": "#1f77b4", "alpha": 0.18, "lw": 0.7},
                )
            if bool(group["freetta_changed_prediction"].iloc[idx]):
                axes[0, 1].annotate(
                    "",
                    xy=(freetta_2d[idx, 0], freetta_2d[idx, 1]),
                    xytext=(clip_2d[idx, 0], clip_2d[idx, 1]),
                    arrowprops={"arrowstyle": "->", "color": "#ff7f0e", "alpha": 0.18, "lw": 0.7},
                )
    axes[0, 1].set_title("Logit Movement Arrows: CLIP -> TDA / FreeTTA")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].scatter(tda_2d[tda_wins, 0], tda_2d[tda_wins, 1], s=14, alpha=0.7, color="#1f77b4", label="TDA correct, FreeTTA wrong")
    axes[1, 0].scatter(
        freetta_2d[freetta_wins, 0],
        freetta_2d[freetta_wins, 1],
        s=14,
        alpha=0.7,
        color="#ff7f0e",
        label="FreeTTA correct, TDA wrong",
    )
    axes[1, 0].scatter(clip_2d[both_wrong, 0], clip_2d[both_wrong, 1], s=10, alpha=0.2, color="#444444", label="Both wrong")
    axes[1, 0].set_title("Special Cases")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    category_colors = {
        "same_prediction_all": "#4daf4a",
        "different_from_clip": "#984ea3",
        "tda_correct_freetta_wrong": "#1f77b4",
        "freetta_correct_tda_wrong": "#ff7f0e",
        "both_wrong": "#333333",
    }
    for name, color in category_colors.items():
        mask = category == name
        axes[1, 1].scatter(clip_2d[mask, 0], clip_2d[mask, 1], s=12, alpha=0.35, color=color, label=name)
    axes[1, 1].set_title("Sample Categories in CLIP Logit Space")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(dataset_dir / "pca_logit_visualization.png", dpi=180)
    plt.close(fig)

    projection_df = pd.DataFrame(
        {
            "dataset": group["dataset"].to_numpy(),
            "sample_index": group["sample_index"].to_numpy(),
            "stream_step": group["stream_step"].to_numpy(),
            "category": category,
            "clip_pc1": clip_2d[:, 0],
            "clip_pc2": clip_2d[:, 1],
            "tda_pc1": tda_2d[:, 0],
            "tda_pc2": tda_2d[:, 1],
            "freetta_pc1": freetta_2d[:, 0],
            "freetta_pc2": freetta_2d[:, 1],
        }
    )
    projection_df.to_csv(dataset_dir / "pca_projection.csv", index=False)
    return projection_df


def _load_font(size: int = 14):
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover
        return None


def save_resized_example(image_path: Path, output_path: Path, caption: str) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        thumb = ImageOps.fit(image, (240, 240), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (240, 280), color=(255, 255, 255))
    canvas.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font()
    draw.text((8, 246), caption, fill=(20, 20, 20), font=font)
    canvas.save(output_path)


def create_contact_sheet(rows: pd.DataFrame, output_path: Path) -> None:
    if rows.empty:
        return
    font = _load_font()
    tile_w, tile_h = 240, 280
    cols = 2 if len(rows) <= 4 else 3
    rows_count = int(math.ceil(len(rows) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows_count * tile_h), color=(255, 255, 255))

    for idx, row in enumerate(rows.itertuples(index=False)):
        image_path = Path(row.sample_path)
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            thumb = ImageOps.fit(image, (240, 240), method=Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), color=(255, 255, 255))
        tile.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(tile)
        caption = f"idx={row.sample_index} gt={row.label} c/t/f={row.clip_pred}/{row.tda_pred}/{row.freetta_pred}"
        draw.text((8, 246), caption, fill=(20, 20, 20), font=font)
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(tile, (x, y))

    sheet.save(output_path)


def export_failure_case_buckets(dataset_dir: Path, group: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    failure_root = dataset_dir / "failure_cases"
    ensure_output_dir(failure_root)
    summary_rows: list[dict] = []

    for bucket_name, matcher in FAILURE_BUCKETS.items():
        mask = matcher(group)
        bucket_df = group.loc[mask].copy()
        bucket_df = bucket_df.sort_values(["clip_entropy", "stream_step"], ascending=[False, True]).reset_index(drop=True)
        bucket_dir = failure_root / bucket_name
        ensure_output_dir(bucket_dir)

        summary_rows.append(
            {
                "bucket": bucket_name,
                "count": int(len(bucket_df)),
                "rate": float(len(bucket_df) / max(len(group), 1)),
            }
        )
        bucket_df.to_csv(bucket_dir / "all_examples.csv", index=False)

        if "sample_path" in bucket_df.columns:
            bucket_df = bucket_df[bucket_df["sample_path"].notna() & (bucket_df["sample_path"] != "")]
        selected = bucket_df.head(max_examples).copy()
        selected.to_csv(bucket_dir / "selected_examples.csv", index=False)

        if not selected.empty and "sample_path" in selected.columns:
            for example_idx, row in enumerate(selected.itertuples(index=False), start=1):
                image_path = Path(row.sample_path)
                if not image_path.exists():
                    continue
                stem = image_path.stem.replace(" ", "_")
                out_path = bucket_dir / f"{example_idx:02d}_{stem}.png"
                caption = f"gt={row.label} c/t/f={row.clip_pred}/{row.tda_pred}/{row.freetta_pred}"
                save_resized_example(image_path, out_path, caption)
            create_contact_sheet(selected, bucket_dir / "contact_sheet.png")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dataset_dir / "failure_buckets.csv", index=False)
    return summary_df


def write_dataset_report(
    dataset_dir: Path,
    summary_row: dict,
    geometry_row: dict,
    prediction_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    disagreement_row: pd.Series,
    latency_row: dict,
    internal_row: pd.Series,
) -> None:
    lines = [
        f"# Dataset Report: {summary_row['dataset']}",
        "",
        "## Accuracy Table",
        pd.DataFrame(
            [
                {
                    "clip_acc": summary_row["clip_acc"],
                    "tda_acc": summary_row["tda_acc"],
                    "freetta_acc": summary_row["freetta_acc"],
                    "tda_gain_vs_clip": summary_row["tda_gain_vs_clip"],
                    "freetta_gain_vs_clip": summary_row["freetta_gain_vs_clip"],
                    "freetta_minus_tda": summary_row["freetta_minus_tda"],
                }
            ]
        ).to_markdown(index=False),
        "",
        "## Geometry Probe",
        pd.DataFrame([geometry_row]).to_markdown(index=False),
        "",
        "## Prediction Change Metrics",
        prediction_df.to_markdown(index=False),
        "",
        "## Entropy / Confidence Metrics",
        entropy_df.to_markdown(index=False),
        "",
        "## Disagreement Metrics",
        pd.DataFrame([disagreement_row.to_dict()]).to_markdown(index=False),
        "",
        "## Latency Metrics",
        pd.DataFrame([latency_row]).to_markdown(index=False),
        "",
        "## Internal Metrics",
        pd.DataFrame([internal_row.to_dict()]).to_markdown(index=False),
        "",
        "## Generated PNG Outputs",
        "- `prediction_change_analysis.png`",
        "- `entropy_confidence_analysis.png`",
        "- `trajectory_analysis.png`",
        "- `freetta_internal_analysis.png`",
        "- `tda_internal_analysis.png`",
        "- `pca_logit_visualization.png`",
    ]
    (dataset_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(
    output_path: Path,
    summary_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    failure_bucket_df: pd.DataFrame,
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
    lines.append("## Prediction Change Analysis")
    lines.append(prediction_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Entropy / Confidence Summary")
    lines.append(entropy_df.to_markdown(index=False))
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
    lines.append("")
    lines.append("## Failure Bucket Summary")
    lines.append(failure_bucket_df.to_markdown(index=False))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive CLIP vs TDA vs FreeTTA analysis pipeline")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/comparative_analysis")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stream-seed", type=int, default=None, help="Use None for natural order, or set a seed for shuffled order")
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--failure-examples", type=int, default=8)
    parser.add_argument("--max-pca-arrows", type=int, default=400)
    parser.add_argument("--tda-config-json", default="outputs/tuning/best_tda_run_results.json")
    parser.add_argument("--freetta-config-json", default="outputs/tuning/best_freetta_run_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    device = resolve_device(args.device)
    tda_param_map = load_best_tda_params(args.tda_config_json)
    freetta_param_map = load_best_freetta_params(args.freetta_config_json)

    summary_rows: list[dict] = []
    geometry_rows: list[dict] = []
    latency_rows: list[dict] = []
    merged_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []

    for dataset in [str(x).lower() for x in args.datasets]:
        print(f"[Dataset] {dataset}", flush=True)
        payload = load_payload(dataset, device=device, features_dir=args.features_dir)
        order = get_order(payload.num_samples, device=device, seed=args.stream_seed)
        order_cpu = order.detach().cpu().numpy()
        dataset_dir = output_dir / payload.dataset
        ensure_output_dir(dataset_dir)
        tda_params = copy.deepcopy(tda_param_map[payload.dataset])
        freetta_params = copy.deepcopy(freetta_param_map[payload.dataset])

        tda_summary, tda_rows, tda_logits = run_tda_stream(payload, order, tda_params)
        freetta_summary, freetta_rows, freetta_logits, mu_drift_by_class = run_freetta_stream(
            payload, order, freetta_params
        )
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
                    "freetta_other_changed_wrong",
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
        merged = merged.sort_values("stream_step").reset_index(drop=True)
        merged["samples"] = payload.num_samples
        merged["tda_total_cache_slots"] = tda_summary["tda_total_cache_slots"]
        for key, value in geometry_row.items():
            if key != "dataset":
                merged[key] = value

        sample_paths = resolve_dataset_sample_paths(payload.dataset, payload.num_samples)
        if sample_paths:
            merged["sample_path"] = [sample_paths[int(idx)] for idx in order_cpu]
        else:
            merged["sample_path"] = ""
            print(
                f"[Warning] Could not reconstruct raw image paths for dataset '{payload.dataset}'. "
                "Failure-case image export will be limited to CSV metadata.",
                flush=True,
            )

        trajectory_df, latency_row = build_trajectory_metrics(merged, window=args.rolling_window)
        prediction_df = compute_prediction_change_metrics(merged)
        entropy_df = compute_entropy_confidence_metrics(merged)
        disagreement_df = compute_disagreement_metrics(merged)
        internal_df = compute_internal_metrics(merged)
        failure_bucket_df = export_failure_case_buckets(dataset_dir, merged, max_examples=args.failure_examples)

        stream_clip_logits = payload.raw_clip_logits[order].detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            dataset_dir / "logits.npz",
            sample_index=merged["sample_index"].to_numpy(dtype=np.int64),
            stream_step=merged["stream_step"].to_numpy(dtype=np.int64),
            labels=merged["label"].to_numpy(dtype=np.int64),
            clip_logits=stream_clip_logits,
            tda_logits=tda_logits.astype(np.float32, copy=False),
            freetta_logits=freetta_logits.astype(np.float32, copy=False),
        )
        np.savez_compressed(
            dataset_dir / "freetta_internal.npz",
            sample_index=merged["sample_index"].to_numpy(dtype=np.int64),
            stream_step=merged["stream_step"].to_numpy(dtype=np.int64),
            mu_drift_by_class=mu_drift_by_class.astype(np.float32, copy=False),
            prior_entropy=merged["freetta_prior_entropy"].to_numpy(dtype=np.float32),
            sigma_trace=merged["freetta_sigma_trace"].to_numpy(dtype=np.float32),
        )

        clip_2d, tda_2d, freetta_2d = compute_pca_projection(stream_clip_logits, tda_logits, freetta_logits)
        plot_prediction_change_analysis(dataset_dir, prediction_df)
        plot_entropy_confidence_analysis(dataset_dir, merged)
        plot_trajectory_analysis(dataset_dir, trajectory_df)
        plot_freetta_internal_analysis(dataset_dir, merged, mu_drift_by_class)
        plot_tda_internal_analysis(dataset_dir, merged, window=args.rolling_window)
        plot_pca_logit_visualization(
            dataset_dir,
            merged,
            clip_2d,
            tda_2d,
            freetta_2d,
            max_arrows=args.max_pca_arrows,
        )

        merged.to_csv(dataset_dir / "per_sample_metrics.csv", index=False)
        trajectory_df.to_csv(dataset_dir / "trajectory_metrics.csv", index=False)
        prediction_df.to_csv(dataset_dir / "prediction_change_metrics.csv", index=False)
        entropy_df.to_csv(dataset_dir / "entropy_confidence_metrics.csv", index=False)
        pd.DataFrame([latency_row]).to_csv(dataset_dir / "latency_metrics.csv", index=False)
        disagreement_df.to_csv(dataset_dir / "disagreement_metrics.csv", index=False)
        internal_df.to_csv(dataset_dir / "internal_metrics.csv", index=False)
        pd.DataFrame([geometry_row]).to_csv(dataset_dir / "geometry_metrics.csv", index=False)
        accuracy_table = pd.DataFrame(
            [
                {
                    "dataset": payload.dataset,
                    "clip_acc": float(tda_summary["clip_acc"]),
                    "tda_acc": float(tda_summary["tda_acc"]),
                    "freetta_acc": float(freetta_summary["freetta_acc"]),
                }
            ]
        )
        accuracy_table.to_csv(dataset_dir / "accuracy_table.csv", index=False)
        (dataset_dir / "used_params.json").write_text(
            json.dumps(
                {
                    "dataset": payload.dataset,
                    "shared_stream_seed": args.stream_seed,
                    "tda_params": tda_params,
                    "freetta_params": freetta_params,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        summary_row = {
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
        summary_rows.append(summary_row)
        latency_rows.append(latency_row)
        merged_frames.append(merged)
        trajectory_frames.append(trajectory_df)

        write_dataset_report(
            dataset_dir,
            summary_row=summary_row,
            geometry_row=geometry_row,
            prediction_df=prediction_df,
            entropy_df=entropy_df,
            disagreement_row=disagreement_df.iloc[0],
            latency_row=latency_row,
            internal_row=internal_df.iloc[0],
        )
        print(
            f"[Done] {payload.dataset}: clip={summary_row['clip_acc']:.4f} "
            f"tda={summary_row['tda_acc']:.4f} freetta={summary_row['freetta_acc']:.4f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    geometry_df = pd.DataFrame(geometry_rows).sort_values("dataset").reset_index(drop=True)
    merged_df = pd.concat(merged_frames, ignore_index=True)
    trajectory_df = pd.concat(trajectory_frames, ignore_index=True)
    prediction_df = compute_prediction_change_metrics(merged_df)
    disagreement_df = compute_disagreement_metrics(merged_df)
    difficulty_df = compute_difficulty_metrics(merged_df)
    internal_df = compute_internal_metrics(merged_df)
    entropy_df = compute_entropy_confidence_metrics(merged_df)
    latency_df = pd.DataFrame(latency_rows).sort_values("dataset").reset_index(drop=True)
    latency_curve_df = trajectory_df[
        [
            "dataset",
            "stream_step",
            "progress_ratio",
            "rolling_tda_vs_clip",
            "rolling_freetta_vs_clip",
            "rolling_freetta_vs_tda",
        ]
    ].copy()
    failure_bucket_df = compute_failure_bucket_metrics(merged_df)

    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    geometry_df.to_csv(output_dir / "geometry_metrics.csv", index=False)
    merged_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    prediction_df.to_csv(output_dir / "flip_metrics.csv", index=False)
    disagreement_df.to_csv(output_dir / "disagreement_metrics.csv", index=False)
    latency_df.to_csv(output_dir / "latency_metrics.csv", index=False)
    latency_curve_df.to_csv(output_dir / "latency_curves.csv", index=False)
    trajectory_df.to_csv(output_dir / "trajectory_metrics.csv", index=False)
    difficulty_df.to_csv(output_dir / "difficulty_metrics.csv", index=False)
    internal_df.to_csv(output_dir / "internal_metrics.csv", index=False)
    entropy_df.to_csv(output_dir / "entropy_confidence_metrics.csv", index=False)
    failure_bucket_df.to_csv(output_dir / "failure_bucket_summary.csv", index=False)

    report = {
        "device": str(device),
        "datasets": [str(x).lower() for x in args.datasets],
        "stream_seed": args.stream_seed,
        "rolling_window": int(args.rolling_window),
        "failure_examples": int(args.failure_examples),
        "tda_config_json": str(args.tda_config_json),
        "freetta_config_json": str(args.freetta_config_json),
        "summary_metrics_csv": str(output_dir / "summary_metrics.csv"),
        "geometry_metrics_csv": str(output_dir / "geometry_metrics.csv"),
        "per_sample_metrics_csv": str(output_dir / "per_sample_metrics.csv"),
        "flip_metrics_csv": str(output_dir / "flip_metrics.csv"),
        "entropy_confidence_metrics_csv": str(output_dir / "entropy_confidence_metrics.csv"),
        "disagreement_metrics_csv": str(output_dir / "disagreement_metrics.csv"),
        "latency_metrics_csv": str(output_dir / "latency_metrics.csv"),
        "trajectory_metrics_csv": str(output_dir / "trajectory_metrics.csv"),
        "difficulty_metrics_csv": str(output_dir / "difficulty_metrics.csv"),
        "internal_metrics_csv": str(output_dir / "internal_metrics.csv"),
        "failure_bucket_summary_csv": str(output_dir / "failure_bucket_summary.csv"),
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_summary(
        output_dir / "analysis_summary.md",
        summary_df,
        geometry_df,
        prediction_df,
        entropy_df,
        latency_df,
        difficulty_df,
        disagreement_df,
        internal_df,
        failure_bucket_df,
    )

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
