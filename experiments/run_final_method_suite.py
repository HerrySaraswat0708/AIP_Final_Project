from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_edgefreetta_comparison import DEFAULT_EDGE_PARAMS
from models.EdgeFreeTTA import EdgeFreeTTA
from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.feature_store import load_dataset_features
from src.paper_configs import DEFAULT_DATASETS, DEFAULT_FREETTA_PARAMS, PAPER_TDA_DEFAULTS
from src.paper_setup import EXPECTED_TEST_SPLIT_SIZES


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_tda_params(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {key: {"params": value, "shuffle_stream": True, "stream_seed": 1} for key, value in PAPER_TDA_DEFAULTS.items()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    params = {
        key: {"params": value, "shuffle_stream": True, "stream_seed": 1}
        for key, value in PAPER_TDA_DEFAULTS.items()
    }
    for row in results:
        dataset = str(row["dataset"]).lower()
        params[dataset] = {
            "params": dict(row["params"]),
            "shuffle_stream": bool(row.get("shuffle_stream", True)),
            "stream_seed": int(row.get("stream_seed", 1)),
        }
    return params


def load_best_freetta_params(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {
            key: {
                "alpha": float(value["alpha"]),
                "beta": float(value["beta"]),
                "shuffle_stream": True,
                "stream_seed": 1,
            }
            for key, value in DEFAULT_FREETTA_PARAMS.items()
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = {
        key: {
            "alpha": float(value["alpha"]),
            "beta": float(value["beta"]),
            "shuffle_stream": True,
            "stream_seed": 1,
        }
        for key, value in DEFAULT_FREETTA_PARAMS.items()
    }
    for dataset, row in payload.items():
        params[str(dataset).lower()] = {
            "alpha": float(row["alpha"]),
            "beta": float(row["beta"]),
            "shuffle_stream": bool(row.get("shuffle_stream", True)),
            "stream_seed": int(row.get("stream_seed", 1)),
        }
    return params


def load_dataset(dataset: str, device: torch.device, features_dir: Path) -> dict:
    raw = load_dataset_features(features_dir, dataset)
    image = raw["image_features"]
    text = raw["text_features"]
    labels = raw["labels"]
    dataset_key = str(raw["dataset_key"]).lower()

    sample_count = int(labels.shape[0])
    expected = EXPECTED_TEST_SPLIT_SIZES.get(dataset_key)
    if expected is not None and sample_count != expected:
        raise ValueError(f"Dataset '{dataset_key}' has {sample_count} samples, expected {expected}.")

    image_features = F.normalize(torch.from_numpy(image).float().to(device), dim=-1)
    text_features = F.normalize(torch.from_numpy(text).float().to(device), dim=-1)
    label_tensor = torch.from_numpy(labels).long().to(device)
    raw_clip_logits = image_features @ text_features.t()
    return {
        "dataset": dataset_key,
        "image_features": image_features,
        "text_features": text_features,
        "labels": label_tensor,
        "num_samples": sample_count,
        "raw_clip_logits": raw_clip_logits,
    }


def get_order(num_samples: int, device: torch.device, seed: int | None) -> torch.Tensor:
    if seed is None:
        return torch.arange(num_samples, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_samples, generator=generator).to(device)


def entropy_from_probs(probs: torch.Tensor) -> float:
    return float((-(probs * torch.log(probs + 1e-12)).sum()).item())


def top2_margin(logits: torch.Tensor) -> float:
    values = torch.topk(logits.reshape(-1), k=min(2, logits.numel())).values
    if values.numel() < 2:
        return float("nan")
    return float((values[0] - values[1]).item())


def run_dataset(
    payload: dict,
    clip_order: torch.Tensor,
    tda_cfg: dict,
    freetta_cfg: dict,
    edge_params: dict,
) -> tuple[dict, pd.DataFrame]:
    device = payload["image_features"].device
    tda_params = dict(tda_cfg["params"])
    freetta_params = {
        "alpha": float(freetta_cfg["alpha"]),
        "beta": float(freetta_cfg["beta"]),
    }
    tda = TDA(text_features=payload["text_features"], device=device, **tda_params)
    freetta = FreeTTA(text_features=payload["text_features"], device=device, **freetta_params)
    edge = EdgeFreeTTA(text_features=payload["text_features"], device=device, **edge_params)
    tda_order = get_order(payload["num_samples"], device=device, seed=tda_cfg["stream_seed"]) if tda_cfg["shuffle_stream"] else torch.arange(payload["num_samples"], device=device)
    freetta_order = get_order(payload["num_samples"], device=device, seed=freetta_cfg["stream_seed"]) if freetta_cfg["shuffle_stream"] else torch.arange(payload["num_samples"], device=device)
    tda_sequence = [int(x) for x in tda_order.tolist()]
    freetta_sequence = [int(x) for x in freetta_order.tolist()]
    tda_rank = {idx: pos for pos, idx in enumerate(tda_sequence)}
    freetta_rank = {idx: pos for pos, idx in enumerate(freetta_sequence)}
    tda_ptr = 0
    freetta_ptr = 0

    rows: list[dict] = []
    clip_correct = 0
    tda_correct = 0
    freetta_correct = 0
    edge_correct = 0

    with torch.inference_mode(False):
        for stream_step, idx_t in enumerate(clip_order):
            idx = int(idx_t.item())
            x = payload["image_features"][idx]
            y = int(payload["labels"][idx].item())
            clip_logits = payload["raw_clip_logits"][idx]
            clip_probs = torch.softmax(clip_logits, dim=-1)
            clip_pred = int(torch.argmax(clip_probs).item())
            clip_is_correct = int(clip_pred == y)
            clip_correct += clip_is_correct

            while tda_ptr < len(tda_sequence) and tda_sequence[tda_ptr] != idx:
                warm_idx = tda_sequence[tda_ptr]
                tda.predict(payload["image_features"][warm_idx])
                tda_ptr += 1
            tda_pred_t, tda_conf_t, tda_logits_t = tda.predict(x)
            tda_ptr += 1
            tda_pred = int(tda_pred_t.item())
            tda_logits = tda_logits_t.squeeze(0)
            tda_probs = torch.softmax(tda_logits, dim=-1)
            tda_is_correct = int(tda_pred == y)
            tda_correct += tda_is_correct

            while freetta_ptr < len(freetta_sequence) and freetta_sequence[freetta_ptr] != idx:
                warm_idx = freetta_sequence[freetta_ptr]
                warm_x = payload["image_features"][warm_idx]
                warm_logits = payload["raw_clip_logits"][warm_idx].unsqueeze(0)
                freetta.predict(warm_x, warm_logits)
                freetta_ptr += 1
            freetta_pred_t, freetta_probs_t = freetta.predict(x, clip_logits.unsqueeze(0))
            freetta_ptr += 1
            freetta_pred = int(freetta_pred_t.squeeze(0).item())
            freetta_probs = freetta_probs_t.squeeze(0)
            freetta_is_correct = int(freetta_pred == y)
            freetta_correct += freetta_is_correct

            edge_pred_t, edge_probs_t, edge_stats = edge.predict_and_adapt(x, clip_logits)
            edge_pred = int(edge_pred_t.squeeze(0).item())
            edge_probs = edge_probs_t.squeeze(0)
            edge_is_correct = int(edge_pred == y)
            edge_correct += edge_is_correct

            rows.append(
                {
                    "dataset": payload["dataset"],
                    "sample_index": idx,
                    "stream_step": stream_step,
                    "label": y,
                    "clip_pred": clip_pred,
                    "clip_correct": clip_is_correct,
                    "clip_confidence": float(torch.max(clip_probs).item()),
                    "clip_entropy": entropy_from_probs(clip_probs),
                    "clip_margin": top2_margin(clip_logits),
                    "tda_pred": tda_pred,
                    "tda_correct": tda_is_correct,
                    "tda_order_rank": int(tda_rank[idx]),
                    "tda_confidence": float(tda_conf_t.item()),
                    "tda_entropy": entropy_from_probs(tda_probs),
                    "freetta_pred": freetta_pred,
                    "freetta_correct": freetta_is_correct,
                    "freetta_order_rank": int(freetta_rank[idx]),
                    "freetta_confidence": float(torch.max(freetta_probs).item()),
                    "freetta_entropy": entropy_from_probs(freetta_probs),
                    "edgefreetta_pred": edge_pred,
                    "edgefreetta_correct": edge_is_correct,
                    "edgefreetta_confidence": float(torch.max(edge_probs).item()),
                    "edgefreetta_entropy": entropy_from_probs(edge_probs),
                    "edgefreetta_updated": float(edge_stats["updated"]),
                    "edgefreetta_update_weight": float(edge_stats["update_weight"]),
                    "edgefreetta_adapter_norm": float(edge_stats["adapter_norm"]),
                    "edgefreetta_residual_norm": float(edge_stats["residual_norm"]),
                }
            )

    total = max(int(payload["num_samples"]), 1)
    summary = {
        "dataset": payload["dataset"],
        "samples": int(payload["num_samples"]),
        "clip_acc": clip_correct / total,
        "tda_acc": tda_correct / total,
        "freetta_acc": freetta_correct / total,
        "edgefreetta_acc": edge_correct / total,
        "tda_minus_clip": (tda_correct - clip_correct) / total,
        "freetta_minus_clip": (freetta_correct - clip_correct) / total,
        "edgefreetta_minus_clip": (edge_correct - clip_correct) / total,
        "freetta_minus_tda": (freetta_correct - tda_correct) / total,
        "edgefreetta_minus_tda": (edge_correct - tda_correct) / total,
        "edgefreetta_minus_freetta": (edge_correct - freetta_correct) / total,
    }
    return summary, pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    matrix = [headers] + rows
    widths = [max(len(row[i]) for row in matrix) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_overall_accuracy(summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = summary_df["dataset"].tolist()
    methods = ["clip_acc", "tda_acc", "freetta_acc", "edgefreetta_acc"]
    labels = ["CLIP", "TDA", "FreeTTA", "EdgeFreeTTA"]
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, (method, label) in enumerate(zip(methods, labels)):
        ax.bar(x + (offset - 1.5) * width, summary_df[method].to_numpy(), width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy")
    ax.set_title("Full-Dataset Accuracy Comparison")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dataset_dynamics(dataset_df: pd.DataFrame, output_path: Path) -> None:
    df = dataset_df.sort_values("stream_step").reset_index(drop=True)
    steps = np.arange(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, label in [
        ("clip_correct", "CLIP"),
        ("tda_correct", "TDA"),
        ("freetta_correct", "FreeTTA"),
        ("edgefreetta_correct", "EdgeFreeTTA"),
    ]:
        cumulative = df[col].astype(float).cumsum() / steps
        axes[0].plot(steps, cumulative, label=label, linewidth=2)
    axes[0].set_title(f"{df['dataset'].iloc[0]}: Cumulative Accuracy")
    axes[0].set_xlabel("Stream Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    window = max(25, len(df) // 20)
    for col, label in [
        ("clip_correct", "CLIP"),
        ("tda_correct", "TDA"),
        ("freetta_correct", "FreeTTA"),
        ("edgefreetta_correct", "EdgeFreeTTA"),
    ]:
        rolling = df[col].astype(float).rolling(window, min_periods=window).mean()
        axes[1].plot(steps, rolling, label=label, linewidth=2)
    axes[1].set_title(f"{df['dataset'].iloc[0]}: Rolling Accuracy (window={window})")
    axes[1].set_xlabel("Stream Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tda_freetta_report(dataset_df: pd.DataFrame, output_path: Path) -> None:
    df = dataset_df.copy()
    entropy_rank = df["clip_entropy"].rank(method="first")
    df["difficulty_bin"] = pd.qcut(entropy_rank, q=3, labels=["easy", "medium", "hard"])
    grouped = (
        df.groupby("difficulty_bin", observed=True)[["clip_correct", "tda_correct", "freetta_correct"]]
        .mean()
        .reset_index()
    )

    changes = {
        "TDA beneficial": int(((df["tda_pred"] != df["clip_pred"]) & (df["tda_correct"] == 1) & (df["clip_correct"] == 0)).sum()),
        "TDA harmful": int(((df["tda_pred"] != df["clip_pred"]) & (df["tda_correct"] == 0) & (df["clip_correct"] == 1)).sum()),
        "FreeTTA beneficial": int(((df["freetta_pred"] != df["clip_pred"]) & (df["freetta_correct"] == 1) & (df["clip_correct"] == 0)).sum()),
        "FreeTTA harmful": int(((df["freetta_pred"] != df["clip_pred"]) & (df["freetta_correct"] == 0) & (df["clip_correct"] == 1)).sum()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(grouped))
    width = 0.25
    axes[0].bar(x - width, grouped["clip_correct"], width=width, label="CLIP")
    axes[0].bar(x, grouped["tda_correct"], width=width, label="TDA")
    axes[0].bar(x + width, grouped["freetta_correct"], width=width, label="FreeTTA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grouped["difficulty_bin"].astype(str).tolist())
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{df['dataset'].iloc[0]}: Accuracy by Difficulty Bin")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    keys = list(changes.keys())
    vals = [changes[key] for key in keys]
    axes[1].bar(np.arange(len(keys)), vals, color=["#2f7ed8", "#d84a4a", "#6aa84f", "#f1a340"])
    axes[1].set_xticks(np.arange(len(keys)))
    axes[1].set_xticklabels(keys, rotation=20, ha="right")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{df['dataset'].iloc[0]}: CLIP Decision Changes")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP, TDA, FreeTTA, and EdgeFreeTTA on all datasets with plots")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/final_method_suite")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--stream-seed", type=int, default=None, help="Use None for natural order")
    parser.add_argument("--tda-config", default="outputs/tuning/best_tda_run_results.json")
    parser.add_argument("--freetta-config", default="outputs/tuning/best_freetta_run_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    features_dir = Path(args.features_dir)
    tda_params_map = load_best_tda_params(Path(args.tda_config))
    freetta_params_map = load_best_freetta_params(Path(args.freetta_config))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    summary_rows: list[dict] = []
    all_frames: list[pd.DataFrame] = []

    for dataset in [str(x).lower() for x in args.datasets]:
        payload = load_dataset(dataset=dataset, device=device, features_dir=features_dir)
        clip_order = get_order(payload["num_samples"], device=device, seed=args.stream_seed)
        summary, per_sample = run_dataset(
            payload=payload,
            clip_order=clip_order,
            tda_cfg=tda_params_map[dataset],
            freetta_cfg=freetta_params_map[dataset],
            edge_params=DEFAULT_EDGE_PARAMS[dataset],
        )
        summary_rows.append(summary)
        all_frames.append(per_sample)

        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        per_sample.to_csv(dataset_dir / "per_sample_metrics.csv", index=False)
        plot_dataset_dynamics(per_sample, dataset_dir / "adaptation_dynamics.png")
        plot_tda_freetta_report(per_sample, dataset_dir / "tda_vs_freetta_report.png")

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    per_sample_df = pd.concat(all_frames, ignore_index=True)

    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    write_markdown_table(summary_df, output_dir / "summary_table.md")
    per_sample_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    plot_overall_accuracy(summary_df, output_dir / "overall_accuracy.png")

    report = {
        "device": str(device),
        "datasets": [str(x).lower() for x in args.datasets],
        "stream_seed": args.stream_seed,
        "tda_config": args.tda_config,
        "freetta_config": args.freetta_config,
        "summary_table_csv": str(output_dir / "summary_table.csv"),
        "summary_table_md": str(output_dir / "summary_table.md"),
        "overall_accuracy_plot": str(output_dir / "overall_accuracy.png"),
        "per_sample_metrics_csv": str(output_dir / "per_sample_metrics.csv"),
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
