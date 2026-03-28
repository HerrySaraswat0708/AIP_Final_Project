from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import FreeTTAOnlineEMAdapter, TDAMemoryAdapter
from src.feature_store import list_available_datasets, load_dataset_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proposal comparison: TDA vs FreeTTA."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing *_image_features.npy, *_text_features.npy, *_labels.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proposal_comparison"),
        help="Where results and plots will be saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dtd", "caltech", "eurosat", "pets", "imagenet"],
        help="Dataset keys to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick debug runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tda-cache-size", type=int, default=1024)
    parser.add_argument("--tda-top-k", type=int, default=64)
    parser.add_argument("--tda-alpha", type=float, default=0.45)
    parser.add_argument("--tda-temperature", type=float, default=20.0)

    parser.add_argument("--freetta-alpha", type=float, default=0.05)
    parser.add_argument("--freetta-beta", type=float, default=7.0)
    # parser.add_argument("--freetta-cov-momentum", type=float, default=0.02)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_adapter(
    method_name: str,
    adapter,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: np.ndarray,
) -> Dict[str, object]:
    total = int(labels.shape[0])
    running_accuracy: List[float] = []
    per_sample_ms: List[float] = []
    correctness_flags: List[int] = []
    predictions: List[int] = []
    confidences: List[float] = []
    entropies: List[float] = []
    weights: List[float] = []
    extra_series: Dict[str, List[float]] = {}

    correct = 0
    peak_state_nbytes = int(adapter.state_nbytes())

    for idx in range(total):
        x = image_features[idx]
        x = F.normalize(x, dim=-1)
        clip_logits = 100.0 * torch.matmul(text_features, x)
        clip_probs = torch.softmax(clip_logits, dim=-1)

        start = time.perf_counter_ns()
        out = adapter.process_sample(x=x, clip_probs=clip_probs, clip_logits=clip_logits)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        per_sample_ms.append(float(elapsed_ms))

        hit = int(out.pred == int(labels[idx]))
        predictions.append(int(out.pred))
        correctness_flags.append(hit)
        correct += hit
        running_accuracy.append(correct / float(idx + 1))

        for key, val in out.extra.items():
            extra_series.setdefault(key, []).append(float(val))

        if "entropy" in out.extra:
            entropies.append(float(out.extra["entropy"]))
        if "pred_confidence" in out.extra:
            confidences.append(float(out.extra["pred_confidence"]))
        if "weight" in out.extra:
            weights.append(float(out.extra["weight"]))

        peak_state_nbytes = max(peak_state_nbytes, int(adapter.state_nbytes()))

    final_accuracy = float(correct / max(total, 1))
    threshold = 0.9 * final_accuracy
    samples_to_90 = total
    for i, val in enumerate(running_accuracy, start=1):
        if val >= threshold:
            samples_to_90 = i
            break

    stability_window = max(1, total // 4)
    stability_std = float(np.std(running_accuracy[-stability_window:]))

    result = {
        "method": method_name,
        "final_accuracy": final_accuracy,
        "mean_time_ms": float(np.mean(per_sample_ms)),
        "std_time_ms": float(np.std(per_sample_ms)),
        "peak_state_mb": float(peak_state_nbytes / (1024.0 * 1024.0)),
        "samples_to_90pct_final_acc": int(samples_to_90),
        "stability_std_last_quarter": stability_std,
        "running_accuracy": running_accuracy,
        "correctness_flags": correctness_flags,
        "predictions": predictions,
        "confidences": confidences,
        "entropies": entropies,
        "weights": weights,
        "extra_series": extra_series,
    }

    if hasattr(adapter, "get_class_means"):
        initial_mu, final_mu = adapter.get_class_means()
        result["initial_mu"] = initial_mu.numpy().astype(np.float32)
        result["final_mu"] = final_mu.numpy().astype(np.float32)

    return result


def expected_calibration_error(
    confidences: List[float], correctness: List[int], num_bins: int = 15
) -> float:
    if not confidences:
        return math.nan
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness[: len(confidences)], dtype=np.float64)
    if conf.size == 0:
        return math.nan

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < num_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(corr[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def prediction_churn_rate(predictions: List[int], labels: np.ndarray) -> float:
    if len(predictions) < 2:
        return 0.0
    p = np.asarray(predictions, dtype=np.int64)
    y = np.asarray(labels[: len(predictions)], dtype=np.int64)
    same_label_mask = y[1:] == y[:-1]
    denom = int(np.sum(same_label_mask))
    if denom == 0:
        return 0.0
    unstable_changes = np.sum((p[1:] != p[:-1]) & same_label_mask)
    return float(unstable_changes / denom)


def error_recovery_latency(correctness: List[int]) -> float:
    if not correctness:
        return math.nan
    arr = np.asarray(correctness, dtype=np.int64)
    recoveries: List[int] = []
    i = 0
    n = len(arr)
    while i < n:
        if arr[i] == 1:
            i += 1
            continue
        j = i
        while j < n and arr[j] == 0:
            j += 1
        if j < n:
            recoveries.append(j - i)
        i = j
    if not recoveries:
        return 0.0
    return float(np.mean(recoveries))


def compute_disagreement_advantage(
    labels: np.ndarray, tda_preds: List[int], freetta_preds: List[int]
) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    t = np.asarray(tda_preds, dtype=np.int64)
    f = np.asarray(freetta_preds, dtype=np.int64)
    disagree = t != f
    total = len(y)
    n_disagree = int(np.sum(disagree))

    if n_disagree == 0:
        return {
            "disagreement_rate": 0.0,
            "freetta_win_rate_on_disagreement": 0.0,
            "tda_win_rate_on_disagreement": 0.0,
            "net_advantage_freetta_minus_tda": 0.0,
        }

    t_correct = t == y
    f_correct = f == y
    freetta_wins = int(np.sum(disagree & f_correct & (~t_correct)))
    tda_wins = int(np.sum(disagree & t_correct & (~f_correct)))

    return {
        "disagreement_rate": float(n_disagree / total),
        "freetta_win_rate_on_disagreement": float(freetta_wins / n_disagree),
        "tda_win_rate_on_disagreement": float(tda_wins / n_disagree),
        "net_advantage_freetta_minus_tda": float((freetta_wins - tda_wins) / n_disagree),
    }


def compute_entropy_conditioned_gains(
    entropy_values: List[float],
    tda_correct: List[int],
    freetta_correct: List[int],
    bins: int = 4,
) -> List[Dict[str, float]]:
    e = np.asarray(entropy_values, dtype=np.float64)
    t = np.asarray(tda_correct[: len(e)], dtype=np.float64)
    f = np.asarray(freetta_correct[: len(e)], dtype=np.float64)
    if e.size == 0:
        return []

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(e, quantiles)
    rows: List[Dict[str, float]] = []
    for i in range(bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        mask = (e >= lo) & (e < hi if i < bins - 1 else e <= hi)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": i,
                    "entropy_min": lo,
                    "entropy_max": hi,
                    "count": 0,
                    "tda_acc": math.nan,
                    "freetta_acc": math.nan,
                    "freetta_minus_tda": math.nan,
                }
            )
            continue
        t_acc = float(np.mean(t[mask]))
        f_acc = float(np.mean(f[mask]))
        rows.append(
            {
                "bin_index": i,
                "entropy_min": lo,
                "entropy_max": hi,
                "count": count,
                "tda_acc": t_acc,
                "freetta_acc": f_acc,
                "freetta_minus_tda": f_acc - t_acc,
            }
        )
    return rows


def summarize_internal_metrics(
    method_name: str, extra_series: Dict[str, List[float]], correctness_flags: List[int]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, series in sorted(extra_series.items()):
        values = np.asarray(series, dtype=np.float64)
        if values.size == 0:
            continue
        corr_flags = np.asarray(correctness_flags[: values.size], dtype=np.float64)
        if np.std(values) < 1e-12 or np.std(corr_flags) < 1e-12:
            corr = math.nan
        else:
            corr = float(np.corrcoef(values, corr_flags)[0, 1])

        correct_mask = corr_flags == 1.0
        incorrect_mask = corr_flags == 0.0
        mean_correct = float(np.mean(values[correct_mask])) if np.any(correct_mask) else math.nan
        mean_incorrect = (
            float(np.mean(values[incorrect_mask])) if np.any(incorrect_mask) else math.nan
        )

        rows.append(
            {
                "method": method_name,
                "metric": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "corr_with_correctness": corr,
                "mean_when_correct": mean_correct,
                "mean_when_incorrect": mean_incorrect,
            }
        )
    return rows


def _window_mean(series: List[float], frac: float, tail: bool = False) -> float:
    if not series:
        return math.nan
    n = max(1, int(len(series) * frac))
    if tail:
        segment = series[-n:]
    else:
        segment = series[:n]
    return float(np.mean(np.asarray(segment, dtype=np.float64)))


def write_architecture_internal_analysis(
    dataset_dir: Path,
    dataset: str,
    method_results: Dict[str, Dict[str, object]],
    custom_metrics: Dict[str, object],
    internal_rows: List[Dict[str, object]],
) -> Path:
    tda_series = method_results["tda"]["extra_series"]
    freetta_series = method_results["freetta"]["extra_series"]
    disagreement = custom_metrics["disagreement_advantage"]

    tda_cache_fill_start = _window_mean(tda_series.get("cache_fill_ratio", []), 0.1, tail=False)
    tda_cache_fill_end = _window_mean(tda_series.get("cache_fill_ratio", []), 0.1, tail=True)
    freetta_mu_drift_start = _window_mean(
        freetta_series.get("mu_drift_from_init", []), 0.1, tail=False
    )
    freetta_mu_drift_end = _window_mean(
        freetta_series.get("mu_drift_from_init", []), 0.1, tail=True
    )

    row_lookup = {(r["method"], r["metric"]): r for r in internal_rows}
    tda_jsd_corr = row_lookup.get(("tda", "jsd_clip_fused"), {}).get("corr_with_correctness", math.nan)
    freetta_em_corr = row_lookup.get(("freetta", "em_weight"), {}).get(
        "corr_with_correctness", math.nan
    )
    freetta_mu_update_corr = row_lookup.get(("freetta", "mu_update_norm"), {}).get(
        "corr_with_correctness", math.nan
    )

    lines = [
        f"# Architecture/Loss/Internal Analysis - {dataset}",
        "",
        "## Method Mechanics Linked to Architecture and Objectives",
        "- TDA is memory-based and non-parametric: predictions are fused with a cache retrieval posterior.",
        "- FreeTTA is generative and online-EM style: it updates class distribution parameters using entropy-weighted posteriors.",
        "",
        "## Empirical Link: TDA Internal Signals",
        f"- Cache fill ratio rose from ~{tda_cache_fill_start:.3f} (early) to ~{tda_cache_fill_end:.3f} (late), showing growing memory reliance.",
        f"- Corr(jsd_clip_fused, correctness) = {tda_jsd_corr:.4f}.",
        f"- Prediction churn = {custom_metrics['tda']['prediction_churn_rate']:.4f}, recovery latency = {custom_metrics['tda']['error_recovery_latency']:.4f}.",
        "",
        "## Empirical Link: FreeTTA Internal Signals",
        f"- Mean EM confidence-weight vs correctness corr = {freetta_em_corr:.4f}.",
        f"- Corr(mu_update_norm, correctness) = {freetta_mu_update_corr:.4f}.",
        f"- Mean prototype drift from init moved from ~{freetta_mu_drift_start:.4f} (early) to ~{freetta_mu_drift_end:.4f} (late).",
        f"- Prediction churn = {custom_metrics['freetta']['prediction_churn_rate']:.4f}, recovery latency = {custom_metrics['freetta']['error_recovery_latency']:.4f}.",
        "",
        "## TDA vs FreeTTA Decision Regions",
        f"- Disagreement rate = {disagreement['disagreement_rate']*100.0:.2f}%.",
        f"- On disagreement samples: FreeTTA win rate = {disagreement['freetta_win_rate_on_disagreement']*100.0:.2f}%, TDA win rate = {disagreement['tda_win_rate_on_disagreement']*100.0:.2f}%.",
        "- This quantifies which internal mechanism (memory retrieval vs online distribution update) is more reliable in contested regions.",
    ]

    out = dataset_dir / "architecture_internal_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def compute_entropy_summary(
    entropies: List[float], correctness_flags: List[int], bins: int = 10
) -> Dict[str, object]:
    if not entropies:
        return {"correlation_entropy_correct": math.nan, "binned_accuracy": []}

    e = np.asarray(entropies, dtype=np.float64)
    c = np.asarray(correctness_flags[: len(entropies)], dtype=np.float64)

    if np.std(e) < 1e-12 or np.std(c) < 1e-12:
        corr = math.nan
    else:
        corr = float(np.corrcoef(e, c)[0, 1])

    edges = np.linspace(float(e.min()), float(e.max()) + 1e-9, bins + 1)
    binned_rows = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (e >= lo) & (e < hi if i < bins - 1 else e <= hi)
        count = int(mask.sum())
        if count == 0:
            acc = math.nan
        else:
            acc = float(np.mean(c[mask]))
        binned_rows.append(
            {
                "bin_index": i,
                "entropy_min": float(lo),
                "entropy_max": float(hi),
                "count": count,
                "accuracy": acc,
            }
        )

    return {"correlation_entropy_correct": corr, "binned_accuracy": binned_rows}


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_adaptation_dynamics(
    dataset_dir: Path, dataset: str, method_results: Dict[str, Dict[str, object]]
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for method_name, result in method_results.items():
        acc = np.asarray(result["running_accuracy"], dtype=np.float32)
        ax.plot(np.arange(1, len(acc) + 1), acc * 100.0, label=method_name, linewidth=1.6)
    ax.set_title(f"Adaptation Dynamics - {dataset}")
    ax.set_xlabel("Processed test samples")
    ax.set_ylabel("Running Top-1 accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    out = dataset_dir / "adaptation_dynamics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_entropy_analysis(
    dataset_dir: Path,
    dataset: str,
    entropies: List[float],
    correctness_flags: List[int],
    binned_rows: List[Dict[str, object]],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    e = np.asarray(entropies, dtype=np.float32)
    c = np.asarray(correctness_flags[: len(entropies)], dtype=np.float32)

    sample_size = min(4000, len(e))
    if sample_size > 0:
        indices = np.linspace(0, len(e) - 1, sample_size).astype(int)
        axes[0].scatter(e[indices], c[indices], s=6, alpha=0.2, color="#1f77b4")
    axes[0].set_title("Entropy vs Correctness")
    axes[0].set_xlabel("CLIP entropy")
    axes[0].set_ylabel("Correct prediction (0/1)")
    axes[0].grid(alpha=0.2)

    x = [row["bin_index"] for row in binned_rows]
    y = [row["accuracy"] * 100.0 if not math.isnan(row["accuracy"]) else np.nan for row in binned_rows]
    axes[1].plot(x, y, marker="o", color="#2ca02c")
    axes[1].set_title("Binned Entropy Accuracy")
    axes[1].set_xlabel("Entropy bin")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Uncertainty Analysis - {dataset}")
    out = dataset_dir / "entropy_analysis.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_efficiency(
    dataset_dir: Path, dataset: str, method_results: Dict[str, Dict[str, object]]
) -> Path:
    methods = list(method_results.keys())
    acc = [method_results[m]["final_accuracy"] * 100.0 for m in methods]
    time_ms = [method_results[m]["mean_time_ms"] for m in methods]
    mem_mb = [method_results[m]["peak_state_mb"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(methods, acc, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Final Accuracy")
    axes[0].set_ylabel("Top-1 (%)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(methods, time_ms, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_title("Mean Time / Sample")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(methods, mem_mb, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[2].set_title("Peak Adapter State")
    axes[2].set_ylabel("MB")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Computational Efficiency - {dataset}")
    out = dataset_dir / "efficiency_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_custom_comparisons(
    dataset_dir: Path,
    dataset: str,
    custom_metrics: Dict[str, object],
) -> Path:
    disagreement = custom_metrics["disagreement_advantage"]
    tda_metrics = custom_metrics["tda"]
    freetta_metrics = custom_metrics["freetta"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    axes[0].bar(
        ["TDA wins", "FreeTTA wins"],
        [
            disagreement["tda_win_rate_on_disagreement"] * 100.0,
            disagreement["freetta_win_rate_on_disagreement"] * 100.0,
        ],
        color=["#f58518", "#54a24b"],
    )
    axes[0].set_title("Disagreement Win Rate")
    axes[0].set_ylabel("Win rate (%)")

    axes[1].bar(
        ["TDA", "FreeTTA"],
        [tda_metrics["prediction_churn_rate"], freetta_metrics["prediction_churn_rate"]],
        color=["#f58518", "#54a24b"],
    )
    axes[1].set_title("Prediction Churn")
    axes[1].set_ylabel("Class-switch ratio")

    axes[2].bar(
        ["TDA", "FreeTTA"],
        [tda_metrics["ece"], freetta_metrics["ece"]],
        color=["#f58518", "#54a24b"],
    )
    axes[2].set_title("Calibration Error (ECE)")
    axes[2].set_ylabel("ECE (lower is better)")

    fig.suptitle(f"Custom TDA vs FreeTTA Comparisons - {dataset}")
    out = dataset_dir / "custom_comparisons.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_distribution_modeling(
    dataset_dir: Path,
    dataset: str,
    image_features: np.ndarray,
    initial_mu: np.ndarray,
    final_mu: np.ndarray,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)
    total = image_features.shape[0]
    sample_n = min(total, 5000)
    indices = rng.choice(total, size=sample_n, replace=False)
    sampled = image_features[indices]

    stack = np.vstack([sampled, initial_mu, final_mu])
    pca = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(stack)

    p_sample = proj[:sample_n]
    p_init = proj[sample_n : sample_n + initial_mu.shape[0]]
    p_final = proj[sample_n + initial_mu.shape[0] :]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(p_sample[:, 0], p_sample[:, 1], s=3, alpha=0.12, color="#6f6f6f", label="Samples")
    ax.scatter(p_init[:, 0], p_init[:, 1], s=45, marker="x", color="#d62728", label="Initial means")
    ax.scatter(p_final[:, 0], p_final[:, 1], s=45, marker="+", color="#1f77b4", label="Adapted means")

    move_count = min(30, p_init.shape[0])
    for i in range(move_count):
        ax.plot(
            [p_init[i, 0], p_final[i, 0]],
            [p_init[i, 1], p_final[i, 1]],
            linewidth=0.8,
            alpha=0.35,
            color="#2ca02c",
        )

    ax.set_title(f"Distribution Modeling (PCA) - {dataset}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.15)
    ax.legend(loc="best")

    out = dataset_dir / "distribution_modeling_pca.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    run_config = {
        "seed": args.seed,
        "tda": {
            "cache_size": args.tda_cache_size,
            "top_k": args.tda_top_k,
            "alpha": args.tda_alpha,
            "temperature": args.tda_temperature,
        },
        "freetta": {
            "alpha": args.freetta_alpha,
            "beta": args.freetta_beta,
            "cov_momentum": args.freetta_cov_momentum,
        },
    }

    available = list_available_datasets(args.features_dir)
    print(f"[Info] Device: {device}")
    print(f"[Info] Available datasets in {args.features_dir}: {available}")
    print(f"[Info] Run config: {run_config}")

    overall_rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    for requested_dataset in args.datasets:
        print(f"\n[Dataset] {requested_dataset}")
        try:
            payload = load_dataset_features(args.features_dir, requested_dataset)
        except FileNotFoundError as exc:
            print(f"[Skip] {exc}")
            skipped.append(requested_dataset)
            continue

        dataset_key = str(payload["dataset_key"])
        image_features = payload["image_features"]
        text_features = payload["text_features"]
        labels = payload["labels"]

        if args.max_samples is not None:
            image_features = image_features[: args.max_samples]
            labels = labels[: args.max_samples]

        dataset_dir = output_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        image_t = torch.from_numpy(image_features).to(device=device, dtype=torch.float32)
        text_t = torch.from_numpy(text_features).to(device=device, dtype=torch.float32)
        text_t = F.normalize(text_t, dim=-1)

        num_classes, feature_dim = text_t.shape[0], text_t.shape[1]
        print(
            f"[Info] samples={image_t.shape[0]}, classes={num_classes}, feature_dim={feature_dim}"
        )

        tda = TDAMemoryAdapter(
            num_classes=num_classes,
            feature_dim=feature_dim,
            cache_size=args.tda_cache_size,
            top_k=args.tda_top_k,
            alpha=args.tda_alpha,
            temperature=args.tda_temperature,
            device=device,
        )
        freetta = FreeTTAOnlineEMAdapter(
            text_features=text_t,
            alpha=args.freetta_alpha,
            beta=args.freetta_beta,
            cov_momentum=args.freetta_cov_momentum,
            device=device,
        )

        method_results: Dict[str, Dict[str, object]] = {}
        method_results["tda"] = evaluate_adapter(
            method_name="tda",
            adapter=tda,
            image_features=image_t,
            text_features=text_t,
            labels=labels,
        )
        method_results["freetta"] = evaluate_adapter(
            method_name="freetta",
            adapter=freetta,
            image_features=image_t,
            text_features=text_t,
            labels=labels,
        )

        freetta_entropy = method_results["freetta"]["entropies"]
        freetta_flags = method_results["freetta"]["correctness_flags"]
        entropy_summary = compute_entropy_summary(
            entropies=freetta_entropy, correctness_flags=freetta_flags, bins=10
        )
        custom_metrics = {
            "tda": {
                "prediction_churn_rate": prediction_churn_rate(
                    method_results["tda"]["predictions"], labels
                ),
                "error_recovery_latency": error_recovery_latency(
                    method_results["tda"]["correctness_flags"]
                ),
                "ece": expected_calibration_error(
                    method_results["tda"]["confidences"],
                    method_results["tda"]["correctness_flags"],
                ),
            },
            "freetta": {
                "prediction_churn_rate": prediction_churn_rate(
                    method_results["freetta"]["predictions"], labels
                ),
                "error_recovery_latency": error_recovery_latency(
                    method_results["freetta"]["correctness_flags"]
                ),
                "ece": expected_calibration_error(
                    method_results["freetta"]["confidences"],
                    method_results["freetta"]["correctness_flags"],
                ),
            },
            "disagreement_advantage": compute_disagreement_advantage(
                labels=labels,
                tda_preds=method_results["tda"]["predictions"],
                freetta_preds=method_results["freetta"]["predictions"],
            ),
            "entropy_conditioned_gains": compute_entropy_conditioned_gains(
                entropy_values=freetta_entropy,
                tda_correct=method_results["tda"]["correctness_flags"],
                freetta_correct=method_results["freetta"]["correctness_flags"],
                bins=4,
            ),
        }
        internal_rows: List[Dict[str, object]] = []
        internal_rows.extend(
            summarize_internal_metrics(
                method_name="tda",
                extra_series=method_results["tda"]["extra_series"],
                correctness_flags=method_results["tda"]["correctness_flags"],
            )
        )
        internal_rows.extend(
            summarize_internal_metrics(
                method_name="freetta",
                extra_series=method_results["freetta"]["extra_series"],
                correctness_flags=method_results["freetta"]["correctness_flags"],
            )
        )

        adaptation_plot = plot_adaptation_dynamics(
            dataset_dir=dataset_dir, dataset=dataset_key, method_results=method_results
        )
        efficiency_plot = plot_efficiency(
            dataset_dir=dataset_dir, dataset=dataset_key, method_results=method_results
        )
        entropy_plot = plot_entropy_analysis(
            dataset_dir=dataset_dir,
            dataset=dataset_key,
            entropies=freetta_entropy,
            correctness_flags=freetta_flags,
            binned_rows=entropy_summary["binned_accuracy"],
        )
        custom_plot = plot_custom_comparisons(
            dataset_dir=dataset_dir,
            dataset=dataset_key,
            custom_metrics=custom_metrics,
        )
        architecture_md = write_architecture_internal_analysis(
            dataset_dir=dataset_dir,
            dataset=dataset_key,
            method_results=method_results,
            custom_metrics=custom_metrics,
            internal_rows=internal_rows,
        )

        dist_plot_path = None
        if "initial_mu" in method_results["freetta"] and "final_mu" in method_results["freetta"]:
            dist_plot_path = plot_distribution_modeling(
                dataset_dir=dataset_dir,
                dataset=dataset_key,
                image_features=image_features,
                initial_mu=method_results["freetta"]["initial_mu"],
                final_mu=method_results["freetta"]["final_mu"],
                seed=args.seed,
            )

        compact_result = {
            "dataset": dataset_key,
            "num_samples": int(len(labels)),
            "config": run_config,
            "feature_files": {
                "image": payload["image_path"],
                "text": payload["text_path"],
                "labels": payload["labels_path"],
            },
            "methods": {
                name: {
                    "final_accuracy": value["final_accuracy"],
                    "mean_time_ms": value["mean_time_ms"],
                    "std_time_ms": value["std_time_ms"],
                    "peak_state_mb": value["peak_state_mb"],
                    "samples_to_90pct_final_acc": value["samples_to_90pct_final_acc"],
                    "stability_std_last_quarter": value["stability_std_last_quarter"],
                }
                for name, value in method_results.items()
            },
            "entropy_analysis": entropy_summary,
            "custom_comparisons": custom_metrics,
            "architecture_internal_metrics": internal_rows,
            "artifacts": {
                "adaptation_dynamics": str(adaptation_plot),
                "efficiency_summary": str(efficiency_plot),
                "entropy_analysis": str(entropy_plot),
                "custom_comparisons": str(custom_plot),
                "architecture_internal_analysis": str(architecture_md),
                "distribution_modeling": str(dist_plot_path) if dist_plot_path else None,
            },
        }

        save_json(dataset_dir / "dataset_report.json", compact_result)

        dynamics_rows = []
        total_steps = len(method_results["tda"]["running_accuracy"])
        for i in range(total_steps):
            dynamics_rows.append(
                {
                    "step": i + 1,
                    "tda_running_acc": method_results["tda"]["running_accuracy"][i],
                    "freetta_running_acc": method_results["freetta"]["running_accuracy"][i],
                }
            )
        save_csv(
            dataset_dir / "adaptation_dynamics.csv",
            rows=dynamics_rows,
            fieldnames=[
                "step",
                "tda_running_acc",
                "freetta_running_acc",
            ],
        )

        save_csv(
            dataset_dir / "freetta_entropy_bins.csv",
            rows=entropy_summary["binned_accuracy"],
            fieldnames=["bin_index", "entropy_min", "entropy_max", "count", "accuracy"],
        )
        save_csv(
            dataset_dir / "custom_entropy_conditioned_gains.csv",
            rows=custom_metrics["entropy_conditioned_gains"],
            fieldnames=[
                "bin_index",
                "entropy_min",
                "entropy_max",
                "count",
                "tda_acc",
                "freetta_acc",
                "freetta_minus_tda",
            ],
        )
        save_csv(
            dataset_dir / "internal_metric_summary.csv",
            rows=internal_rows,
            fieldnames=[
                "method",
                "metric",
                "mean",
                "std",
                "corr_with_correctness",
                "mean_when_correct",
                "mean_when_incorrect",
            ],
        )

        for method_name, values in method_results.items():
            extra_row = {
                "prediction_churn_rate": custom_metrics[method_name]["prediction_churn_rate"],
                "error_recovery_latency": custom_metrics[method_name]["error_recovery_latency"],
                "ece": custom_metrics[method_name]["ece"],
            }
            overall_rows.append(
                {
                    "dataset": dataset_key,
                    "method": method_name,
                    "final_accuracy": values["final_accuracy"],
                    "mean_time_ms": values["mean_time_ms"],
                    "std_time_ms": values["std_time_ms"],
                    "peak_state_mb": values["peak_state_mb"],
                    "samples_to_90pct_final_acc": values["samples_to_90pct_final_acc"],
                    "stability_std_last_quarter": values["stability_std_last_quarter"],
                    **extra_row,
                }
            )

        overall_rows.append(
            {
                "dataset": dataset_key,
                "method": "custom_disagreement",
                "final_accuracy": math.nan,
                "mean_time_ms": math.nan,
                "std_time_ms": math.nan,
                "peak_state_mb": math.nan,
                "samples_to_90pct_final_acc": math.nan,
                "stability_std_last_quarter": math.nan,
                "prediction_churn_rate": math.nan,
                "error_recovery_latency": math.nan,
                "ece": math.nan,
                "disagreement_rate": custom_metrics["disagreement_advantage"]["disagreement_rate"],
                "freetta_win_rate_on_disagreement": custom_metrics["disagreement_advantage"][
                    "freetta_win_rate_on_disagreement"
                ],
                "tda_win_rate_on_disagreement": custom_metrics["disagreement_advantage"][
                    "tda_win_rate_on_disagreement"
                ],
            }
        )

        for method_name in ["tda", "freetta"]:
            print(
                f"[Result] {dataset_key:>8} | {method_name:>8} | "
                f"acc={method_results[method_name]['final_accuracy']*100:.2f}% | "
                f"time={method_results[method_name]['mean_time_ms']:.4f} ms | "
                f"state={method_results[method_name]['peak_state_mb']:.4f} MB"
            )

    if overall_rows:
        save_csv(
            output_dir / "summary_metrics.csv",
            rows=overall_rows,
            fieldnames=[
                "dataset",
                "method",
                "final_accuracy",
                "mean_time_ms",
                "std_time_ms",
                "peak_state_mb",
                "samples_to_90pct_final_acc",
                "stability_std_last_quarter",
                "prediction_churn_rate",
                "error_recovery_latency",
                "ece",
                "disagreement_rate",
                "freetta_win_rate_on_disagreement",
                "tda_win_rate_on_disagreement",
            ],
        )

    final_report = {
        "requested_datasets": args.datasets,
        "available_datasets": available,
        "skipped_datasets": skipped,
        "device": device,
        "config": run_config,
        "output_dir": str(output_dir),
        "rows_written": len(overall_rows),
    }
    save_json(output_dir / "run_report.json", final_report)
    print(f"\n[Done] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
