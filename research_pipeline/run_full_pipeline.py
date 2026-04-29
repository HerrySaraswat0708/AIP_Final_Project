#!/usr/bin/env python3
"""
Full Research Pipeline: CLIP vs TDA vs FreeTTA
Runs all 13 sections, saves all outputs, generates final report.

Usage:
  cd AIP-Final-project
  python research_pipeline/run_full_pipeline.py [options]

Options:
  --features-dir  PATH  path to .npy feature files      [default: data/processed]
  --output-dir    PATH  where to save all outputs        [default: outputs/research_pipeline]
  --datasets      NAMES  space-separated dataset keys    [default: all 5]
  --device        auto|cpu|cuda                          [default: auto]
  --seed          INT   stream order seed                [default: 42]
  --window        INT   rolling window size              [default: 50]
  --verbose             print per-step progress
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_pipeline.data.loader import load_features, DATASETS
from research_pipeline.analysis.collector import collect
from research_pipeline.analysis.metrics import (
    compute_accuracy_table,
    compute_per_class_accuracy,
    compute_flip_metrics,
    compute_entropy_confidence_stats,
    compute_trajectory,
    compute_break_even,
    compute_freetta_internal,
    compute_tda_internal,
    compute_disagreement,
    compute_failure_buckets,
    export_failure_metadata,
    compute_difficulty_split,
)
from research_pipeline.analysis.novel_metrics import compute_all_novel_metrics
from research_pipeline.analysis.synthesis import generate_full_synthesis

from research_pipeline.plots.change_plots import (
    plot_prediction_change_breakdown,
    plot_correction_efficiency_comparison,
)
from research_pipeline.plots.distribution_plots import (
    plot_entropy_confidence,
    plot_lmm_analysis,
)
from research_pipeline.plots.trajectory_plots import (
    plot_trajectory,
    plot_stability_scores,
)
from research_pipeline.plots.internal_plots import (
    plot_freetta_internals,
    plot_tda_internals,
)
from research_pipeline.plots.pca_plots import plot_pca_logits
from research_pipeline.plots.improvement_plots import (
    plot_improvement_comparison,
    plot_skip_rate_analysis,
)
from research_pipeline.plots.summary_plots import (
    plot_accuracy_summary,
    plot_difficulty_split,
    plot_disagreement_summary,
    plot_failure_bucket_summary,
)
from research_pipeline.report.writer import write_dataset_report, write_final_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full CLIP / TDA / FreeTTA research pipeline")
    p.add_argument("--features-dir", default="data/processed")
    p.add_argument("--output-dir",   default="outputs/research_pipeline")
    p.add_argument("--datasets",     nargs="*", default=list(DATASETS))
    p.add_argument("--device",       choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--window",       type=int, default=50)
    p.add_argument("--verbose",      action="store_true")
    return p.parse_args()


def resolve_device(req: str) -> str:
    if req == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available, falling back to CPU.")
            return "cpu"
        return "cuda"
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.features_dir

    datasets = [str(d).lower() for d in args.datasets]
    print(f"[Pipeline] device={device}  datasets={datasets}  seed={args.seed}", flush=True)

    all_frames: list[pd.DataFrame] = []
    all_prob_arrs: dict[str, np.ndarray] = {}  # method -> (N_total, C) after concat
    per_ds_arrs: dict[str, dict[str, np.ndarray]] = {}

    t_start = time.time()

    # ── Section 1: Data Collection ────────────────────────────────────────────
    print("\n[Step 1/4] Collecting per-sample data...", flush=True)
    for ds in datasets:
        print(f"  Loading {ds}...", flush=True)
        try:
            data = load_features(ds, features_dir=features_dir, device=device, seed=args.seed)
        except FileNotFoundError as exc:
            print(f"  [Skip] {exc}", flush=True)
            continue

        print(f"  Running methods on {ds} ({data['num_samples']} samples, {data['num_classes']} classes)...", flush=True)
        df, logit_arrs = collect(data, verbose=args.verbose)
        all_frames.append(df)
        per_ds_arrs[ds] = logit_arrs

        # Save per-dataset logit arrays
        ds_out = output_dir / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ds_out / "prob_arrays.npz",
            **{k: v for k, v in logit_arrs.items() if isinstance(v, np.ndarray)}
        )
        df.to_csv(ds_out / "per_sample_metrics.csv", index=False)

    if not all_frames:
        print("[Error] No datasets loaded. Check --features-dir.")
        sys.exit(1)

    merged = pd.concat(all_frames, ignore_index=True)
    merged.to_csv(output_dir / "per_sample_metrics_all.csv", index=False)

    # ── Sections 2–9: Metric Computation ──────────────────────────────────────
    print("\n[Step 2/4] Computing metrics...", flush=True)
    acc_df        = compute_accuracy_table(merged)
    flip_df       = compute_flip_metrics(merged)
    entropy_df    = compute_entropy_confidence_stats(merged)
    traj          = compute_trajectory(merged, window=args.window)
    be_df         = compute_break_even(traj)
    freetta_int   = compute_freetta_internal(merged)
    tda_int       = compute_tda_internal(merged)
    disagree_df   = compute_disagreement(merged)
    fail_df       = compute_failure_buckets(merged)
    diff_df       = compute_difficulty_split(merged)

    # Save CSVs
    acc_df.to_csv(output_dir / "accuracy_table.csv", index=False)
    flip_df.to_csv(output_dir / "flip_metrics.csv", index=False)
    entropy_df.to_csv(output_dir / "entropy_confidence.csv", index=False)
    traj.to_csv(output_dir / "trajectory.csv", index=False)
    be_df.to_csv(output_dir / "break_even.csv", index=False)
    freetta_int.to_csv(output_dir / "freetta_internal.csv", index=False)
    tda_int.to_csv(output_dir / "tda_internal.csv", index=False)
    disagree_df.to_csv(output_dir / "disagreement.csv", index=False)
    fail_df.to_csv(output_dir / "failure_buckets.csv", index=False)
    diff_df.to_csv(output_dir / "difficulty_split.csv", index=False)
    export_failure_metadata(merged, output_dir)

    # ── Section 10: Novel Metrics ─────────────────────────────────────────────
    novel_metrics = compute_all_novel_metrics(merged, flip_df, traj, per_ds_arrs)
    for name, df_nm in novel_metrics.items():
        df_nm.to_csv(output_dir / f"novel_{name}.csv", index=False)

    # ── Plotting ──────────────────────────────────────────────────────────────
    print("\n[Step 3/4] Generating plots...", flush=True)
    plot_prediction_change_breakdown(merged, flip_df, output_dir)

    if "dataset" in novel_metrics.get("correction_efficiency", pd.DataFrame()).columns:
        plot_correction_efficiency_comparison(novel_metrics["correction_efficiency"], output_dir)

    plot_entropy_confidence(merged, output_dir)

    # Per-dataset LMM (each dataset has its own class count)
    for ds in datasets:
        if ds not in per_ds_arrs:
            continue
        ds_df = merged[merged["dataset"] == ds].reset_index(drop=True)
        plot_lmm_analysis(ds_df, per_ds_arrs[ds], output_dir)
        plot_pca_logits(ds_df, per_ds_arrs[ds], output_dir)

    plot_trajectory(traj, output_dir)
    plot_stability_scores(novel_metrics.get("stability_score", pd.DataFrame()), output_dir)

    for ds in datasets:
        if ds not in per_ds_arrs:
            continue
        ds_df = merged[merged["dataset"] == ds].reset_index(drop=True)
        plot_freetta_internals(ds_df, per_ds_arrs[ds], output_dir)
    plot_tda_internals(merged, output_dir)

    plot_improvement_comparison(acc_df, output_dir)
    plot_skip_rate_analysis(merged, output_dir)

    plot_accuracy_summary(acc_df, output_dir)
    plot_difficulty_split(diff_df, output_dir)
    plot_disagreement_summary(disagree_df, output_dir)
    plot_failure_bucket_summary(fail_df, output_dir)

    # ── Sections 11 + 13: Synthesis & Final Report ────────────────────────────
    print("\n[Step 4/4] Generating reports...", flush=True)
    synthesis_text = generate_full_synthesis(
        acc_df=acc_df,
        flip_df=flip_df,
        entropy_df=entropy_df,
        disagree_df=disagree_df,
        novel_metrics=novel_metrics,
    )
    (output_dir / "synthesis.md").write_text(synthesis_text, encoding="utf-8")

    # Per-dataset markdown reports
    for ds in datasets:
        ds_data = merged[merged["dataset"] == ds]
        if ds_data.empty:
            continue
        acc_row = acc_df[acc_df["dataset"] == ds].iloc[0]
        flip_sub = flip_df[flip_df["dataset"] == ds]
        ent_sub = entropy_df[entropy_df["dataset"] == ds]
        dis_row = disagree_df[disagree_df["dataset"] == ds].iloc[0] if (disagree_df["dataset"] == ds).any() else pd.Series()
        tda_row = tda_int[tda_int["dataset"] == ds].iloc[0] if (tda_int["dataset"] == ds).any() else pd.Series()
        ftta_row = freetta_int[freetta_int["dataset"] == ds].iloc[0] if (freetta_int["dataset"] == ds).any() else pd.Series()
        # Filter novel metrics to this dataset where possible
        ds_novel = {}
        for k, v in novel_metrics.items():
            if "dataset" in v.columns:
                ds_novel[k] = v[v["dataset"] == ds]
            else:
                ds_novel[k] = v
        write_dataset_report(
            dataset=ds,
            output_dir=output_dir,
            acc_row=acc_row,
            flip_sub=flip_sub,
            entropy_sub=ent_sub,
            disagree_row=dis_row,
            tda_internal=tda_row,
            freetta_internal=ftta_row,
            novel_sub=ds_novel,
        )

    elapsed = time.time() - t_start
    run_meta = {
        "datasets": datasets,
        "device": device,
        "seed": args.seed,
        "rolling_window": args.window,
        "total_samples": int(len(merged)),
        "elapsed_seconds": round(elapsed, 1),
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    report_path = write_final_report(
        output_dir=output_dir,
        acc_df=acc_df,
        flip_df=flip_df,
        entropy_df=entropy_df,
        disagree_df=disagree_df,
        tda_int_df=tda_int,
        freetta_int_df=freetta_int,
        novel_metrics=novel_metrics,
        fail_df=fail_df,
        synthesis_text=synthesis_text,
        run_meta=run_meta,
    )

    print(f"\n[Done] Elapsed: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"Final report:     {report_path}")
    print("\nAccuracy Summary:")
    print(acc_df.to_string(index=False))


if __name__ == "__main__":
    main()
