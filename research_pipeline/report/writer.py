from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=float_fmt)


def _section(title: str, content: str) -> str:
    return f"\n## {title}\n\n{content}\n"


def write_dataset_report(
    dataset: str,
    output_dir: Path,
    acc_row: pd.Series,
    flip_sub: pd.DataFrame,
    entropy_sub: pd.DataFrame,
    disagree_row: pd.Series,
    tda_internal: pd.Series,
    freetta_internal: pd.Series,
    novel_sub: dict[str, pd.DataFrame],
) -> None:
    lines = [f"# Dataset Report: {dataset.upper()}", ""]

    # Accuracy
    acc_cols = [c for c in acc_row.index if c.endswith("_acc")]
    acc_data = {c.replace("_acc", ""): f"{acc_row[c]:.4f}" for c in acc_cols}
    lines.append("## Accuracy")
    lines.append(pd.DataFrame([acc_data]).to_markdown(index=False))
    lines.append("")

    gain_cols = [c for c in acc_row.index if c.endswith("_gain")]
    if gain_cols:
        gain_data = {c.replace("_gain", ""): f"{acc_row[c]:+.4f}" for c in gain_cols}
        lines.append("### Gains over CLIP")
        lines.append(pd.DataFrame([gain_data]).to_markdown(index=False))
        lines.append("")

    # Flip analysis
    lines.append("## Prediction Change Metrics (Section 3)")
    lines.append(_md_table(flip_sub))
    lines.append("")

    # Entropy
    lines.append("## Entropy / Confidence (Section 4)")
    pivot = entropy_sub.pivot_table(
        index="method", columns="subset",
        values=["mean_entropy", "mean_confidence"],
        aggfunc="first",
    )
    lines.append(pivot.to_markdown())
    lines.append("")

    # Disagreement
    lines.append("## Disagreement Analysis (Section 7)")
    lines.append(pd.DataFrame([disagree_row]).to_markdown(index=False))
    lines.append("")

    # TDA internal
    lines.append("## TDA Internal (Section 6)")
    lines.append(pd.DataFrame([tda_internal]).to_markdown(index=False))
    lines.append("")

    # FreeTTA internal
    lines.append("## FreeTTA Internal (Section 6)")
    lines.append(pd.DataFrame([freetta_internal]).to_markdown(index=False))
    lines.append("")

    # Novel metrics
    lines.append("## Novel Metrics (Section 10)")
    if "correction_efficiency" in novel_sub:
        ce = novel_sub["correction_efficiency"]
        if "dataset" in ce.columns:
            ce = ce[ce["dataset"] == dataset]
        lines.append("### Correction Efficiency")
        lines.append(_md_table(ce[["method", "n_beneficial", "n_harmful", "correction_efficiency"]]))
        lines.append("")
    if "overconfidence_error_rate" in novel_sub:
        oer = novel_sub["overconfidence_error_rate"]
        if "dataset" in oer.columns:
            oer = oer[oer["dataset"] == dataset]
        lines.append("### Overconfidence Error Rate")
        lines.append(_md_table(oer[["method", "n_wrong", "oer"]]))
        lines.append("")

    lines.append("## Generated Plots")
    for png in sorted((output_dir / dataset).glob("*.png")) if (output_dir / dataset).exists() else []:
        lines.append(f"- `{png.name}`")

    (output_dir / dataset / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    output_dir: Path,
    acc_df: pd.DataFrame,
    flip_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    disagree_df: pd.DataFrame,
    tda_int_df: pd.DataFrame,
    freetta_int_df: pd.DataFrame,
    novel_metrics: dict[str, pd.DataFrame],
    fail_df: pd.DataFrame,
    synthesis_text: str,
    run_meta: dict,
) -> Path:
    """Write the structured final report as a single Markdown file."""

    lines = [
        "# Final Research Report",
        "## CLIP vs TDA vs FreeTTA: Deep Comparative Analysis",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
        "## Table of Contents",
        "1. [Executive Summary](#executive-summary)",
        "2. [Section 2: Baseline Accuracy](#section-2-baseline-accuracy)",
        "3. [Section 3: Prediction Change Analysis](#section-3-prediction-change-analysis)",
        "4. [Section 4: Entropy & Confidence](#section-4-entropy--confidence)",
        "5. [Section 7: Disagreement Analysis](#section-7-disagreement-analysis)",
        "6. [Section 8: Failure Buckets](#section-8-failure-buckets)",
        "7. [Section 10: Novel Metrics](#section-10-novel-metrics)",
        "8. [Section 12: Improvement Attempts](#section-12-improvement-attempts)",
        "9. [Section 11: Synthesis & Insights](#section-11-synthesis--insights)",
        "",
        "---",
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines.append("\n## Executive Summary\n")
    best_per_ds = acc_df.copy()
    adapt_acc_cols = [c for c in acc_df.columns if c.endswith("_acc") and c != "clip_acc"]
    if adapt_acc_cols:
        best_per_ds["best_adapter"] = acc_df[adapt_acc_cols].idxmax(axis=1).str.replace("_acc", "")
        best_per_ds["best_adapter_acc"] = acc_df[adapt_acc_cols].max(axis=1)
        best_per_ds["best_gain"] = best_per_ds["best_adapter_acc"] - acc_df["clip_acc"]
    lines.append(_md_table(acc_df))
    lines.append("")

    n_ds_tda_wins = int((acc_df.get("tda_acc", pd.Series(dtype=float)) > acc_df["clip_acc"]).sum())
    n_ds_ftta_wins = int((acc_df.get("freetta_acc", pd.Series(dtype=float)) > acc_df["clip_acc"]).sum())
    lines.append(
        f"- TDA outperforms CLIP on **{n_ds_tda_wins}/{len(acc_df)}** datasets.\n"
        f"- FreeTTA outperforms CLIP on **{n_ds_ftta_wins}/{len(acc_df)}** datasets.\n"
    )

    # ── Section 2 ─────────────────────────────────────────────────────────────
    lines.append("\n## Section 2: Baseline Accuracy\n")
    lines.append(_md_table(acc_df))

    # ── Section 3 ─────────────────────────────────────────────────────────────
    lines.append("\n## Section 3: Prediction Change Analysis\n")
    core_flip = flip_df[flip_df["method"].isin(("tda", "freetta"))]
    lines.append(_md_table(core_flip[[
        "dataset", "method", "change_rate", "beneficial_rate", "harmful_rate",
        "net_correction_rate", "correction_efficiency"
    ]]))
    lines.append(
        "\n**Correction Efficiency (CE) = beneficial_flips / total_changes.**  "
        "CE > 0.5 means more than half of all changes are beneficial."
    )

    # ── Section 4 ─────────────────────────────────────────────────────────────
    lines.append("\n## Section 4: Entropy & Confidence\n")
    ent_pivot = entropy_df[
        (entropy_df["method"].isin(("clip", "tda", "freetta")))
        & (entropy_df["subset"].isin(("correct", "wrong")))
    ].pivot_table(
        index=["dataset", "method"],
        columns="subset",
        values="mean_confidence",
        aggfunc="first",
    ).round(4)
    lines.append("### Mean Confidence (correct vs wrong)\n")
    lines.append(ent_pivot.to_markdown())

    # ── Section 7 ─────────────────────────────────────────────────────────────
    lines.append("\n## Section 7: Disagreement Analysis\n")
    lines.append(_md_table(disagree_df))

    # ── Section 8 ─────────────────────────────────────────────────────────────
    lines.append("\n## Section 8: Failure Buckets\n")
    pivot8 = fail_df.pivot_table(index="dataset", columns="bucket", values="rate").round(4)
    lines.append(pivot8.to_markdown())

    # ── Section 10 ────────────────────────────────────────────────────────────
    lines.append("\n## Section 10: Novel Metrics\n")

    if "correction_efficiency" in novel_metrics:
        ce = novel_metrics["correction_efficiency"]
        lines.append("### Correction Efficiency (CE)\n")
        lines.append(_md_table(ce))
        lines.append("")

    if "overconfidence_error_rate" in novel_metrics:
        oer = novel_metrics["overconfidence_error_rate"]
        lines.append("### Overconfidence Error Rate (OER)\n")
        lines.append(_md_table(oer[["dataset", "method", "n_wrong", "oer"]]) if "dataset" in oer.columns else _md_table(oer))
        lines.append("")

    if "logit_movement_magnitude" in novel_metrics:
        lmm = novel_metrics["logit_movement_magnitude"]
        lines.append("### Logit Movement Magnitude (LMM)\n")
        lines.append(_md_table(lmm))
        lines.append(
            "\n*LMM = ||prob_method - prob_clip||₂.  "
            "Beneficial flips should have moderate LMM; very large LMM with harmful flips "
            "suggests the method is moving predictions aggressively in the wrong direction.*"
        )
        lines.append("")

    if "stability_score" in novel_metrics:
        ss = novel_metrics["stability_score"]
        lines.append("### Stability Score (SS)\n")
        lines.append(_md_table(ss))
        lines.append(
            "\n*SS = 1 / (1 + σ(rolling_accuracy)).  "
            "Values close to 1.0 indicate a smooth, stable adaptation trajectory.*"
        )
        lines.append("")

    # ── Section 12 ────────────────────────────────────────────────────────────
    lines.append("\n## Section 12: Improvement Attempts\n")

    improvement_cols = ["dataset", "clip_acc", "tda_acc", "freetta_acc",
                        "conf_ftta_acc", "ent_tda_acc", "hybrid_acc",
                        "conf_ftta_gain", "ent_tda_gain", "hybrid_gain"]
    imp_cols = [c for c in improvement_cols if c in acc_df.columns]
    lines.append(_md_table(acc_df[imp_cols]))

    lines.append("""
### Method 1: Confidence-Gated FreeTTA (ConfGatedFreeTTA)
- **Idea**: Skip the M-step (class mean update) when max CLIP confidence < threshold.
- **Rationale**: Prevents uncertain pseudo-labels from corrupting class means.
- **Expected win case**: Datasets with many uncertain CLIP predictions (DTD, EuroSAT).
- **Expected fail case**: Datasets where CLIP is mostly confident — gating starves adaptation.

### Method 2: Entropy-Gated TDA (EntropyGatedTDA)
- **Idea**: Adaptively tighten the entropy threshold for cache insertion using the running p-th percentile.
- **Rationale**: The original fixed threshold may be too permissive — only the most confident samples should enter the positive cache.
- **Expected win case**: Noisy datasets where standard TDA fills cache with borderline samples.
- **Expected fail case**: Any dataset where the cache needs volume to find similar samples.

### Method 3: Hybrid (TDA + FreeTTA)
- **Idea**: Combine local cache adjustment (TDA) + global mean adjustment (FreeTTA) via weighted sum.
- **Rationale**: Local memory captures short-range stream consistency; global stats capture domain shift.
- **Expected win case**: High disagreement between TDA and FreeTTA = complementary signals.
- **Expected fail case**: When both methods are partially wrong, hybrid inherits both errors.
""")

    # ── Section 11: Synthesis ─────────────────────────────────────────────────
    lines.append("\n## Section 11: Synthesis & Deep Insights\n")
    lines.append(synthesis_text)

    # ── Plots inventory ───────────────────────────────────────────────────────
    lines.append("\n## Generated Plots\n")
    all_pngs = sorted(output_dir.rglob("*.png"))
    for png in all_pngs:
        rel = png.relative_to(output_dir)
        lines.append(f"- `{rel}`")

    # ── Run metadata ──────────────────────────────────────────────────────────
    lines.append("\n## Run Metadata\n")
    lines.append(f"```json\n{json.dumps(run_meta, indent=2, default=str)}\n```")

    report_path = output_dir / "FINAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
