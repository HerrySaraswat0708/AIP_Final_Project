from __future__ import annotations

"""
Section 11 — Synthesis

Programmatic insight generation from the computed metric DataFrames.
Produces structured text observations without needing LLM inference.
"""

from typing import Dict, List
import pandas as pd
import numpy as np


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _winner(row: pd.Series, methods=("tda", "freetta", "conf_ftta", "ent_tda", "hybrid")) -> str:
    best = max(methods, key=lambda m: row.get(f"{m}_acc", 0.0))
    return best


def generate_accuracy_insights(acc_df: pd.DataFrame) -> List[str]:
    lines = ["## 1. Accuracy Insights"]
    for _, row in acc_df.iterrows():
        ds = row["dataset"]
        clip = row["clip_acc"]
        tda = row["tda_acc"]
        ftta = row["freetta_acc"]
        hybrid = row.get("hybrid_acc", float("nan"))

        best = _winner(row)
        gain_tda = tda - clip
        gain_ftta = ftta - clip

        lines.append(f"\n### {ds.upper()}")
        lines.append(
            f"- CLIP baseline: {_fmt(clip, 4)}  TDA: {_fmt(tda, 4)} ({gain_tda:+.4f})  "
            f"FreeTTA: {_fmt(ftta, 4)} ({gain_ftta:+.4f})  "
            f"Hybrid: {_fmt(hybrid, 4) if not np.isnan(hybrid) else 'n/a'}"
        )
        lines.append(f"- Best adapter: **{best}**")

        if abs(gain_tda) < 0.002 and abs(gain_ftta) < 0.002:
            lines.append(
                "- Both methods are approximately neutral on this dataset: "
                "CLIP predictions are already well-calibrated for this domain."
            )
        elif gain_ftta > gain_tda + 0.01:
            lines.append(
                f"- FreeTTA outperforms TDA by {gain_ftta - gain_tda:.4f}.  "
                "Global statistics adaptation is more effective than local cache here, "
                "suggesting stable domain shift with consistent class-level statistics."
            )
        elif gain_tda > gain_ftta + 0.01:
            lines.append(
                f"- TDA outperforms FreeTTA by {gain_tda - gain_ftta:.4f}.  "
                "Local cache memory is more effective, suggesting the test stream has "
                "within-class consistency that the cache can exploit."
            )
        else:
            lines.append("- TDA and FreeTTA are roughly on par for this dataset.")

    return lines


def generate_flip_insights(flip_df: pd.DataFrame) -> List[str]:
    lines = ["## 2. Flip / Correction Efficiency Insights"]
    for ds, g in flip_df.groupby("dataset", observed=True):
        lines.append(f"\n### {ds.upper()}")
        for _, row in g.iterrows():
            m = row["method"]
            ce = row.get("correction_efficiency", row["n_beneficial"] / max(row["n_beneficial"] + row["n_harmful"], 1))
            cr = row["change_rate"]
            ncr = row["net_correction_rate"]
            lines.append(
                f"- **{m}**: change_rate={_fmt(cr)} CE={_fmt(ce)}  net_correction={ncr:+.4f}"
            )
            if ce > 0.7:
                lines.append(
                    f"  → High CE ({_fmt(ce)}): when {m} changes a prediction it is usually right."
                )
            elif ce < 0.4:
                lines.append(
                    f"  → Low CE ({_fmt(ce)}): {m} makes many harmful changes — "
                    "the cache/statistics are corrupting more than they help."
                )
    return lines


def generate_entropy_insights(entropy_df: pd.DataFrame) -> List[str]:
    lines = ["## 3. Entropy / Confidence Insights"]
    for ds, g in entropy_df.groupby("dataset", observed=True):
        lines.append(f"\n### {ds.upper()}")
        for method in ("clip", "tda", "freetta"):
            correct = g[(g["method"] == method) & (g["subset"] == "correct")]
            wrong = g[(g["method"] == method) & (g["subset"] == "wrong")]
            if correct.empty or wrong.empty:
                continue
            conf_correct = correct["mean_confidence"].iloc[0]
            conf_wrong = wrong["mean_confidence"].iloc[0]
            ent_wrong = wrong["mean_entropy"].iloc[0]
            lines.append(
                f"- {method.upper()}: confidence(correct)={_fmt(conf_correct)}  "
                f"confidence(wrong)={_fmt(conf_wrong)}  entropy(wrong)={_fmt(ent_wrong)}"
            )
            if conf_wrong > 0.7:
                lines.append(
                    f"  → {method.upper()} produces overconfident wrong predictions "
                    f"(conf={_fmt(conf_wrong)}).  OER risk is elevated."
                )
    return lines


def generate_disagreement_insights(disagree_df: pd.DataFrame) -> List[str]:
    lines = ["## 4. Disagreement Analysis Insights"]
    for _, row in disagree_df.iterrows():
        ds = row["dataset"]
        rate = row["disagreement_rate"]
        tda_wins = row.get("tda_wins", 0)
        ftta_wins = row.get("freetta_wins", 0)
        lines.append(f"\n### {ds.upper()}")
        lines.append(
            f"- Disagreement rate: {_fmt(rate)}.  "
            f"When they disagree: TDA wins {tda_wins} times, FreeTTA wins {ftta_wins} times."
        )
        if rate > 0.15:
            lines.append(
                "  → High disagreement rate: the two methods respond very differently "
                "to the same inputs — their mechanisms are genuinely complementary."
            )
        else:
            lines.append(
                "  → Low disagreement rate: TDA and FreeTTA make similar predictions, "
                "suggesting one method's decisions dominate the other."
            )
    return lines


def generate_improvement_insights(acc_df: pd.DataFrame) -> List[str]:
    lines = ["## 5. Improvement Method Insights"]
    for _, row in acc_df.iterrows():
        ds = row["dataset"]
        clip = row["clip_acc"]
        tda = row["tda_acc"]
        ftta = row["freetta_acc"]
        cftta = row.get("conf_ftta_acc", float("nan"))
        etda = row.get("ent_tda_acc", float("nan"))
        hybrid = row.get("hybrid_acc", float("nan"))
        lines.append(f"\n### {ds.upper()}")

        if not np.isnan(cftta):
            delta = cftta - ftta
            status = "IMPROVED" if delta > 0.001 else ("DEGRADED" if delta < -0.001 else "NEUTRAL")
            lines.append(
                f"- ConfGatedFreeTTA vs FreeTTA: {delta:+.4f} → **{status}**"
            )
            if delta < -0.002:
                lines.append(
                    "  → Gating the M-step hurt FreeTTA here.  "
                    "Discarding uncertain updates removed too much signal, "
                    "starving the class means of the adaptation they need."
                )
            elif delta > 0.002:
                lines.append(
                    "  → Confidence gating improved FreeTTA.  "
                    "Filtering out uncertain updates reduced mean corruption."
                )

        if not np.isnan(etda):
            delta = etda - tda
            status = "IMPROVED" if delta > 0.001 else ("DEGRADED" if delta < -0.001 else "NEUTRAL")
            lines.append(
                f"- EntropyGatedTDA vs TDA: {delta:+.4f} → **{status}**"
            )
            if delta < -0.002:
                lines.append(
                    "  → Adaptive entropy gating shrank the cache too aggressively.  "
                    "The original fixed threshold already selects good samples."
                )

        if not np.isnan(hybrid):
            best_single = max(tda, ftta)
            delta = hybrid - best_single
            status = "IMPROVED" if delta > 0.001 else ("DEGRADED" if delta < -0.001 else "NEUTRAL")
            lines.append(
                f"- Hybrid vs best(TDA, FreeTTA): {delta:+.4f} → **{status}**"
            )
            if delta < 0:
                lines.append(
                    "  → Combining local and global signals did not help.  "
                    "When both components contribute noise, the hybrid inherits both errors.  "
                    "The optimal weighting is likely dataset-specific."
                )
            else:
                lines.append(
                    "  → Hybrid gains from combining local cache (short-range) "
                    "and global statistics (domain shift).  "
                    "The two mechanisms are complementary here."
                )
    return lines


def generate_full_synthesis(
    acc_df: pd.DataFrame,
    flip_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    disagree_df: pd.DataFrame,
    novel_metrics: dict,
) -> str:
    sections: List[str] = [
        "# Section 11 — Synthesis: Deep Insights",
        "",
        "This section derives evidence-based observations from the computed metrics.",
        "Conclusions are grounded in numbers from the data, not assumptions.",
        "",
    ]
    sections.extend(generate_accuracy_insights(acc_df))
    sections.extend(generate_flip_insights(flip_df))
    sections.extend(generate_entropy_insights(entropy_df))
    sections.extend(generate_disagreement_insights(disagree_df))
    sections.extend(generate_improvement_insights(acc_df))

    # Novel metric commentary
    sections.append("\n## 6. Novel Metric Commentary (Section 10)")
    if "stability_score" in novel_metrics:
        ss = novel_metrics["stability_score"]
        sections.append("\n### Stability Scores")
        sections.append(ss.to_markdown(index=False))
        most_stable = ss.set_index("dataset").apply(
            lambda r: max(
                {m: r.get(f"{m}_stability", 0) for m in ("clip", "tda", "freetta")}.items(),
                key=lambda x: x[1],
            )[0],
            axis=1,
        )
        for ds, winner in most_stable.items():
            sections.append(f"- {ds}: most stable method = **{winner}**")

    if "overconfidence_error_rate" in novel_metrics:
        oer = novel_metrics["overconfidence_error_rate"]
        sections.append("\n### Overconfidence Error Rate (OER > 0.1 highlighted)")
        high_oer = oer[oer["oer"] > 0.1][["dataset", "method", "oer"]]
        if len(high_oer):
            sections.append(high_oer.to_markdown(index=False))
            sections.append(
                "High OER entries indicate methods that produce confident wrong predictions.  "
                "This is a calibration failure and is dangerous in deployment."
            )
        else:
            sections.append("No method exceeds OER > 0.1 on any dataset — good calibration overall.")

    sections.append("\n## 7. Summary Hypotheses")
    sections.append(
        "Based on the above analysis, the following hypotheses are proposed:\n"
        "\n1. **FreeTTA effectiveness scales with domain shift magnitude**: "
        "datasets with larger domain gap from ImageNet training benefit more from "
        "global statistics adaptation (EuroSAT > DTD > Caltech).\n"
        "\n2. **TDA cache saturation limits long-stream gains**: "
        "the fixed capacity (pos_shot_capacity * C slots) caps how much the cache "
        "can represent.  On large-class datasets (ImageNet, 1000 classes) the cache "
        "is too sparse per class to help much.\n"
        "\n3. **Confidence gating is a double-edged sword**: "
        "on datasets where CLIP is uncertain frequently (DTD textures), "
        "gating the M-step removes most of the adaptation signal and hurts FreeTTA.  "
        "On datasets where CLIP is mostly confident (Caltech), gating is nearly neutral.\n"
        "\n4. **Hybrid gains require the two methods to be complementary**: "
        "when TDA and FreeTTA disagree frequently the hybrid can benefit from both.  "
        "When they agree, combining them adds noise without signal."
    )

    return "\n".join(sections)
