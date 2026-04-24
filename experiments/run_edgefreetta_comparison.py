from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_edgefreetta import evaluate_loaded as evaluate_edge_loaded
from experiments.evaluate_freetta import evaluate_loaded as evaluate_freetta_loaded
from experiments.evaluate_tda import (
    evaluate_clip_loaded,
    evaluate_loaded as evaluate_tda_loaded,
    load_tda_dataset,
)
from src.paper_configs import DEFAULT_DATASETS, DEFAULT_FREETTA_PARAMS, PAPER_TDA_DEFAULTS
from src.paper_setup import EXPECTED_TEST_SPLIT_SIZES


DEFAULT_EDGE_PARAMS = {
    "dtd": {
        "rank": 8,
        "fusion_alpha": 0.6,
        "learning_rate": 5e-3,
        "beta": 4.5,
        "min_confidence": 0.70,
        "align_weight": 0.5,
        "residual_weight": 0.08,
        "weight_decay": 1e-4,
    },
    "caltech": {
        "rank": 8,
        "fusion_alpha": 0.5,
        "learning_rate": 8e-3,
        "beta": 4.5,
        "min_confidence": 0.70,
        "align_weight": 0.5,
        "residual_weight": 0.05,
        "weight_decay": 1e-4,
    },
    "eurosat": {
        "rank": 8,
        "fusion_alpha": 0.7,
        "learning_rate": 1e-2,
        "beta": 4.0,
        "min_confidence": 0.65,
        "align_weight": 0.7,
        "residual_weight": 0.05,
        "weight_decay": 1e-4,
    },
    "pets": {
        "rank": 8,
        "fusion_alpha": 0.55,
        "learning_rate": 8e-3,
        "beta": 4.5,
        "min_confidence": 0.70,
        "align_weight": 0.5,
        "residual_weight": 0.05,
        "weight_decay": 1e-4,
    },
    "imagenet": {
        "rank": 8,
        "fusion_alpha": 0.45,
        "learning_rate": 5e-3,
        "beta": 5.0,
        "min_confidence": 0.75,
        "align_weight": 0.4,
        "residual_weight": 0.08,
        "weight_decay": 1e-4,
    },
}


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_payload_size(dataset: str, num_samples: int) -> None:
    expected = EXPECTED_TEST_SPLIT_SIZES.get(dataset)
    if expected is not None and int(num_samples) != int(expected):
        raise ValueError(
            f"Dataset '{dataset}' has {num_samples} samples, expected {expected} from the official split."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CLIP, TDA, FreeTTA, and EdgeFreeTTA on all datasets")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/edgefreetta_comparison")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle-stream", action="store_true")
    parser.add_argument("--stream-seed", type=int, default=1)
    return parser.parse_args()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    table = [headers] + rows
    widths = [max(len(row[idx]) for row in table) for idx in range(len(headers))]

    def format_row(row: list[str]) -> str:
        cells = [row[idx].ljust(widths[idx]) for idx in range(len(row))]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    rows: list[dict] = []

    for dataset in [str(x).lower() for x in args.datasets]:
        payload = load_tda_dataset(
            dataset=dataset,
            device=device,
            max_samples=args.max_samples,
            features_dir=args.features_dir,
        )
        dataset_key = str(payload["dataset"]).lower()
        validate_payload_size(dataset_key, int(payload["num_samples"]))

        clip_acc = evaluate_clip_loaded(payload)
        tda_acc = evaluate_tda_loaded(
            payload=payload,
            device=device,
            shuffle_stream=args.shuffle_stream,
            stream_seed=args.stream_seed,
            **PAPER_TDA_DEFAULTS[dataset_key],
        )
        freetta_acc = evaluate_freetta_loaded(
            payload=payload,
            device=device,
            shuffle_stream=args.shuffle_stream,
            stream_seed=args.stream_seed,
            **DEFAULT_FREETTA_PARAMS[dataset_key],
        )
        edge_acc = evaluate_edge_loaded(
            payload=payload,
            device=device,
            shuffle_stream=args.shuffle_stream,
            stream_seed=args.stream_seed,
            **DEFAULT_EDGE_PARAMS[dataset_key],
        )

        rows.append(
            {
                "dataset": dataset_key,
                "samples": int(payload["num_samples"]),
                "clip_acc": float(clip_acc),
                "tda_acc": float(tda_acc),
                "freetta_acc": float(freetta_acc),
                "edgefreetta_acc": float(edge_acc),
                "tda_minus_clip": float(tda_acc - clip_acc),
                "freetta_minus_clip": float(freetta_acc - clip_acc),
                "edgefreetta_minus_clip": float(edge_acc - clip_acc),
                "edgefreetta_minus_tda": float(edge_acc - tda_acc),
                "edgefreetta_minus_freetta": float(edge_acc - freetta_acc),
            }
        )

    df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    df.to_csv(output_dir / "summary_table.csv", index=False)
    (output_dir / "summary_table.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    report = {
        "device": str(device),
        "datasets": [str(x).lower() for x in args.datasets],
        "max_samples": args.max_samples,
        "shuffle_stream": bool(args.shuffle_stream),
        "stream_seed": int(args.stream_seed),
        "summary_table_csv": str(output_dir / "summary_table.csv"),
        "summary_table_md": str(output_dir / "summary_table.md"),
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
