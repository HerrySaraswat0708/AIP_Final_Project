from pathlib import Path
from typing import Dict, List

import numpy as np


class DatasetFeaturePaths(object):
    def __init__(self, dataset_key, image_features, text_features, labels):
        self.dataset_key = dataset_key
        self.image_features = image_features
        self.text_features = text_features
        self.labels = labels


DATASET_ALIASES: Dict[str, str] = {
    "dtd": "dtd",
    "caltech": "caltech",
    "caltech101": "caltech",
    "eurosat": "eurosat",
    "pets": "pets",
    "pet": "pets",
    "oxford_pets": "pets",
    "oxfordpets": "pets",
    "imagenet": "imagenet",
    "imagenet64": "imagenet64",
    "imagenet64_val": "imagenet64",
    "imagenet_64": "imagenet64",
    "imagenet_64eval": "imagenet64",
}


def _canonical_name(dataset: str) -> str:
    normalized = dataset.strip().lower().replace("-", "_")
    return DATASET_ALIASES.get(normalized, normalized)


def index_feature_files(features_dir: Path) -> Dict[str, DatasetFeaturePaths]:
    records: Dict[str, Dict[str, Path]] = {}
    suffixes = {
        "_image_features.npy": "image_features",
        "_text_features.npy": "text_features",
        "_labels.npy": "labels",
    }

    for path in sorted(features_dir.glob("*.npy")):
        lower_name = path.name.lower()
        for suffix, key in suffixes.items():
            if lower_name.endswith(suffix):
                prefix = lower_name[: -len(suffix)]
                canonical = _canonical_name(prefix)
                records.setdefault(canonical, {})[key] = path
                break

    indexed: Dict[str, DatasetFeaturePaths] = {}
    for dataset_key, item in records.items():
        if {"image_features", "text_features", "labels"} <= set(item.keys()):
            indexed[dataset_key] = DatasetFeaturePaths(
                dataset_key=dataset_key,
                image_features=item["image_features"],
                text_features=item["text_features"],
                labels=item["labels"],
            )
    return indexed


def list_available_datasets(features_dir: Path) -> List[str]:
    indexed = index_feature_files(features_dir)
    return sorted(indexed.keys())


def _safe_load_npy(path: Path, kind: str, dataset: str) -> np.ndarray:
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to load {kind} for dataset '{dataset}' from {path}. "
            "The .npy file appears to be truncated, corrupted, or only partially copied. "
            "Please regenerate that dataset's features or recopy the file and try again."
        ) from exc


def load_dataset_features(features_dir: Path, dataset: str) -> Dict[str, np.ndarray]:
    indexed = index_feature_files(features_dir)
    canonical = _canonical_name(dataset)
    if canonical not in indexed:
        available = ", ".join(sorted(indexed.keys()))
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in {features_dir}. "
            f"Available datasets: {available or 'none'}"
        )

    paths = indexed[canonical]
    image_features = _safe_load_npy(paths.image_features, "image features", canonical).astype(np.float32)
    text_features = _safe_load_npy(paths.text_features, "text features", canonical).astype(np.float32)
    labels = _safe_load_npy(paths.labels, "labels", canonical).astype(np.int64)

    if image_features.ndim != 2:
        raise ValueError(
            f"Expected 2D image features for dataset '{canonical}', got shape {image_features.shape} "
            f"from {paths.image_features}"
        )
    if text_features.ndim != 2:
        raise ValueError(
            f"Expected 2D text features for dataset '{canonical}', got shape {text_features.shape} "
            f"from {paths.text_features}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"Expected 1D labels for dataset '{canonical}', got shape {labels.shape} "
            f"from {paths.labels}"
        )
    if image_features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatched sample counts for dataset '{canonical}': "
            f"{image_features.shape[0]} image features vs {labels.shape[0]} labels. "
            f"Files: {paths.image_features}, {paths.labels}"
        )
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError(
            f"Mismatched embedding dimensions for dataset '{canonical}': "
            f"{image_features.shape[1]} for image features vs {text_features.shape[1]} for text features. "
            f"Files: {paths.image_features}, {paths.text_features}"
        )

    return {
        "dataset_key": canonical,
        "image_features": image_features,
        "text_features": text_features,
        "labels": labels,
        "image_path": str(paths.image_features),
        "text_path": str(paths.text_features),
        "labels_path": str(paths.labels),
    }
