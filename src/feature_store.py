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
    image_features = np.load(paths.image_features).astype(np.float32)
    text_features = np.load(paths.text_features).astype(np.float32)
    labels = np.load(paths.labels).astype(np.int64)

    return {
        "dataset_key": canonical,
        "image_features": image_features,
        "text_features": text_features,
        "labels": labels,
        "image_path": str(paths.image_features),
        "text_path": str(paths.text_features),
        "labels_path": str(paths.labels),
    }
