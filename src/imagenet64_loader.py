from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.imagenet_loader import _parse_reference_classnames


class ImageNet64Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx], mode="RGB")
        label = int(self.labels[idx])
        return self.transform(image), label


def _resolve_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)

    candidates = [
        Path("data/raw/Imagenet64_val"),
        Path("data/raw/imagenet64_val"),
        Path("data/raw/imagenet_64eval"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_val_data_path(root: Path) -> Path:
    candidates = [
        root / "val_data",
        root / "val_data.npz",
        root / "val_data.pkl",
        root / "val_data.pickle",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    npz_files = sorted(root.glob("*.npz"))
    if len(npz_files) == 1:
        return npz_files[0]
    raise FileNotFoundError(f"Could not find an ImageNet64 validation dump under {root}")


def _resolve_classnames_path(root: Path) -> Path | None:
    candidates = [
        root / "map_clsloc.txt",
        root / "imagenet-labels.txt",
        root / "classnames.txt",
        root.parent / "map_clsloc.txt",
        root.parent / "imagenet-labels.txt",
        root.parent / "classnames.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_map_clsloc_classnames(path: Path) -> list[str]:
    ordered: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        _, index_raw, class_name = parts
        try:
            index = int(index_raw)
        except ValueError:
            continue
        ordered[index] = class_name.strip().lower().replace("_", " ")

    expected = list(range(1, 1001))
    if sorted(ordered.keys()) != expected:
        raise ValueError(f"Expected 1000 ImageNet class entries in {path}, found {len(ordered)}")
    return [ordered[index] for index in expected]


def _decode_flattened_images(flat_data: np.ndarray) -> np.ndarray:
    if flat_data.ndim != 2 or flat_data.shape[1] != 64 * 64 * 3:
        raise ValueError(f"Unexpected ImageNet64 data shape: {flat_data.shape}")
    images = flat_data.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
    return images.astype(np.uint8, copy=False)


def _load_pickle_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")

    if not isinstance(payload, dict) or "data" not in payload or "labels" not in payload:
        raise ValueError(f"Unexpected ImageNet64 payload format in {path}")

    images = _decode_flattened_images(np.asarray(payload["data"]))
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if labels.min() == 1 and labels.max() == 1000:
        labels = labels - 1
    return images, labels


def _load_npz_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        if "data" not in payload or "labels" not in payload:
            raise ValueError(f"Unexpected ImageNet64 NPZ payload format in {path}")
        images = _decode_flattened_images(np.asarray(payload["data"]))
        labels = np.asarray(payload["labels"], dtype=np.int64)

    if labels.min() == 1 and labels.max() == 1000:
        labels = labels - 1
    return images, labels


def _load_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        return _load_npz_payload(path)
    return _load_pickle_payload(path)


def load_imagenet64(
    transform,
    root: str | Path | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
):
    root_path = _resolve_root(root)
    payload_path = _resolve_val_data_path(root_path)
    images, labels = _load_payload(payload_path)

    classnames_path = _resolve_classnames_path(root_path)
    if classnames_path is not None:
        class_names = _load_map_clsloc_classnames(classnames_path)
    else:
        class_names = _parse_reference_classnames()
        if class_names is None or len(class_names) != 1000:
            raise RuntimeError(
                "Could not resolve ImageNet64 class names. "
                "Place a `map_clsloc.txt`, `imagenet-labels.txt`, or `classnames.txt` file next to the data."
            )

    dataset = ImageNet64Dataset(images=images, labels=labels, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, class_names
