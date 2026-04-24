from __future__ import annotations

import ast
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGENETV2_DIRNAME = "imagenetv2-matched-frequency-format-val"
IMAGENETV2_ARCHIVE = "imagenetv2-matched-frequency.tar.gz"
IMAGENETV2_URLS = (
    "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz",
    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
)


def _resolve_imagenet_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data/raw/IMAGENET",
        PROJECT_ROOT / "data/raw/imagenet",
        PROJECT_ROOT / "data/raw/IMAGENETV2",
    ]
    for candidate in candidates:
        if (candidate / IMAGENETV2_DIRNAME).exists():
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _download_imagenetv2_archive(root_dir: Path) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    archive_path = root_dir / IMAGENETV2_ARCHIVE
    if archive_path.exists() and _archive_is_valid(archive_path):
        return archive_path
    if archive_path.exists():
        archive_path.unlink()

    last_error: Exception | None = None
    for url in IMAGENETV2_URLS:
        try:
            download_url(url=url, root=str(root_dir), filename=IMAGENETV2_ARCHIVE)
            if not _archive_is_valid(archive_path):
                raise tarfile.ReadError(f"Downloaded file is not a valid tar.gz archive: {archive_path}")
            return archive_path
        except (HTTPError, URLError, OSError) as exc:
            last_error = exc
            if archive_path.exists():
                archive_path.unlink()

    raise RuntimeError(
        "Failed to download ImageNetV2 matched-frequency archive from all configured mirrors."
    ) from last_error


def _archive_is_valid(archive_path: Path) -> bool:
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            return tar.next() is not None
    except (tarfile.TarError, OSError):
        return False


def ensure_imagenetv2(root_dir: Path | None = None) -> Path:
    base_dir = _resolve_imagenet_root() if root_dir is None else Path(root_dir)
    image_dir = base_dir / IMAGENETV2_DIRNAME
    if image_dir.exists():
        return image_dir

    archive_path = _download_imagenetv2_archive(base_dir)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=base_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"ImageNetV2 extraction finished but expected directory was not created: {image_dir}"
        )

    return image_dir


def imagenet_classnames() -> list[str]:
    vendored_imagenet = PROJECT_ROOT / "_tmp_tda_repo" / "datasets" / "imagenet.py"
    if vendored_imagenet.exists():
        module = ast.parse(vendored_imagenet.read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, "id", None) == "imagenet_classes":
                        return list(ast.literal_eval(node.value))

    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    return [str(name).replace("_", " ").lower() for name in categories]


class ImageNetV2Dataset(Dataset):
    def __init__(self, image_dir: Path, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        class_dirs = sorted(
            (path for path in image_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file():
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def load_imagenet(
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    preprocess=None,
):
    transform = preprocess or transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    image_dir = ensure_imagenetv2()
    dataset = ImageNetV2Dataset(image_dir=image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, imagenet_classnames()
