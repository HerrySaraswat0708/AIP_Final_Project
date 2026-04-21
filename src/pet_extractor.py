# import json
# from pathlib import Path

# import clip
# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# from src.pet_loader import load_pets


# class SplitImageDataset(Dataset):
#     def __init__(self, samples, transform):
#         self.samples = samples
#         self.transform = transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         image_path, label = self.samples[idx]
#         image = Image.open(image_path).convert("RGB")
#         return self.transform(image), label


# def _load_pets_split(preprocess):
#     split_path = Path("data/splits/split_zhou_OxfordPets.json")
#     root_candidates = [
#         Path("data/raw/PET_fresh/oxford-iiit-pet/images"),
#         Path("data/raw/PET/oxford-iiit-pet/images"),
#     ]

#     if not split_path.exists():
#         return None

#     root_dir = None
#     for candidate in root_candidates:
#         if candidate.exists():
#             root_dir = candidate
#             break

#     if root_dir is None:
#         return None

#     payload = json.loads(split_path.read_text(encoding="utf-8"))
#     test_entries = payload["test"]

#     samples = []
#     max_label = max(entry[1] for entry in test_entries)
#     class_names = [""] * (max_label + 1)

#     missing = 0
#     for rel_name, label, class_name in test_entries:
#         image_path = root_dir / rel_name
#         if not image_path.exists():
#             missing += 1
#             continue
#         samples.append((image_path, int(label)))
#         class_names[int(label)] = str(class_name).strip().lower()

#     if not samples:
#         return None
#     if missing > 0:
#         print(f"[Warning] OxfordPets split: skipped {missing} missing files from split_zhou test set.")

#     dataset = SplitImageDataset(samples=samples, transform=preprocess)
#     loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
#     return loader, class_names


# def extract_pets():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("CUDA available:", torch.cuda.is_available())
#     print("Using device:", device)
#     if device.type == "cuda":
#         print("GPU:", torch.cuda.get_device_name(0))

#     model, preprocess = clip.load("ViT-B/16", device=device)
#     model.eval()

#     split_loaded = _load_pets_split(preprocess=preprocess)
#     if split_loaded is None:
#         print("split_zhou_OxfordPets.json not available/readable, falling back to torchvision loader.")
#         loader, class_names = load_pets()
#         class_names = [c.replace("_", " ").lower() for c in class_names]
#     else:
#         loader, class_names = split_loaded
#         print(f"Using split_zhou test split for OxfordPets: {len(loader.dataset)} samples")

#     image_features = []
#     labels = []
#     with torch.no_grad():
#         for images, targets in tqdm(loader):
#             images = images.to(device, non_blocking=True)
#             features = model.encode_image(images)
#             features = features / features.norm(dim=-1, keepdim=True)
#             image_features.append(features.detach())
#             labels.append(targets)

#     image_features = torch.cat(image_features).cpu().numpy()
#     labels = torch.cat(labels).numpy()

#     templates = [
#         "a photo of a {}, a type of pet.",
#         "a photo of the pet {}.",
#         "a cute photo of a {}.",
#     ]
#     with torch.no_grad():
#         text_feature_list = []
#         for template in templates:
#             prompts = [template.format(c) for c in class_names]
#             tokens = clip.tokenize(prompts).to(device)
#             text_features = model.encode_text(tokens)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             text_feature_list.append(text_features)
#         text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     np.save("data/processed/pets_image_features.npy", image_features)
#     np.save("data/processed/pets_text_features.npy", text_features.cpu().numpy())
#     np.save("data/processed/pets_labels.npy", labels)
#     print("Feature extraction completed!")


# if __name__ == "__main__":
#     extract_pets()



import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_compat import get_clip_module, get_extraction_runtime
from src.paper_setup import EXPECTED_TEST_SPLIT_SIZES, PETS_TEMPLATES
from src.pet_loader import load_pets, resolve_pet_image_root


def _resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


class SplitImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def _load_pets_split(preprocess, batch_size: int, num_workers: int, pin_memory: bool):
    split_path = _resolve_project_path("data", "splits", "split_zhou_OxfordPets.json")

    if not split_path.exists():
        return None

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_entries = payload.get("test", [])
    if not test_entries:
        return None

    root_dir = resolve_pet_image_root(split_entries=test_entries)
    if root_dir is None:
        return None

    samples = []
    max_label = max(entry[1] for entry in test_entries)
    class_names = [""] * (max_label + 1)

    missing = 0
    for rel_name, label, class_name in test_entries:
        image_path = root_dir / rel_name
        if not _safe_exists(image_path):
            missing += 1
            continue
        samples.append((image_path, int(label)))
        class_names[int(label)] = str(class_name).strip().lower()

    if not samples:
        print(f"[Warning] OxfordPets split test set resolved to zero readable samples under: {root_dir}")
        return None
    if missing > 0:
        print(
            "[Warning] OxfordPets split: skipped "
            f"{missing} missing/unreadable files from split_zhou test set under {root_dir}."
        )

    dataset = SplitImageDataset(samples=samples, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, class_names, root_dir


def extract_pets():
    device, batch_size, num_workers, pin_memory = get_extraction_runtime()
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Batch size:", batch_size)
    print("DataLoader workers:", num_workers)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    clip = get_clip_module()
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    split_loaded = _load_pets_split(
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if split_loaded is None:
        print("split_zhou_OxfordPets.json not available/readable, falling back to torchvision loader.")
        loader, class_names = load_pets(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        class_names = [c.replace("_", " ").lower() for c in class_names]
        loader_source = "torchvision OxfordIIITPet loader"
    else:
        loader, class_names, root_dir = split_loaded
        print(f"Using split_zhou test split for OxfordPets: {len(loader.dataset)} samples from {root_dir}")
        loader_source = f"split_zhou test split ({root_dir})"

    dataset_size = len(loader.dataset)
    print(f"OxfordPets sample count from {loader_source}: {dataset_size}")
    if dataset_size == 0:
        raise RuntimeError(
            "OxfordPets feature extraction received an empty dataset. "
            f"Source: {loader_source}. cwd={Path.cwd()} project_root={PROJECT_ROOT}"
        )

    image_features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader, total=len(loader), desc="Encoding OxfordPets images"):
            images = images.to(device, non_blocking=True)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            image_features.append(features.detach())
            labels.append(targets)

    if not image_features:
        raise RuntimeError(
            "OxfordPets feature extraction produced zero batches before concatenation. "
            f"Source: {loader_source}. Dataset size: {dataset_size}."
        )

    image_features = torch.cat(image_features).cpu().numpy()
    labels = torch.cat(labels).numpy()

    expected = EXPECTED_TEST_SPLIT_SIZES["pets"]
    if dataset_size != expected:
        print(f"[Warning] OxfordPets sample count {dataset_size} differs from paper split size {expected}.")

    with torch.no_grad():
        text_feature_list = []
        for template in PETS_TEMPLATES:
            prompts = [template.format(c) for c in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature_list.append(text_features)
        text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    output_dir = _resolve_project_path("data", "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "pets_image_features.npy", image_features)
    np.save(output_dir / "pets_text_features.npy", text_features.cpu().numpy())
    np.save(output_dir / "pets_labels.npy", labels)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_pets()
