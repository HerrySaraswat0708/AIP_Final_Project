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
from src.eurosat_loader import load_eurosat


class SplitImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


NEW_CLASSNAMES = {
    "annualcrop": "annual crop land",
    "forest": "forest",
    "herbaceousvegetation": "brushland or shrubland",
    "highway": "highway or road",
    "industrial": "industrial buildings or commercial buildings",
    "pasture": "pasture land",
    "permanentcrop": "permanent crop land",
    "residential": "residential buildings or homes or apartments",
    "river": "river",
    "sealake": "lake or sea",
    "annual crop land": "annual crop land",
    "herbaceous vegetation land": "brushland or shrubland",
    "highway or road": "highway or road",
    "industrial buildings": "industrial buildings or commercial buildings",
    "pasture land": "pasture land",
    "permanent crop land": "permanent crop land",
    "residential buildings": "residential buildings or homes or apartments",
    "sea or lake": "lake or sea",
}


def _load_eurosat_split(preprocess, batch_size: int, num_workers: int, pin_memory: bool):
    split_path = Path("data/splits/split_zhou_EuroSAT.json")
    root_candidates = [
        Path("data/raw/EUROSAT/eurosat/2750"),
        Path("data/raw/EUROSAT_fresh/eurosat/2750"),
    ]

    if not split_path.exists():
        return None

    root_dir = None
    for candidate in root_candidates:
        if candidate.exists():
            root_dir = candidate
            break

    if root_dir is None:
        return None

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_entries = payload["test"]

    samples = []
    max_label = max(entry[1] for entry in test_entries)
    class_names = [""] * (max_label + 1)

    for rel_path, label, class_name in test_entries:
        image_path = root_dir / rel_path
        if not image_path.exists():
            return None
        samples.append((image_path, int(label)))
        raw_name = str(class_name).strip()
        mapped = NEW_CLASSNAMES.get(raw_name.lower().replace(" ", ""), None)
        if mapped is None:
            mapped = NEW_CLASSNAMES.get(raw_name.lower(), raw_name.lower())
        class_names[int(label)] = mapped

    dataset = SplitImageDataset(samples=samples, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, class_names


def extract_eurosat():
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

    split_loaded = _load_eurosat_split(
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if split_loaded is None:
        print("split_zhou_EuroSAT.json not available/readable, falling back to torchvision loader.")
        loader, class_names = load_eurosat(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        class_names = [NEW_CLASSNAMES.get(c.lower(), c.replace("_", " ").lower()) for c in class_names]
    else:
        loader, class_names = split_loaded
        print(f"Using split_zhou test split for EuroSAT: {len(loader.dataset)} samples")

    image_features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device, non_blocking=True)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            image_features.append(features.detach())
            labels.append(targets)

    image_features = torch.cat(image_features).cpu().numpy()
    labels = torch.cat(labels).numpy()

    templates = [
        "aerial imagery showing {}.",
        "an orthographic view of {}.",
    ]
    with torch.no_grad():
        text_feature_list = []
        for template in templates:
            prompts = [template.format(c) for c in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature_list.append(text_features)
        text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "eurosat_image_features.npy", image_features)
    np.save(output_dir / "eurosat_text_features.npy", text_features.cpu().numpy())
    np.save(output_dir / "eurosat_labels.npy", labels)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_eurosat()
