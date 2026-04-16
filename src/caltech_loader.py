# import torch
# from torchvision.datasets import Caltech101
# from torchvision import transforms
# from torch.utils.data import DataLoader


# def load_caltech(batch_size=32):

#     transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.Lambda(lambda img: img.convert("RGB")),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466,0.4578275,0.40821073],
#             std=[0.26862954,0.26130258,0.27577711]
#         )
#     ])

#     dataset = Caltech101(
#         root="data/raw/CALTECH",
#         download=True,
#         transform=transform
#     )

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     return loader, dataset.categories

# import torch
from pathlib import Path
from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_caltech_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data/raw/CALTECH_clean",
        PROJECT_ROOT / "data/raw/CALTECH_fresh",
        PROJECT_ROOT / "data/raw/CALTECH",
    ]
    extracted_suffix = Path("caltech101/101_ObjectCategories")

    for candidate in candidates:
        if (candidate / extracted_suffix).exists():
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


def load_caltech(batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False):

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466,0.4578275,0.40821073],
            std=[0.26862954,0.26130258,0.27577711]
        )
    ])

    root_dir = _resolve_caltech_root()
    dataset = Caltech101(
        root=str(root_dir),
        download=True,
        transform=transform
    )

    if len(dataset) == 0:
        categories_dir = Path(dataset.root) / "101_ObjectCategories"
        if categories_dir.exists():
            category_count = len([item for item in categories_dir.iterdir() if item.is_dir()])
            details = f"{category_count} category folders found in {categories_dir}"
        else:
            details = f"missing dataset directory: {categories_dir}"
        raise RuntimeError(
            "Caltech101 loader resolved to an empty dataset. "
            f"Root: {root_dir}. Details: {details}."
        )

    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset.categories
