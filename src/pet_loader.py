# from torchvision.datasets import OxfordIIITPet
# from torchvision import transforms
# from torch.utils.data import DataLoader


# def load_pets(batch_size=32):

#     transform=transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466,0.4578275,0.40821073],
#             std=[0.26862954,0.26130258,0.27577711]
#         )
#     ])

#     dataset=OxfordIIITPet(
#         root="data/raw/PET",
#         split="test",
#         download=True,
#         transform=transform
#     )

#     loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)

#     return loader,dataset.classes

# from pathlib import Path
# from torchvision.datasets import OxfordIIITPet
# from torchvision import transforms
# from torch.utils.data import DataLoader


# def _resolve_pet_root() -> str:
#     candidates = [
#         Path("data/raw/PET_fresh"),
#         Path("data/raw/PET"),
#     ]
#     for candidate in candidates:
#         if (candidate / "oxford-iiit-pet").exists():
#             return str(candidate)
#     # Keep legacy behavior as a last resort.
#     return "data/raw/PET"


# def _filter_missing_pet_files(dataset: OxfordIIITPet) -> int:
#     if not hasattr(dataset, "_images"):
#         return 0

#     valid_indices = []
#     for idx, impath in enumerate(dataset._images):
#         if Path(impath).exists():
#             valid_indices.append(idx)

#     removed = len(dataset._images) - len(valid_indices)
#     if removed <= 0:
#         return 0

#     def _subset_if_present(attr_name: str) -> None:
#         if hasattr(dataset, attr_name):
#             values = getattr(dataset, attr_name)
#             try:
#                 setattr(dataset, attr_name, [values[i] for i in valid_indices])
#             except Exception:
#                 pass

#     _subset_if_present("_images")
#     _subset_if_present("_labels")
#     _subset_if_present("_segs")
#     return removed


# def load_pets():

#     transform=transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466,0.4578275,0.40821073],
#             std=[0.26862954,0.26130258,0.27577711]
#         )
#     ])

#     root_dir = _resolve_pet_root()
#     dataset=OxfordIIITPet(
#         root=root_dir,
#         split="test",
#         download=True,
#         transform=transform
#     )
#     removed = _filter_missing_pet_files(dataset)
#     if removed > 0:
#         print(f"[Warning] OxfordPets: skipped {removed} missing/corrupt image files.")

#     # loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
#     loader = DataLoader(
#         dataset,
#         batch_size=128,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#     )

#     return loader,dataset.classes



# from torchvision.datasets import OxfordIIITPet
# from torchvision import transforms
# from torch.utils.data import DataLoader


# def load_pets(batch_size=32):

#     transform=transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466,0.4578275,0.40821073],
#             std=[0.26862954,0.26130258,0.27577711]
#         )
#     ])

#     dataset=OxfordIIITPet(
#         root="data/raw/PET",
#         split="test",
#         download=True,
#         transform=transform
#     )

#     loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)

#     return loader,dataset.classes

from __future__ import annotations

from pathlib import Path
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _pet_root_candidates() -> list[Path]:
    return [
        # _resolve_project_path("data", "raw", "PET_fresh"),
        _resolve_project_path("data", "raw", "PET"),
    ]


def _pet_image_dir(root_dir: Path) -> Path:
    return root_dir / "oxford-iiit-pet" / "images"


def _count_missing_split_entries(image_dir: Path, split_entries: list[list[object]]) -> int:
    missing = 0
    for rel_name, *_ in split_entries:
        if not _safe_exists(image_dir / str(rel_name)):
            missing += 1
    return missing


def resolve_pet_root(split_entries: list[list[object]] | None = None) -> Path | None:
    existing_roots = [candidate for candidate in _pet_root_candidates() if _safe_exists(_pet_image_dir(candidate))]
    if not existing_roots:
        return None

    if not split_entries:
        return existing_roots[0]

    best_root = None
    best_missing = None
    for candidate in existing_roots:
        missing = _count_missing_split_entries(_pet_image_dir(candidate), split_entries)
        if best_missing is None or missing < best_missing:
            best_root = candidate
            best_missing = missing

    return best_root


def resolve_pet_image_root(split_entries: list[list[object]] | None = None) -> Path | None:
    root_dir = resolve_pet_root(split_entries=split_entries)
    if root_dir is None:
        return None
    return _pet_image_dir(root_dir)


def _resolve_pet_root() -> str:
    root_dir = resolve_pet_root()
    if root_dir is not None:
        return str(root_dir)
    return "data/raw/PET"


def _filter_missing_pet_files(dataset: OxfordIIITPet) -> int:
    if not hasattr(dataset, "_images"):
        return 0

    valid_indices = []
    for idx, impath in enumerate(dataset._images):
        if _safe_exists(Path(impath)):
            valid_indices.append(idx)

    removed = len(dataset._images) - len(valid_indices)
    if removed <= 0:
        return 0

    def _subset_if_present(attr_name: str) -> None:
        if hasattr(dataset, attr_name):
            values = getattr(dataset, attr_name)
            try:
                setattr(dataset, attr_name, [values[i] for i in valid_indices])
            except Exception:
                pass

    _subset_if_present("_images")
    _subset_if_present("_labels")
    _subset_if_present("_segs")
    return removed


def load_pets(batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    root_dir = _resolve_pet_root()
    dataset = OxfordIIITPet(
        root=root_dir,
        split="test",
        download=True,
        transform=transform
    )
    removed = _filter_missing_pet_files(dataset)
    if removed > 0:
        print(f"[Warning] OxfordPets: skipped {removed} missing/corrupt image files.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader, dataset.classes
