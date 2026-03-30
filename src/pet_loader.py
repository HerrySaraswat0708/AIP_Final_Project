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

from pathlib import Path
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader


def _resolve_pet_root() -> str:
    candidates = [
        Path("data/raw/PET_fresh"),
        Path("data/raw/PET"),
    ]
    for candidate in candidates:
        if (candidate / "oxford-iiit-pet").exists():
            return str(candidate)
    # Keep legacy behavior as a last resort.
    return "data/raw/PET"


def _filter_missing_pet_files(dataset: OxfordIIITPet) -> int:
    if not hasattr(dataset, "_images"):
        return 0

    valid_indices = []
    for idx, impath in enumerate(dataset._images):
        if Path(impath).exists():
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


def load_pets():

    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466,0.4578275,0.40821073],
            std=[0.26862954,0.26130258,0.27577711]
        )
    ])

    root_dir = _resolve_pet_root()
    dataset=OxfordIIITPet(
        root=root_dir,
        split="test",
        download=True,
        transform=transform
    )
    removed = _filter_missing_pet_files(dataset)
    if removed > 0:
        print(f"[Warning] OxfordPets: skipped {removed} missing/corrupt image files.")

    # loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return loader,dataset.classes
