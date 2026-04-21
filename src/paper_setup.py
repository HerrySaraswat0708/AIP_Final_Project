from __future__ import annotations

from typing import Dict, Tuple


CALTECH_TEMPLATES: Tuple[str, ...] = (
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)

DTD_TEMPLATES: Tuple[str, ...] = ("{} texture.",)

EUROSAT_TEMPLATES: Tuple[str, ...] = ("a centered satellite photo of {}.",)

PETS_TEMPLATES: Tuple[str, ...] = ("a photo of a {}, a type of pet.",)

IMAGENET_TEMPLATES: Tuple[str, ...] = CALTECH_TEMPLATES

EUROSAT_CLASSNAMES: Dict[str, str] = {
    "annualcrop": "annual crop land",
    "forest": "forest",
    "herbaceousvegetation": "herbaceous vegetation land",
    "highway": "highway or road",
    "industrial": "industrial buildings",
    "pasture": "pasture land",
    "permanentcrop": "permanent crop land",
    "residential": "residential buildings",
    "river": "river",
    "sealake": "sea or lake",
    "annual crop land": "annual crop land",
    "herbaceous vegetation land": "herbaceous vegetation land",
    "highway or road": "highway or road",
    "industrial buildings": "industrial buildings",
    "pasture land": "pasture land",
    "permanent crop land": "permanent crop land",
    "residential buildings": "residential buildings",
    "sea or lake": "sea or lake",
}

EXPECTED_TEST_SPLIT_SIZES: Dict[str, int] = {
    "caltech": 2465,
    "dtd": 1880,
    "eurosat": 8100,
    "pets": 3669,
    "imagenet": 10000,
}


def normalize_eurosat_classname(name: str) -> str:
    raw = str(name).strip().lower()
    compact = raw.replace(" ", "")
    return EUROSAT_CLASSNAMES.get(compact, EUROSAT_CLASSNAMES.get(raw, raw))
