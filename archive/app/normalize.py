"""
app/normalize.py

normalize(names: list) -> list

Takes a list of raw names (from detector or typed by user).
Returns list of canonical ingredient names using src/ingredient_synonyms.json.
Also maps detector class names (brinjal -> eggplant, green_chilli -> green chilli, etc.)
Lowercases and strips before lookup. Unknown names pass through unchanged.
"""

import os
import sys
import json
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SYNONYMS_PATH = os.path.join(PROJECT_ROOT, "src", "ingredient_synonyms.json")

# Additional detector-class-name normalizations not in the JSON synonym map.
# These handle YOLO class label conventions (underscores, regional aliases).
_DETECTOR_CLASS_MAP = {
    "brinjal": "eggplant",
    "baingan": "eggplant",
    "aubergine": "eggplant",
    "green_chilli": "green chilli",
    "green_chili": "green chilli",
    "chilli": "green chilli",
    "chili": "green chilli",
    "spring_onion": "spring onion",
    "capsicum": "capsicum",
    "bell_pepper": "capsicum",
    "lady_finger": "okra",
    "ladyfinger": "okra",
    "bhindi": "okra",
    "bitter_gourd": "bitter gourd",
    "karela": "bitter gourd",
    "ridge_gourd": "ridge gourd",
    "turai": "ridge gourd",
    "bottle_gourd": "bottle gourd",
    "lauki": "bottle gourd",
    "dudhi": "bottle gourd",
    "drumstick": "drumstick",
    "moringa": "drumstick",
    "raw_banana": "raw banana",
    "plantain": "raw banana",
    "coconut": "coconut",
    "nariyal": "coconut",
    "naral": "coconut",
}

# Module-level synonym cache: loaded once
_synonyms = None  # type: dict


def _load_synonyms() -> dict:
    global _synonyms
    if _synonyms is None:
        if os.path.exists(SYNONYMS_PATH):
            with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                _synonyms = json.load(f)
        else:
            _synonyms = {}
    return _synonyms


def normalize(names: List[str]) -> List[str]:
    """
    Canonicalize a list of raw ingredient names.

    Steps per name:
    1. Lowercase + strip whitespace
    2. Replace underscores with spaces (detector class name cleanup)
    3. Check detector class map
    4. Check JSON synonym map (direct lookup)
    5. Check JSON synonym map (substring/token lookup)
    6. Return unchanged if no match found
    """
    synonyms = _load_synonyms()
    result = []

    for raw in names:
        # Step 1: normalize whitespace
        name = raw.lower().strip()

        # Step 2: underscores -> spaces (YOLO class name style)
        name_spaced = name.replace("_", " ")

        # Step 3: detector class map (check both underscore and spaced forms)
        if name in _DETECTOR_CLASS_MAP:
            result.append(_DETECTOR_CLASS_MAP[name])
            continue
        if name_spaced in _DETECTOR_CLASS_MAP:
            result.append(_DETECTOR_CLASS_MAP[name_spaced])
            continue

        # From here on use the space-normalized form
        name = name_spaced

        # Step 4: direct synonym lookup
        if name in synonyms:
            result.append(synonyms[name])
            continue

        # Step 5: token/substring synonym lookup
        matched = None
        for syn, target in synonyms.items():
            if syn in name or name in syn:
                matched = target
                break
        if matched is not None:
            result.append(matched)
            continue

        # Step 6: unknown — pass through (space-normalized)
        result.append(name)

    return result
