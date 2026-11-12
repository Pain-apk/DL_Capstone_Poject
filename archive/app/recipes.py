"""
app/recipes.py

get_recipe(dish_id: str, mode: str = "pilot") -> dict or None

mode="pilot": returns recipe for any drafted row (verified or not)
mode="final": returns recipe only for verified=yes rows

Returns:
  {
    "dish_id": str,
    "dish_name": str,
    "local_name": str,
    "state": str,
    "diet": str,
    "meat_type": str,
    "servings": float,
    "time_minutes": int,
    "time_notes": str,
    "ingredients_with_quantity": list,  # [{"qty": str, "item": str}, ...]
    "steps": list,                       # list of step strings
    "main_ingredients": list,
    "pantry_items": list,
    "verified": bool,
    "source": str,
    "review_flag": str
  }
  or None if not found / not verified in final mode
"""

import os
import sys
import csv
import re
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")

# ---------------------------------------------------------------------------
# Ingredient quantity parser (matches Phase 0b fixed parser logic)
# ---------------------------------------------------------------------------

# Unit tokens that mark the END of the quantity portion.
# A quantity ends at the LAST unit token; everything after is the item name.
UNIT_TOKENS = {
    "tsp", "tbsp", "cup", "cups", "tablespoon", "tablespoons",
    "teaspoon", "teaspoons", "g", "gram", "grams", "kg", "ml",
    "liter", "litre", "liters", "litres",
    "medium", "large", "small", "pinch", "pinches",
    "bunch", "clove", "cloves", "piece", "pieces",
    "inch", "inches",
}


def _parse_ingredient_line(line: str) -> Dict[str, str]:
    """
    Parse a single ingredient string like "1.5 cup poha" or "1/2 tsp turmeric"
    into {"qty": str, "item": str}.

    Rule: quantity ends at the last unit token. Everything after is the item.
    If no unit token is found, quantity is the leading number and item is the rest.
    """
    line = line.strip()
    if not line:
        return {"qty": "", "item": ""}

    tokens = line.split()
    if not tokens:
        return {"qty": "", "item": line}

    # Find the index of the last unit token (case-insensitive)
    last_unit_idx = -1
    for i, tok in enumerate(tokens):
        if tok.lower() in UNIT_TOKENS:
            last_unit_idx = i

    if last_unit_idx >= 0:
        qty_tokens = tokens[: last_unit_idx + 1]
        item_tokens = tokens[last_unit_idx + 1 :]
        return {
            "qty": " ".join(qty_tokens),
            "item": " ".join(item_tokens) if item_tokens else " ".join(qty_tokens),
        }

    # No unit token: check if first token looks like a number / fraction
    first = tokens[0]
    is_number = bool(re.match(r"^\d+([./]\d+)?$", first))
    if is_number and len(tokens) > 1:
        return {
            "qty": first,
            "item": " ".join(tokens[1:]),
        }

    # Cannot determine quantity — treat full string as item
    return {"qty": "", "item": line}


def _parse_ingredients_with_quantity(raw: str) -> List[Dict[str, str]]:
    """Split pipe-separated ingredient string and parse each entry."""
    if not raw.strip():
        return []
    return [_parse_ingredient_line(entry.strip()) for entry in raw.split("|") if entry.strip()]


def _split_pipe(value: str) -> List[str]:
    return [v.strip() for v in value.split("|") if v.strip()]


def _make_dish_id(dish_name: str) -> str:
    return dish_name.lower().replace(" ", "_").replace("'", "")


# ---------------------------------------------------------------------------
# Module-level cache: keyed by dish_id
# ---------------------------------------------------------------------------
_recipe_cache = None  # type: Optional[Dict[str, Dict[str, Any]]]


def _load_recipes() -> Dict[str, Dict[str, Any]]:
    global _recipe_cache
    if _recipe_cache is not None:
        return _recipe_cache

    cache = {}
    if not os.path.exists(CSV_PATH):
        _recipe_cache = cache
        return cache

    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_name = row.get("dish_name", "").strip()
            if not dish_name:
                continue

            dish_id = _make_dish_id(dish_name)
            verified_str = row.get("verified", "no").strip().lower()

            try:
                servings = float(row.get("servings", "2.0").strip() or "2.0")
            except ValueError:
                servings = 2.0

            try:
                time_minutes = int(row.get("time_minutes", "0").strip() or "0")
            except ValueError:
                time_minutes = 0

            recipe = {
                "dish_id": dish_id,
                "dish_name": dish_name,
                "local_name": row.get("romanized_local_name", "").strip(),
                "state": row.get("state", "").strip(),
                "diet": row.get("diet", "").strip(),
                "meat_type": row.get("meat_type", "none").strip(),
                "servings": servings,
                "time_minutes": time_minutes,
                "time_notes": row.get("time_notes", "").strip(),
                "ingredients_with_quantity": _parse_ingredients_with_quantity(
                    row.get("ingredients_with_quantity", "")
                ),
                "steps": _split_pipe(row.get("steps", "")),
                "main_ingredients": _split_pipe(row.get("main_ingredients", "")),
                "pantry_items": _split_pipe(row.get("pantry_items", "")),
                "verified": verified_str == "yes",
                "source": row.get("source", "").strip(),
                "review_flag": row.get("review_flag", "").strip(),
            }

            # If the same dish_id appears multiple times, prefer verified rows
            if dish_id not in cache or recipe["verified"]:
                cache[dish_id] = recipe

    _recipe_cache = cache
    return cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recipe(dish_id: str, mode: str = "pilot") -> Optional[Dict[str, Any]]:
    """
    Retrieve a recipe by dish_id.

    Parameters
    ----------
    dish_id : URL-safe slug, e.g. "kanda_poha"
    mode    : "pilot" (any row) | "final" (verified=yes only)

    Returns
    -------
    Recipe dict, or None if not found / not verified in final mode.
    Never returns test fixtures.
    """
    # Safety: never serve from tests/fixtures path
    if "tests/fixtures" in dish_id or "tests\\fixtures" in dish_id:
        return None

    cache = _load_recipes()
    recipe = cache.get(dish_id)

    if recipe is None:
        return None

    if mode == "final" and not recipe["verified"]:
        return None

    return recipe
