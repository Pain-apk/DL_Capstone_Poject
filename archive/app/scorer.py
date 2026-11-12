"""
app/scorer.py

score_dishes(
    pantry: list,           # canonical ingredient names
    state: str,             # e.g. "Maharashtra"
    protein_focus: str,     # "off" | "on" | "high"
    max_time: int,          # minutes, 0=no limit
    vegetarian_only: bool,
    excluded_allergens: list,
    top_n: int = 5
) -> dict

Returns:
  {
    "top": [
      {
        "dish_id": str,          # dish_name (URL-safe slug)
        "dish_name": str,
        "state": str,
        "diet": str,
        "score": float,
        "ingredient_match": float,  # fraction of MAIN ingredients matched
        "missing_mains": list,
        "protein_estimate_g": float,
        "protein_confidence": str,  # "high"|"medium"|"low"
        "verified": bool,
        "time_minutes": int,
        "rank": int
      }
    ],
    "decision": <top dict above>,
    "note": str    # e.g. "2 dishes filtered: vegetarian_only"
  }
"""

import os
import sys
import csv
import re
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.score import RecipeScorer  # noqa: E402

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")

# Meat / seafood / egg keywords to detect non-veg presence in pantry
NON_VEG_INGREDIENTS = {
    "chicken", "mutton", "fish", "prawns", "pork", "beef", "egg", "eggs",
    "anda", "meat", "surmai", "bombil", "pomfret", "lamb", "seafood"
}

# Staple ingredients that are always assumed available and do NOT count in match fraction
STAPLES = {
    "oil", "salt", "water", "turmeric", "mustard seeds", "cumin seeds",
    "hing", "sugar", "red chilli powder", "coriander powder",
}

# protein_focus -> protein_w for RecipeScorer
_PROTEIN_W_MAP = {
    "off": 0.0,
    "on": 0.15,
    "high": 0.25,
}

# Module-level dish cache: loaded once
_dishes_cache = None  # type: Optional[List[Dict[str, Any]]]


def _load_dishes() -> List[Dict[str, Any]]:
    global _dishes_cache
    if _dishes_cache is not None:
        return _dishes_cache
    dishes = []
    if not os.path.exists(CSV_PATH):
        _dishes_cache = dishes
        return dishes
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dishes.append(dict(row))
    _dishes_cache = dishes
    return dishes


def _make_dish_id(dish_name: str) -> str:
    return dish_name.lower().replace(" ", "_").replace("'", "")


def _split_pipe(value: str) -> List[str]:
    """Split a pipe-separated string into stripped non-empty tokens."""
    return [v.strip() for v in value.split("|") if v.strip()]


def _protein_confidence_from_dish(dish: Dict[str, Any]) -> str:
    """
    Determine protein confidence band from dish metadata.
    - "high"   if verified == "yes"
    - "medium" if source contains "web_verified"
    - "low"    otherwise
    """
    if dish.get("verified", "").strip().lower() == "yes":
        return "high"
    source = dish.get("source", "").strip().lower()
    if "web_verified" in source:
        return "medium"
    return "low"


def score_dishes(
    pantry: List[str],
    state: str,
    protein_focus: str,
    max_time: int,
    vegetarian_only: bool,
    excluded_allergens: List[str],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Score and rank dishes from the CSV against the given pantry.

    Parameters
    ----------
    pantry : list of canonical ingredient names (normalized by app.normalize)
    state  : user's state for regional weighting
    protein_focus : "off" | "on" | "high"
    max_time : max cook time in minutes (0 = no limit)
    vegetarian_only : hard-filter non-veg dishes when True
    excluded_allergens : names of allergens to exclude
    top_n : number of top results to return

    Returns
    -------
    dict with keys: "top", "decision", "note"
    """
    dishes = _load_dishes()
    scorer = RecipeScorer()

    protein_w = _PROTEIN_W_MAP.get(protein_focus.lower().strip(), 0.0)

    # Combine user pantry with staples for matching
    effective_pantry = set(i.lower().strip() for i in pantry) | STAPLES

    # Detect if user entered any non-veg item in pantry
    has_nonveg_in_pantry = any(
        any(m in p.lower() for m in NON_VEG_INGREDIENTS) for p in pantry
    )

    # Allergens lowercased for case-insensitive matching
    allergens_clean = [a.lower().strip() for a in excluded_allergens if a.strip()]

    filter_counts = {}  # reason -> count

    scored = []

    for dish in dishes:
        dish_name = dish.get("dish_name", "").strip()
        meat_type = dish.get("meat_type", "none").strip().lower()
        diet = dish.get("diet", "").strip().lower()
        contains_egg = dish.get("contains_egg", "no").strip().lower()
        time_str = dish.get("time_minutes", "0").strip()
        verified_str = dish.get("verified", "no").strip().lower()
        source = dish.get("source", "").strip()
        main_ings_raw = dish.get("main_ingredients", "")
        ings_qty_raw = dish.get("ingredients_with_quantity", "")
        pantry_raw = dish.get("pantry_items", "")

        is_non_veg = (meat_type != "none" or diet == "non-vegetarian" or contains_egg == "yes")

        # --- Hard filter: vegetarian_only ---
        if vegetarian_only and is_non_veg:
            filter_counts["vegetarian_only"] = filter_counts.get("vegetarian_only", 0) + 1
            continue

        # --- Rule: with no meat in input, non-veg dishes do not appear in top 5 ---
        if (not has_nonveg_in_pantry) and is_non_veg:
            filter_counts["no_meat_in_input"] = filter_counts.get("no_meat_in_input", 0) + 1
            continue

        # --- Hard filter: max_time ---
        try:
            cook_time = int(time_str)
        except (ValueError, TypeError):
            cook_time = 0
        if max_time > 0 and cook_time > max_time:
            filter_counts["max_time"] = filter_counts.get("max_time", 0) + 1
            continue

        # --- Hard filter: allergens ---
        combined_text = (main_ings_raw + " " + ings_qty_raw + " " + pantry_raw).lower()
        allergen_hit = False
        for allergen in allergens_clean:
            # Match whole token to avoid partial matches (e.g., "pea" in "peanut")
            pattern = r"\b" + re.escape(allergen) + r"\b"
            if re.search(pattern, combined_text):
                allergen_hit = True
                break
        if allergen_hit:
            filter_counts["allergen"] = filter_counts.get("allergen", 0) + 1
            continue

        # --- Ingredient match (against main_ingredients only) ---
        main_ings = _split_pipe(main_ings_raw)

        # Essentials = main ingredients minus staples
        essential_mains = [m.lower().strip() for m in main_ings
                           if m.lower().strip() not in STAPLES]
        if not essential_mains:
            essential_mains = [m.lower().strip() for m in main_ings]

        matched = []
        missing = []
        for ing in essential_mains:
            # Flexible match: pantry token appears in ing name or vice versa
            hit = any(ing in p or p in ing for p in effective_pantry)
            if hit:
                matched.append(ing)
            else:
                missing.append(ing)

        ing_match = round(len(matched) / max(len(essential_mains), 1), 4)
        missing_mains = missing

        # --- Region weight via RecipeScorer ---
        pan_india = dish.get("pan_india", "no")
        region_w = scorer.compute_region_weight(
            dish.get("state", ""), state, pan_india
        )

        # --- Diet & time fit ---
        time_limit = max_time if max_time > 0 else None
        dt_fit = scorer.compute_diet_time_fit(cook_time, time_limit)

        # --- Popularity ---
        try:
            popularity = float(dish.get("popularity", 0.8))
        except (ValueError, TypeError):
            popularity = 0.8

        # --- Protein estimation ---
        ings_qty_list = _split_pipe(ings_qty_raw)
        servings_str = dish.get("servings", "2.0").strip()
        try:
            servings = float(servings_str)
        except (ValueError, TypeError):
            servings = 2.0

        protein_g, _prot_conf, _ = scorer.protein_calc.compute_recipe_protein(
            ings_qty_list, main_ings, servings=servings
        )
        prot_confidence = _protein_confidence_from_dish(dish)

        # Protein boost only allowed when ing_match >= threshold
        effective_protein_w = protein_w if ing_match >= scorer.protein_threshold else 0.0

        protein_score = min(protein_g / scorer.protein_target, 1.0)

        # --- Final score (mirror RecipeScorer formula) ---
        w = scorer.weights
        final_score = (
            w.get("ingredient_match", 0.45) * ing_match
            + w.get("region_weight", 0.25) * region_w
            + w.get("diet_time_fit", 0.10) * dt_fit
            + w.get("popularity", 0.05) * popularity
            + effective_protein_w * protein_score
        )

        scored.append({
            "dish_id": _make_dish_id(dish_name),
            "dish_name": dish_name,
            "state": dish.get("state", "").strip(),
            "diet": diet,
            "score": round(final_score, 4),
            "ingredient_match": ing_match,
            "missing_mains": missing_mains,
            "protein_estimate_g": protein_g,
            "protein_confidence": prot_confidence,
            "verified": verified_str == "yes",
            "time_minutes": cook_time,
            "rank": 0,  # assigned below
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks
    for i, entry in enumerate(scored):
        entry["rank"] = i + 1

    top = scored[:top_n]

    # Build note
    note_parts = []
    for reason, count in filter_counts.items():
        note_parts.append(f"{count} dishes filtered: {reason}")
    note = "; ".join(note_parts) if note_parts else "No filters applied"

    return {
        "top": top,
        "decision": top[0] if top else {},
        "note": note,
    }


# ---------------------------------------------------------------------------
# Inline test cases — run with: python app/scorer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    PANTRY = ["onion", "potato", "tomato", "besan", "garlic"]

    def _pass_fail(condition: bool, label: str) -> None:
        print(f"{'PASS' if condition else 'FAIL'}: {label}")

    # Test 1: Same pantry -> Maharashtra dishes rank differently from Gujarat
    r_mh = score_dishes(PANTRY, "Maharashtra", "off", 0, False, [])
    r_gj = score_dishes(PANTRY, "Gujarat", "off", 0, False, [])
    mh_top_state = r_mh["decision"].get("state", "") if r_mh["top"] else None
    gj_top_state = r_gj["decision"].get("state", "") if r_gj["top"] else None
    # Maharashtra dishes should score higher for Maharashtra user (region_weight=1.0 vs 0.2)
    mh_top_score = r_mh["decision"].get("score", 0)
    gj_top_score = r_gj["decision"].get("score", 0)
    # For same pantry the top Maharashtra dish in MH run should have region_weight=1.0
    # whereas in Gujarat run the same dish would have region_weight=0.2
    top_dish_in_mh = r_mh["decision"].get("dish_name", "")
    # Find that dish's score in Gujarat run
    gj_scores = {e["dish_name"]: e["score"] for e in r_gj["top"]}
    gj_all = score_dishes(PANTRY, "Gujarat", "off", 0, False, [], top_n=200)
    gj_all_scores = {e["dish_name"]: e["score"] for e in gj_all["top"]}
    mh_dish_score_in_gj = gj_all_scores.get(top_dish_in_mh)
    ranked_differently = (
        mh_dish_score_in_gj is not None
        and r_mh["decision"]["score"] != mh_dish_score_in_gj
    )
    _pass_fail(ranked_differently, "Test 1: Same pantry ranks Maharashtra dishes differently from Gujarat")

    # Test 2: protein_focus=high re-ranks only when ingredient_match >= 0.6
    r_off = score_dishes(PANTRY, "Maharashtra", "off", 0, False, [])
    r_high = score_dishes(PANTRY, "Maharashtra", "high", 0, False, [])
    # Dishes with ing_match < 0.6 should have same score in both runs
    off_map = {e["dish_name"]: e for e in score_dishes(PANTRY, "Maharashtra", "off", 0, False, [], top_n=200)["top"]}
    high_map_all = {e["dish_name"]: e for e in score_dishes(PANTRY, "Maharashtra", "high", 0, False, [], top_n=200)["top"]}
    low_match_same = all(
        off_map[n]["score"] == high_map_all[n]["score"]
        for n in off_map
        if n in high_map_all and off_map[n]["ingredient_match"] < 0.6
    )
    _pass_fail(low_match_same, "Test 2: protein_focus=high only re-ranks dishes with ingredient_match >= 0.6")

    # Test 3: vegetarian_only=True -> no meat_type != 'none' in results
    r_veg = score_dishes(PANTRY, "Maharashtra", "off", 0, True, [], top_n=100)
    all_veg = all(
        e.get("diet", "") in ("vegetarian", "vegan")
        for e in r_veg["top"]
    )
    _pass_fail(all_veg, "Test 3: vegetarian_only=True returns no non-veg dishes")

    # Test 4: Pantry with peanut allergen excluded -> no dish with peanuts in main_ingredients
    r_no_peanut = score_dishes(
        PANTRY + ["peanuts"], "Maharashtra", "off", 0, False, ["peanuts"], top_n=200
    )
    no_peanut_dishes = all(
        "peanut" not in e["dish_name"].lower()
        and not any("peanut" in m for m in e.get("missing_mains", []))
        for e in r_no_peanut["top"]
    )
    # Also check by directly loading CSV that excluded dishes had peanuts
    all_dishes = _load_dishes()
    peanut_dishes = {
        _make_dish_id(d["dish_name"])
        for d in all_dishes
        if "peanut" in d.get("main_ingredients", "").lower()
    }
    result_ids = {e["dish_id"] for e in r_no_peanut["top"]}
    no_overlap = len(peanut_dishes & result_ids) == 0
    _pass_fail(no_overlap, "Test 4: Peanut allergen excluded -> no dish with peanuts in main_ingredients")
