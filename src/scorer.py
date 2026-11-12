"""
src/scorer.py — Pure Python recipe candidate scorer.
NO FastAPI imports allowed in this module.

Formula:
  score = 0.45*ingredient_match + 0.25*region + 0.10*diet_time_fit + 0.05*popularity + protein_w*min(protein_g/25, 1.0)
Rules:
  - Allergen and vegetarian filters run strictly BEFORE scoring.
  - Eligible only if >=50% of non-staple main ingredients are present AND >=2 matched.
    Staples and pantry items NEVER count as matches.
  - No protein boost if ingredient_match < 0.6.
  - Region: same state = 1.0, neighbour = 0.6, else = 0.2.
  - Sort by score descending, tie-break by dish id.
  - If none eligible: {"dishes": [], "message": "No dishes found"}.
"""

import os
import sys
import csv
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.protein import ProteinCalculator

# Canonical mapping for ingredient names
# Canonical names (eggplant -> brinjal, zucchini/courgette -> cucumber)
CANONICAL_MAP = {
    "eggplant": "brinjal",
    "aubergine": "brinjal",
    "baingan": "brinjal",
    "eggplant/aubergine": "brinjal",
    "zucchini": "cucumber",
    "courgette": "cucumber",
    "zucchini/courgette": "cucumber",
    "cuke": "cucumber",
    "cucumber/cuke": "cucumber",
    "bell pepper": "capsicum",
    "bell_pepper": "capsicum",
    "bell pepper/capsicum": "capsicum",
    "chili": "green chilli",
    "chilli": "green chilli",
    "green_chilli": "green chilli",
    "green_chili": "green chilli",
    "green chillies": "green chilli",
    "chili pepper": "green chilli",
    "chili/chili vegetable/chili pepper/chili pepper vegetable/chilli/chilli vegetable/chilly/chilly": "green chilli",
    "garlic/ail": "garlic",
    "ginger/gingerroot": "ginger",
    "pea": "green peas",
    "peas": "green peas",
    "pea/pea food": "green peas",
    "matar": "green peas",
    "orange/orange fruit": "orange",
    "coconut/cocoanut": "coconut",
    "bean curd/tofu": "tofu",
    "chickpea/garbanzo": "chickpea",
    "ladyfinger": "okra",
    "lady_finger": "okra",
    "bhindi": "okra",
    "cilantro": "coriander leaves",
    "kothimbir": "coriander leaves",
    "coriander": "coriander leaves",
    "dhania": "coriander leaves",
    "dhania patta": "coriander leaves",
    "dhaniya patta": "coriander leaves",
    "patta gobi": "cabbage",
    "palak": "spinach",
    "eggs": "egg",
    "anda": "egg",
    "cottage cheese": "paneer",
    "dahi": "curd",
    "yogurt": "curd",
    "yoghurt": "curd",
    "batata": "potato",
    "aloo": "potato",
    "kanda": "onion",
    "pyaaz": "onion",
    "gajar": "carrot",
    "nimbu": "lemon",
    "limbu": "lemon",
    "gobi": "cauliflower",
    "phool gobi": "cauliflower",
    "shengdana": "peanuts",
    "singdana": "peanuts",
    "moongphali": "peanuts",
    "groundnut": "peanuts",
    "groundnuts": "peanuts",
}

# Unit tokens for the ingredient-quantity parser
UNIT_TOKENS = {
    "tsp", "tbsp", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
    "cup", "cups", "g", "kg", "ml", "l", "oz", "lb", "lbs",
    "medium", "large", "small", "pinch", "bunch", "clove", "cloves",
    "piece", "pieces", "slice", "slices",
}

def parse_ingredient_line_fixed(line: str) -> Tuple[str, str]:
    """
    Parses '1.5 cup poha|8 curry leaves|salt to taste' into (quantity, item).
    Fixes the bug where '8 curry leaves' became quantity='8 curry', item='leaves'.
    """
    line = line.strip()
    if not line:
        return ("", "")
    if "to taste" in line.lower():
        name = re.sub(r"\bto taste\b", "", line, flags=re.IGNORECASE).strip()
        return ("to taste", name if name else "salt")
    tokens = line.split()
    num_re = re.compile(r"^\d+([./]\d+)?$")
    if not num_re.match(tokens[0]):
        return ("1 portion", line)
    qty_tokens = [tokens[0]]
    rest = tokens[1:]
    for tok in rest:
        if tok.lower() in UNIT_TOKENS:
            qty_tokens.append(tok)
        else:
            break
    item_tokens = tokens[len(qty_tokens):]
    qty_str = " ".join(qty_tokens)
    item_str = " ".join(item_tokens).strip()
    if not item_str:
        if len(qty_tokens) > 1:
            item_str = qty_tokens[-1]
            qty_str = " ".join(qty_tokens[:-1])
        else:
            item_str = "ingredient"
    return (qty_str, item_str)

def canonical_name(name: str) -> str:
    """Normalize raw ingredient to canonical form."""
    cleaned = name.lower().strip()
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in CANONICAL_MAP:
        return CANONICAL_MAP[cleaned]
    # Check simple plural strip (e.g. onions -> onion, potatoes -> potato, tomatoes -> tomato)
    if cleaned.endswith("es"):
        stem = cleaned[:-2]
        if stem in CANONICAL_MAP:
            return CANONICAL_MAP[stem]
        if stem in ["tomato", "potato"]:
            return stem
    elif cleaned.endswith("s") and not cleaned.endswith("peas"):
        stem = cleaned[:-1]
        if stem in CANONICAL_MAP:
            return CANONICAL_MAP[stem]
        if stem in ["onion", "carrot", "lemon", "egg"]:
            return stem
    return cleaned

class DishScorer:
    def __init__(
        self,
        catalog_path: str = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv"),
        config_path: str = os.path.join(PROJECT_ROOT, "config.json"),
        staples_path: str = os.path.join(PROJECT_ROOT, "data", "staples.json"),
        neighbours_path: str = os.path.join(PROJECT_ROOT, "data", "raw", "state_neighbours.csv")
    ):
        self.catalog_path = catalog_path
        self.config_path = config_path
        self.staples_path = staples_path
        self.neighbours_path = neighbours_path

        # 1. Load config
        self.config = self._load_config()
        self.weights = self.config.get("weights", {
            "ingredient_match": 0.45,
            "region_weight": 0.25,
            "diet_time_fit": 0.10,
            "popularity": 0.05,
            "protein_w": 0.0
        })

        # 2. Load staples
        self.staples = self._load_staples()

        # 3. Load state neighbours
        self.neighbours = self._load_neighbours()

        # 4. Load ProteinCalculator
        self.protein_calc = ProteinCalculator()

        # 5. Load and clean dish catalog
        self.dishes = self._load_and_clean_catalog()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_staples(self) -> Set[str]:
        if os.path.exists(self.staples_path):
            with open(self.staples_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {canonical_name(x) for x in data}
        return {"salt", "oil", "water", "turmeric", "red chilli powder", "mustard seeds", "cumin seeds", "hing", "sugar", "ghee"}

    def _load_neighbours(self) -> Dict[str, Set[str]]:
        nb: Dict[str, Set[str]] = {}
        if os.path.exists(self.neighbours_path):
            with open(self.neighbours_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    st = row["state"].strip().title()
                    nbr = row["neighbour"].strip().title()
                    if st not in nb:
                        nb[st] = set()
                    nb[st].add(nbr)
        return nb

    def _load_and_clean_catalog(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.catalog_path):
            print(f"[Scorer Warning] Catalog path {self.catalog_path} not found.")
            return []

        raw_rows = []
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(dict(row))

        # Check ingredient split bug before and after fix
        before_broken = 0
        legacy_regex = re.compile(r"^(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)\s*([a-zA-Z]+)?\s*(.*)$")
        for r in raw_rows:
            iq_str = r.get("ingredients_with_quantity", "")
            lines = [x.strip() for x in str(iq_str).split("|") if x.strip()]
            for line in lines:
                m = legacy_regex.match(line)
                if m:
                    _, unit, name = m.groups()
                    if (unit and unit.lower() == "curry" and name and name.lower() == "leaves") or (name and name.lower() == "leaves"):
                        before_broken += 1
                        break

        cleaned_dishes = []
        after_broken = 0
        for idx, r in enumerate(raw_rows):
            dish_name = r.get("dish_name", "").strip()
            dish_id = dish_name.lower().replace(" ", "_").replace("'", "").replace("/", "_")

            # Parse ingredients with quantity using fixed parser
            iq_str = r.get("ingredients_with_quantity", "")
            iq_lines = [x.strip() for x in str(iq_str).split("|") if x.strip()]
            parsed_iq = []
            for line in iq_lines:
                qty, item = parse_ingredient_line_fixed(line)
                if item.lower() == "leaves" or qty.lower().endswith("curry"):
                    after_broken += 1
                parsed_iq.append((qty, item))

            # Non-staple main ingredients
            main_raw = r.get("main_ingredients", "")
            mains_tokens = [canonical_name(x) for x in re.split(r"[|,]", main_raw) if x.strip()]
            non_staple_mains = [m for m in mains_tokens if m not in self.staples]
            if not non_staple_mains:
                non_staple_mains = mains_tokens  # fallback if all were staples

            # Protein estimate per serving
            servings = float(r.get("servings", 2.0)) if str(r.get("servings", "")).replace(".", "").isdigit() else 2.0
            raw_iq_items = [f"{q} {i}".strip() for q, i in parsed_iq]
            try:
                prot_g, _, _ = self.protein_calc.compute_recipe_protein(
                    raw_iq_items, non_staple_mains, servings=servings
                )
            except Exception:
                prot_g = None

            # Local name
            local_name = r.get("romanized_local_name", "").strip() or r.get("native_name", "").strip() or dish_name

            # Status: reviewed only if demo_ready == "yes"
            demo_ready_val = str(r.get("demo_ready", "")).strip().lower()
            recipe_status = "reviewed" if demo_ready_val == "yes" else "unreviewed"

            cleaned_dishes.append({
                "id": dish_id,
                "name": dish_name,
                "local_name": local_name,
                "state": r.get("state", "Maharashtra").strip(),
                "diet": r.get("diet", "vegetarian").strip().lower(),
                "meat_type": str(r.get("meat_type", "none")).strip().lower(),
                "contains_egg": str(r.get("contains_egg", "no")).strip().lower() in ["yes", "true", "1"],
                "time_minutes": int(r.get("time_minutes", 30)) if str(r.get("time_minutes", "")).isdigit() else 30,
                "non_staple_mains": non_staple_mains,
                "all_ingredients_text": (r.get("main_ingredients", "") + " " + iq_str + " " + r.get("pantry_items", "")).lower(),
                "protein_g": prot_g,
                "recipe_status": recipe_status,
                "raw_row": r
            })

        print(f"[DishScorer] Loaded {len(cleaned_dishes)} dishes. Ingredient split bug check: {before_broken} broken rows (legacy parser) -> {after_broken} broken rows (fixed parser).")
        return cleaned_dishes

    def score_dishes(
        self,
        ingredients: List[str],
        pantry: Optional[List[str]] = None,
        state: str = "Maharashtra",
        vegetarian_only: bool = False,
        allergens: Optional[List[str]] = None,
        protein_w: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Scores dishes against input pantry and preferences.
        """
        pantry = pantry or []
        allergens = allergens or []
        user_state = state.strip().title() if state else "Maharashtra"
        effective_protein_w = protein_w if protein_w is not None else self.weights.get("protein_w", 0.0)

        # 1. Canonicalize user inputs
        user_ings = {canonical_name(x) for x in ingredients if x.strip()}
        # Note: staples and pantry items NEVER count as matches
        user_match_pool = {ing for ing in user_ings if ing not in self.staples}

        clean_allergens = [a.lower().strip() for a in allergens if a.strip()]

        eligible_dishes = []

        for dish in self.dishes:
            # Step 1: Vegetarian and allergen filters
            if vegetarian_only:
                if dish["diet"] != "vegetarian" or dish["meat_type"] not in ["none", "nan", ""] or dish["contains_egg"]:
                    continue

            if clean_allergens:
                has_allergen = False
                for a in clean_allergens:
                    if re.search(r"\b" + re.escape(a) + r"\b", dish["all_ingredients_text"]):
                        has_allergen = True
                        break
                if has_allergen:
                    continue

            # Step 2: Eligibility: >= 50% non-staple main ingredients present AND >= 2 matched
            mains = dish["non_staple_mains"]
            if not mains:
                continue

            matched = [m for m in mains if m in user_match_pool]
            missing = [m for m in mains if m not in user_match_pool]

            matched_count = len(matched)
            match_fraction = matched_count / float(len(mains))

            # Eligibility rule: >= 50% AND >= 2 matched
            if matched_count < 2 or match_fraction < 0.50:
                continue

            # Step 3: Scoring formula for eligible dishes
            # 0.45*ingredient_match + 0.25*region + 0.10*diet_time_fit + 0.05*popularity + protein_w*min(protein_g/25,1)
            # Region score
            dish_state = dish["state"].title()
            if dish_state == user_state:
                region_score = 1.0
            elif dish_state in self.neighbours.get(user_state, set()):
                region_score = 0.6
            else:
                region_score = 0.2

            # Diet & time fit
            cook_time = dish["time_minutes"]
            if cook_time <= 30:
                diet_time_fit = 1.0
            else:
                diet_time_fit = max(0.0, 1.0 - (cook_time - 30) / 30.0)

            # Popularity (0.8 default)
            popularity = 0.8

            # Protein boost: No protein boost if ingredient_match < 0.6
            prot_g = dish["protein_g"]
            if match_fraction >= 0.60 and prot_g is not None and effective_protein_w > 0:
                protein_boost = effective_protein_w * min(prot_g / 25.0, 1.0)
            else:
                protein_boost = 0.0

            w_ing = self.weights.get("ingredient_match", 0.45)
            w_reg = self.weights.get("region_weight", 0.25)
            w_dt = self.weights.get("diet_time_fit", 0.10)
            w_pop = self.weights.get("popularity", 0.05)

            final_score = (
                w_ing * match_fraction +
                w_reg * region_score +
                w_dt * diet_time_fit +
                w_pop * popularity +
                protein_boost
            )

            eligible_dishes.append({
                "id": dish["id"],
                "name": dish["name"],
                "local_name": dish["local_name"],
                "score": round(final_score, 4),
                "ingredient_match": round(match_fraction, 4),
                "matched": matched,
                "missing": missing,
                "region_score": round(region_score, 4),
                "protein_g_estimate": round(prot_g, 2) if prot_g is not None else None,
                "recipe_status": dish["recipe_status"]
            })

        # Step 4: Sort by score descending, tie-break by dish id
        eligible_dishes.sort(key=lambda d: (-d["score"], d["id"]))

        # Step 5: Return top 5 or none eligible
        if not eligible_dishes:
            return {
                "dishes": [],
                "message": "No dishes found"
            }

        return {
            "dishes": eligible_dishes[:5],
            "message": f"Found {len(eligible_dishes)} eligible dishes"
        }
