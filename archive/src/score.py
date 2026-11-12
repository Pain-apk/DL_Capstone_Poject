"""
score.py - Candidate dish scoring module.
Formula:
  score = 0.45*ingredient_match + 0.25*region_weight + 0.10*diet_time_fit + 0.05*popularity + protein_w*protein_score
  protein_score = min(protein_g_per_serving / 25.0, 1.0)
  protein_w is 0.0, 0.15, or 0.25

Rules:
- Diet and allergy filters run strictly BEFORE scoring.
- No protein boost if ingredient_match < 0.6 (protein_w forced to 0.0).
- All weights editable via config.json.
"""

import os
import sys
import json
import csv
from typing import List, Dict, Any, Optional, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.protein import ProteinCalculator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
NEIGHBOURS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "state_neighbours.csv")


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "weights": {
            "ingredient_match": 0.45,
            "region_weight": 0.25,
            "diet_time_fit": 0.10,
            "popularity": 0.05,
            "protein_w": 0.0
        },
        "protein_target_g": 25.0,
        "protein_boost_ingredient_threshold": 0.60,
        "default_state": "Maharashtra",
        "pantry_items": ["oil", "salt", "water", "turmeric", "mustard seeds", "cumin seeds"]
    }


def load_neighbours() -> Dict[str, Set[str]]:
    neighbours: Dict[str, Set[str]] = {}
    if os.path.exists(NEIGHBOURS_PATH):
        with open(NEIGHBOURS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                st = row["state"].strip()
                nb = row["neighbour"].strip()
                if st not in neighbours:
                    neighbours[st] = set()
                neighbours[st].add(nb)
    return neighbours


class RecipeScorer:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or load_config()
        self.weights = self.config.get("weights", {})
        self.protein_target = self.config.get("protein_target_g", 25.0)
        self.protein_threshold = self.config.get("protein_boost_ingredient_threshold", 0.60)
        self.pantry = set(self.config.get("pantry_items", []))
        self.neighbours = load_neighbours()
        self.protein_calc = ProteinCalculator()

    def filter_candidates(self,
                          candidates: List[Dict[str, Any]],
                          user_diet: str,
                          excluded_allergens: List[str]) -> List[Dict[str, Any]]:
        """
        Hard constraints: Filter out recipes that violate diet or contain allergens.
        Runs strictly BEFORE scoring.
        """
        allowed = []
        user_diet_clean = user_diet.lower().strip()
        allergens_clean = [a.lower().strip() for a in excluded_allergens if a.strip()]

        for dish in candidates:
            dish_diet = dish.get("diet", "").lower().strip()

            # Diet filtering
            if user_diet_clean == "vegetarian" and dish_diet not in ["vegetarian", "vegan"]:
                continue
            if user_diet_clean == "vegan" and dish_diet != "vegan":
                continue

            # Allergen filtering
            all_ingredients_text = (
                dish.get("main_ingredients", "") + " " +
                dish.get("ingredients_with_quantity", "") + " " +
                dish.get("pantry_items", "")
            ).lower()

            violated = False
            for allergen in allergens_clean:
                if allergen in all_ingredients_text:
                    violated = True
                    break
            if violated:
                continue

            allowed.append(dish)

        return allowed

    def compute_ingredient_match(self,
                                 detected_ingredients: List[str],
                                 dish_main_ingredients: List[str]) -> float:
        """
        Calculates fraction of dish main ingredients satisfied by detected ingredients.
        Pantry items are excluded from required ingredients.
        """
        norm_detected = {self.protein_calc.normalize_name(i) for i in detected_ingredients}
        main_list = [self.protein_calc.normalize_name(i) for i in dish_main_ingredients if i.strip()]
        
        # Filter out pantry items from denominator if present
        essential_mains = [m for m in main_list if m not in self.pantry]
        if not essential_mains:
            essential_mains = main_list
        if not essential_mains:
            return 1.0

        matches = 0
        for m in essential_mains:
            if any(m in d or d in m for d in norm_detected):
                matches += 1

        return min(1.0, matches / len(essential_mains))

    def compute_region_weight(self, dish_state: str, user_state: str, pan_india: str) -> float:
        """
        Same state = 1.0
        Neighbour state = 0.6
        Pan-India = 0.5
        Other = 0.2
        """
        d_state = dish_state.strip().title()
        u_state = user_state.strip().title()

        if d_state == u_state:
            return 1.0
        if d_state in self.neighbours.get(u_state, set()):
            return 0.6
        if pan_india.strip().lower() in ["yes", "true", "1"]:
            return 0.5
        return 0.2

    def compute_diet_time_fit(self, cook_time: int, time_limit: Optional[int]) -> float:
        if not time_limit or time_limit <= 0:
            return 1.0
        if cook_time <= time_limit:
            return 1.0
        diff = cook_time - time_limit
        return max(0.0, 1.0 - (diff / float(time_limit)))

    def score_dish(self,
                   dish: Dict[str, Any],
                   detected_ingredients: List[str],
                   user_state: str = "Maharashtra",
                   time_limit: Optional[int] = 30,
                   protein_w: float = 0.0) -> Dict[str, Any]:

        # Parse ingredients
        main_ings = [i.strip() for i in dish.get("main_ingredients", "").split("|") if i.strip()]
        raw_ings_qty = [i.strip() for i in dish.get("ingredients_with_quantity", "").split("|") if i.strip()]

        # Compute protein
        servings = float(dish.get("servings", 2.0))
        prot_per_serving, confidence, breakdown = self.protein_calc.compute_recipe_protein(
            raw_ings_qty, main_ings, servings=servings
        )

        # 1. Ingredient Match
        ing_match = self.compute_ingredient_match(detected_ingredients, main_ings)

        # 2. Region Weight
        reg_weight = self.compute_region_weight(
            dish.get("state", "Maharashtra"),
            user_state,
            dish.get("pan_india", "no")
        )

        # 3. Diet & Time Fit
        cook_time = int(dish.get("time_minutes", 30))
        dt_fit = self.compute_diet_time_fit(cook_time, time_limit)

        # 4. Popularity
        popularity = float(dish.get("popularity", 0.8))

        # 5. Protein Score
        protein_score = min(prot_per_serving / self.protein_target, 1.0)

        # Rule: no protein boost if ingredient_match < 0.6
        effective_protein_w = protein_w if ing_match >= self.protein_threshold else 0.0

        w_ing = self.weights.get("ingredient_match", 0.45)
        w_reg = self.weights.get("region_weight", 0.25)
        w_dt = self.weights.get("diet_time_fit", 0.10)
        w_pop = self.weights.get("popularity", 0.05)

        final_score = (
            w_ing * ing_match +
            w_reg * reg_weight +
            w_dt * dt_fit +
            w_pop * popularity +
            effective_protein_w * protein_score
        )

        return {
            "dish_name": dish.get("dish_name", ""),
            "local_name": dish.get("local_name", ""),
            "state": dish.get("state", "Maharashtra"),
            "final_score": round(final_score, 4),
            "ingredient_match": round(ing_match, 4),
            "region_weight": round(reg_weight, 4),
            "diet_time_fit": round(dt_fit, 4),
            "popularity": round(popularity, 4),
            "protein_g_per_serving": prot_per_serving,
            "protein_score": round(protein_score, 4),
            "protein_confidence": confidence,
            "effective_protein_w": effective_protein_w,
            "raw_dish": dish
        }

    def rank_candidates(self,
                        candidates: List[Dict[str, Any]],
                        detected_ingredients: List[str],
                        user_state: str = "Maharashtra",
                        user_diet: str = "vegetarian",
                        time_limit: Optional[int] = 30,
                        excluded_allergens: Optional[List[str]] = None,
                        protein_w: float = 0.0,
                        top_k: int = 5) -> List[Dict[str, Any]]:

        allergens = excluded_allergens or []
        # Pre-scoring hard filters
        filtered = self.filter_candidates(candidates, user_diet, allergens)

        scored = []
        for dish in filtered:
            res = self.score_dish(
                dish,
                detected_ingredients=detected_ingredients,
                user_state=user_state,
                time_limit=time_limit,
                protein_w=protein_w
            )
            scored.append(res)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]

