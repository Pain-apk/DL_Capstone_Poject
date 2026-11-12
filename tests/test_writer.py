"""
tests/test_writer.py — Unit tests for Layer 3 (Recipe Writer & Validators).

Tests:
1. Header built from code contains the right dish and scaled quantities.
2. Each validator catches a bad example:
   - JSON invalid
   - Dish mismatch
   - Invented ingredient
   - Scaled quantity mismatch (> 10%)
   - Allergen violation
   - Step count out of range (< 3 or > 10)
   - Time limit violation (> 50% off)
   - Spice blend not a free staple
3. LLM_MODE=off returns stored catalog recipe.
4. Forced model failure (exception / OOM) returns stored catalog recipe without 500.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.writer import RecipeWriter, build_decision_header, format_scaled_quantity
from src.validators import (
    validate_json_schema,
    validate_dish_match,
    validate_allowed_ingredients,
    validate_scaled_quantities,
    validate_no_allergens,
    validate_steps_count,
    validate_time_limit,
    validate_spice_blends,
    validate_recipe
)


class TestRecipeWriterAndValidators(unittest.TestCase):
    def setUp(self):
        self.sample_dish = {
            "id": "kanda_poha",
            "name": "Kanda Poha",
            "local_name": "Kande Pohe",
            "state": "Maharashtra",
            "diet": "vegetarian",
            "time_minutes": 20,
            "raw_row": {
                "dish_name": "Kanda Poha",
                "romanized_local_name": "Kande Pohe",
                "native_name": "कांदे पोहे",
                "state": "Maharashtra",
                "diet": "vegetarian",
                "meat_type": "none",
                "contains_egg": "no",
                "time_minutes": "20",
                "servings": "2",
                "main_ingredients": "poha, onion, roasted peanuts, green chilli",
                "ingredients_with_quantity": "1.5 cup poha|1 medium onion|3 tbsp roasted peanuts|2 green chilli|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/2 tsp turmeric|8 curry leaves|1 tbsp lemon juice|2 tbsp coriander leaves|salt to taste",
                "steps": "Rinse 1.5 cup poha gently in a strainer|Heat 2 tbsp oil and splutter mustard and cumin seeds|Add curry leaves, chilli and onion, saute till soft|Add turmeric, peanuts and salt|Add poha and toss gently for 2 minutes|Garnish with lemon juice and coriander leaves",
                "pantry_items": "oil, mustard seeds, cumin seeds, turmeric, salt",
                "demo_ready": "yes"
            }
        }
        self.writer = RecipeWriter()

    # 1. Header built from code contains the right dish and scaled quantities
    def test_header_built_contains_dish_and_scaled_quantities(self):
        header = build_decision_header(
            dish=self.sample_dish,
            user_ingredients=["poha", "onion"],
            pantry=["oil", "salt"],
            servings=4.0,  # Base servings is 2.0 -> scale = 2.0
            time_limit=25,
            allergens=["dairy"]
        )
        self.assertIn("dish: Kanda Poha", header)
        self.assertIn("servings: 4", header)
        self.assertIn("time limit: 25 minutes", header)
        self.assertIn("avoid list: dairy", header)
        # Scaled quantities: 1.5 cup * 2 = 3 cup, 1 medium * 2 = 2 medium, 3 tbsp * 2 = 6 tbsp
        self.assertIn("poha: 3 cup", header)
        self.assertIn("onion: 2 medium", header)
        self.assertIn("roasted peanuts: 6 tbsp", header)
        self.assertIn("curry leaves: 16", header)
        self.assertIn("END DECISION", header)

    # 2. Each validator catches a bad example
    def test_validator_catches_invalid_json(self):
        bad_json = "{dish_name: 'broken', invalid"
        valid, parsed, err = validate_json_schema(bad_json)
        self.assertFalse(valid)
        self.assertIn("JSON parse error", err)

        bad_schema = {"dish_name": "Kanda Poha"}  # missing ingredients, steps, etc.
        valid2, parsed2, err2 = validate_json_schema(bad_schema)
        self.assertFalse(valid2)
        self.assertIn("Missing required JSON key", err2)

    def test_validator_catches_dish_mismatch(self):
        recipe = {"dish_name": "Surti Locho"}
        valid, err = validate_dish_match(recipe, expected_dish="Kanda Poha")
        self.assertFalse(valid)
        self.assertIn("Dish mismatch", err)

    def test_validator_catches_invented_ingredients(self):
        recipe = {
            "ingredients": [
                {"item": "poha", "quantity": "1.5 cup"},
                {"item": "chicken breast", "quantity": "200 g"}  # Invented / forbidden
            ]
        }
        allowed = {"poha", "onion", "peanuts", "oil", "mustard seeds", "turmeric", "salt"}
        valid, failures = validate_allowed_ingredients(recipe, allowed)
        self.assertFalse(valid)
        self.assertTrue(any("chicken breast" in f for f in failures))

    def test_validator_catches_scaled_quantity_mismatch(self):
        # Expected: poha = 3 cup (numeric 3.0). Got: 6 cup (numeric 6.0, 100% error)
        recipe = {
            "ingredients": [
                {"item": "poha", "quantity": "6 cup"},
                {"item": "onion", "quantity": "2 medium"}
            ]
        }
        expected_quantities = {"poha": "3 cup", "onion": "2 medium"}
        valid, failures = validate_scaled_quantities(recipe, expected_quantities, tolerance=0.10)
        self.assertFalse(valid)
        self.assertTrue(any("Quantity mismatch for 'poha'" in f for f in failures))

    def test_validator_catches_allergen_violation(self):
        recipe = {
            "ingredients": [
                {"item": "poha", "quantity": "1.5 cup"},
                {"item": "paneer", "quantity": "100 g"}  # Dairy allergen
            ],
            "steps": ["Fry paneer in butter"],
            "missing_ingredients": []
        }
        valid, failures = validate_no_allergens(recipe, excluded_allergens=["dairy"])
        self.assertFalse(valid)
        self.assertTrue(any("Allergen violation" in f for f in failures))

    def test_validator_catches_step_count_violations(self):
        # Too few steps (< 3)
        recipe_too_few = {"steps": ["Step 1: cook everything."]}
        valid_few, err_few = validate_steps_count(recipe_too_few)
        self.assertFalse(valid_few)
        self.assertIn("Step count must be 3-10", err_few)

        # Too many steps (> 10)
        recipe_too_many = {"steps": [f"Step {i}: do instruction number {i}" for i in range(12)]}
        valid_many, err_many = validate_steps_count(recipe_too_many)
        self.assertFalse(valid_many)
        self.assertIn("Step count must be 3-10", err_many)

    def test_validator_catches_time_limit_violation(self):
        # Limit = 20 mins. Allowed = 10m to 30m. Recipe time = 45m (> 150%)
        recipe = {"time_minutes": 45}
        valid, err = validate_time_limit(recipe, time_limit=20)
        self.assertFalse(valid)
        self.assertIn("Cooking time 45m not within 50%", err)

    def test_validator_catches_unallowed_spice_blend(self):
        # Pav bhaji masala is a complex spice blend, not a basic staple
        recipe = {
            "ingredients": [
                {"item": "poha", "quantity": "1.5 cup"},
                {"item": "pav bhaji masala", "quantity": "1 tsp"}
            ]
        }
        allowed_blends = set()  # Kanda Poha catalog has no spice blend
        valid, failures = validate_spice_blends(recipe, allowed_blends)
        self.assertFalse(valid)
        self.assertTrue(any("pav bhaji masala" in f for f in failures))

    # 3. LLM_MODE=off returns stored catalog recipe
    def test_llm_mode_off_returns_stored(self):
        res = self.writer.generate(
            dish=self.sample_dish,
            user_ingredients=["poha", "onion"],
            pantry=["oil"],
            servings=4.0,
            mode_override="off"
        )
        self.assertEqual(res["source"], "stored")
        self.assertEqual(res["recipe"]["dish_name"], "Kanda Poha")
        self.assertEqual(res["recipe_status"], "reviewed")
        self.assertTrue(res["validation"]["passed"])
        # Check scaling in stored recipe
        poha_ing = next(i for i in res["recipe"]["ingredients"] if "poha" in i["item"].lower())
        self.assertEqual(poha_ing["quantity"], "3 cup")

    # 4. Forced model failure returns stored catalog recipe
    def test_forced_model_failure_returns_stored(self):
        # Force load failure or generation exception
        with patch.object(self.writer, "generate_raw_llm", side_effect=RuntimeError("Simulated CUDA Out Of Memory")):
            res = self.writer.generate(
                dish=self.sample_dish,
                user_ingredients=["poha", "onion"],
                servings=2.0,
                mode_override="base"
            )
            # Must gracefully fall back to stored recipe without crashing
            self.assertEqual(res["source"], "stored")
            self.assertEqual(res["recipe"]["dish_name"], "Kanda Poha")
            self.assertFalse(res["validation"]["passed"])
            self.assertIn("Simulated CUDA Out Of Memory", str(res["validation"]["failures"]))


if __name__ == "__main__":
    unittest.main()

