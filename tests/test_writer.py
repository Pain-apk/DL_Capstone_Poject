"""
tests/test_writer.py — Unit tests for Layer 3 (v2) Recipe Writer and Step Validators.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from src.writer import RecipeWriter, parse_numbered_steps, format_scaled_quantity
from src.validators import (
    validate_steps,
    validate_steps_count,
    validate_step_word_counts,
    validate_step_ingredients,
    validate_step_allergens,
    validate_step_spice_blends,
    load_catalog_ingredient_vocabulary
)


class TestRecipeWriterV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.writer = RecipeWriter()
        cls.sample_dish = {
            "id": "kanda_poha",
            "name": "Kanda Poha",
            "local_name": "Kande Pohe",
            "time_minutes": 20,
            "raw_row": {
                "dish_name": "Kanda Poha",
                "romanized_local_name": "Kande Pohe",
                "demo_ready": "yes",
                "servings": "2",
                "time_minutes": "20",
                "main_ingredients": "poha|onion|peanuts|green chilli",
                "ingredients_with_quantity": "1.5 cup poha|1 large onion|3 tbsp raw peanuts|2 green chillies|2 tbsp peanut oil|1/2 tsp mustard seeds|1/4 tsp turmeric powder|8 fresh curry leaves|1 tbsp fresh lemon juice|2 tbsp fresh coriander leaves|salt to taste",
                "steps": "Rinse poha in a colander until soft.|Heat oil and fry peanuts until crunchy.|Add mustard seeds, green chillies, and curry leaves.|Add chopped onions and saute until soft.|Fold in turmeric, salt, and drained poha.|Cover and steam on low flame for 2 minutes.|Finish with lemon juice and chopped coriander."
            }
        }
        cls.catalog_vocab = load_catalog_ingredient_vocabulary()

    # 1. Step parser tests
    def test_parse_numbered_steps(self):
        raw = (
            "1. Rinse poha thoroughly under cold running water.\n"
            "2) Heat peanut oil in a pan and fry raw peanuts.\n"
            "Step 3: Crackle mustard seeds and add green chillies.\n"
            "4. Add sliced onions and cook until soft and translucent.\n"
            "5 - Fold in turmeric powder, salt, and drained poha gently."
        )
        steps = parse_numbered_steps(raw)
        self.assertEqual(len(steps), 5)
        self.assertEqual(steps[0], "Rinse poha thoroughly under cold running water.")
        self.assertEqual(steps[1], "Heat peanut oil in a pan and fry raw peanuts.")
        self.assertEqual(steps[2], "Crackle mustard seeds and add green chillies.")
        self.assertEqual(steps[3], "Add sliced onions and cook until soft and translucent.")
        self.assertEqual(steps[4], "Fold in turmeric powder, salt, and drained poha gently.")

    # 2. Step count validator
    def test_validator_catches_step_count_violations(self):
        # Too few (< 3)
        too_few = ["Heat oil in pan.", "Add vegetables and cook."]
        ok, err = validate_steps_count(too_few)
        self.assertFalse(ok)
        self.assertIn("between 3 and 10", err)

        # Too many (> 10)
        too_many = [f"Step number {i} for cooking food item." for i in range(11)]
        ok, err = validate_steps_count(too_many)
        self.assertFalse(ok)
        self.assertIn("between 3 and 10", err)

        # Valid count (4)
        valid = [f"Step number {i} for cooking food item." for i in range(4)]
        ok, err = validate_steps_count(valid)
        self.assertTrue(ok)

    # 3. Word count per step (5-40 words)
    def test_validator_catches_word_count_violations(self):
        # Step too short (< 5 words)
        short_steps = [
            "Rinse the poha thoroughly now.",
            "Heat oil.",
            "Cook vegetables gently until soft.",
            "Serve hot with coriander."
        ]
        ok, failures = validate_step_word_counts(short_steps)
        self.assertFalse(ok)
        self.assertTrue(any("only 2 words" in f for f in failures))

        # Step too long (> 40 words)
        long_step = " ".join(["word"] * 45)
        long_steps = [
            "Rinse the poha thoroughly under water.",
            long_step,
            "Cook vegetables gently until tender.",
            "Serve warm with fresh lemon juice."
        ]
        ok, failures = validate_step_word_counts(long_steps)
        self.assertFalse(ok)
        self.assertTrue(any("exceeds 40 words" in f for f in failures))

    # 4. Unallowed catalog ingredient in steps
    def test_validator_catches_unallowed_ingredient(self):
        allowed = {"poha", "onion", "peanuts", "oil", "mustard seeds", "salt", "turmeric"}
        bad_steps = [
            "Rinse poha thoroughly under cold running water.",
            "Heat oil in a pan and fry raw peanuts.",
            "Add finely chopped potatoes and saute until tender.",  # potato not allowed
            "Fold in drained poha and mix gently."
        ]
        ok, failures = validate_step_ingredients(bad_steps, allowed, self.catalog_vocab)
        self.assertFalse(ok)
        self.assertTrue(any("potato" in f for f in failures))

    # 5. Allergen exclusion in steps
    def test_validator_catches_allergen_violation(self):
        steps_with_dairy = [
            "Rinse poha thoroughly under cold running water.",
            "Melt 2 tablespoons of butter in a pan over medium heat.",  # butter is dairy
            "Add mustard seeds and saute chopped onions until soft.",
            "Serve hot garnished with fresh coriander."
        ]
        ok, failures = validate_step_allergens(steps_with_dairy, excluded_allergens=["dairy"])
        self.assertFalse(ok)
        self.assertTrue(any("butter" in f for f in failures))

    # 6. Unallowed spice blends in steps
    def test_validator_catches_unallowed_spice_blend(self):
        steps_with_blend = [
            "Rinse poha thoroughly under cold running water.",
            "Heat oil in a pan and crackle mustard seeds.",
            "Stir in one teaspoon of pav bhaji masala for flavor.",  # pav bhaji masala not allowed
            "Fold in drained poha and serve hot."
        ]
        ok, failures = validate_step_spice_blends(steps_with_blend, allowed_catalog_blends=set())
        self.assertFalse(ok)
        self.assertTrue(any("pav bhaji masala" in f for f in failures))

    # 7. Valid steps pass master validation cleanly
    def test_valid_steps_pass_cleanly(self):
        allowed = {"poha", "onion", "peanuts", "oil", "mustard seeds", "green chillies", "salt", "turmeric", "curry leaves", "coriander leaves", "lemon juice", "water"}
        valid_steps = [
            "Rinse poha thoroughly under running water until softened.",
            "Heat peanut oil in a pan and fry raw peanuts until crunchy.",
            "Add mustard seeds, green chillies, and curry leaves to the pan.",
            "Add finely chopped onions and saute slowly until translucent.",
            "Fold in turmeric powder, salt, and drained poha gently.",
            "Finish with fresh lemon juice and chopped coriander leaves."
        ]
        passed, failures = validate_steps(
            steps=valid_steps,
            allowed_ingredients=allowed,
            catalog_vocab=self.catalog_vocab,
            excluded_allergens=["dairy"],
            allowed_catalog_blends=set()
        )
        self.assertTrue(passed)
        self.assertEqual(len(failures), 0)

    # 8. Quantity scaling helper
    def test_format_scaled_quantity(self):
        self.assertEqual(format_scaled_quantity("1.5 cup", 2.0), "3 cup")
        self.assertEqual(format_scaled_quantity("1/2 tsp", 2.0), "1 tsp")
        self.assertEqual(format_scaled_quantity("to taste", 4.0), "to taste")

    # 9. LLM_MODE=off returns stored recipe
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
        poha_ing = next(i for i in res["recipe"]["ingredients"] if "poha" in i["item"].lower())
        self.assertEqual(poha_ing["quantity"], "3 cup")

    # 10. Forced model failure returns stored recipe gracefully (never 500)
    def test_forced_model_failure_returns_stored(self):
        with patch.object(self.writer, "generate_raw_steps", side_effect=RuntimeError("Simulated CUDA Out Of Memory")):
            res = self.writer.generate(
                dish=self.sample_dish,
                user_ingredients=["poha", "onion"],
                servings=2.0,
                mode_override="base"
            )
            self.assertEqual(res["source"], "stored")
            self.assertEqual(res["recipe"]["dish_name"], "Kanda Poha")
            self.assertFalse(res["validation"]["passed"])
            self.assertIn("Simulated CUDA Out Of Memory", str(res["validation"]["failures"]))


if __name__ == "__main__":
    unittest.main()
