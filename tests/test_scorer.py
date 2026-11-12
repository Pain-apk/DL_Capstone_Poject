import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scorer import DishScorer

class TestScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = DishScorer()

    def test_kanda_poha_absent_for_tomato_cucumber(self):
        """Kanda Poha must NOT appear for tomato + cucumber."""
        res = self.scorer.score_dishes(["tomato", "cucumber"])
        dish_names = [d["name"].lower() for d in res["dishes"]]
        self.assertNotIn("kanda poha", dish_names)
        self.assertNotIn("batata poha", dish_names)

    def test_10_different_input_sets_give_ge_5_top1(self):
        """10 different input sets must yield at least 5 different top-1 dishes."""
        input_sets = [
            ["poha", "onion", "peanuts"],
            ["potato", "onion", "tomato"],
            ["brinjal", "onion", "tomato"],
            ["besan", "onion", "green chilli"],
            ["rice", "green peas", "onion"],
            ["cauliflower", "potato", "onion"],
            ["chicken", "onion", "tomato"],
            ["cucumber", "curd", "green chilli"],
            ["paneer", "green peas", "tomato"],
            ["carrot", "green peas", "potato"]
        ]
        top_dishes = set()
        for ings in input_sets:
            res = self.scorer.score_dishes(ings, vegetarian_only=False)
            if res["dishes"]:
                top_dishes.add(res["dishes"][0]["name"])

        print(f"\nUnique top-1 dishes count: {len(top_dishes)}: {top_dishes}")
        self.assertGreaterEqual(len(top_dishes), 5)

    def test_pantry_only_never_eligible(self):
        """Pantry-only input never makes any dish eligible."""
        res1 = self.scorer.score_dishes([], pantry=["salt", "oil", "turmeric", "water"])
        self.assertEqual(res1["dishes"], [])
        self.assertEqual(res1["message"], "No dishes found")

        res2 = self.scorer.score_dishes([], pantry=["onion", "potato", "poha", "salt"])
        self.assertEqual(res2["dishes"], [])
        self.assertEqual(res2["message"], "No dishes found")

    def test_vegetarian_only_removes_non_veg(self):
        """vegetarian_only removes all non-vegetarian dishes."""
        res_veg = self.scorer.score_dishes(["chicken", "onion", "tomato"], vegetarian_only=True)
        for d in res_veg["dishes"]:
            dish_entry = next((item for item in self.scorer.dishes if item["id"] == d["id"]), None)
            self.assertIsNotNone(dish_entry)
            self.assertEqual(dish_entry["diet"], "vegetarian")
            self.assertIn(dish_entry["meat_type"], ["none", "nan", ""])
            self.assertFalse(dish_entry["contains_egg"])

    def test_allergen_removes_dishes(self):
        """An allergen removes every dish containing it."""
        res_no_allergen = self.scorer.score_dishes(["poha", "onion", "peanuts"], allergens=[])
        self.assertTrue(any("poha" in d["name"].lower() for d in res_no_allergen["dishes"]))

        res_allergen = self.scorer.score_dishes(["poha", "onion", "peanuts"], allergens=["peanuts"])
        for d in res_allergen["dishes"]:
            self.assertNotIn("poha", d["name"].lower())
            dish_entry = next((item for item in self.scorer.dishes if item["id"] == d["id"]), None)
            self.assertNotIn("peanuts", dish_entry["all_ingredients_text"])

    def test_deterministic_identical_output(self):
        """Same input twice gives exact identical output."""
        inp = ["potato", "onion", "green chilli"]
        res1 = self.scorer.score_dishes(inp, state="Maharashtra", vegetarian_only=True)
        res2 = self.scorer.score_dishes(inp, state="Maharashtra", vegetarian_only=True)
        self.assertEqual(res1, res2)

    def test_empty_or_garbage_input_no_crash(self):
        """Empty or garbage input gives 'No dishes found' without crashing."""
        res_empty = self.scorer.score_dishes([])
        self.assertEqual(res_empty["dishes"], [])
        self.assertEqual(res_empty["message"], "No dishes found")

        res_garbage = self.scorer.score_dishes(["unknown_food_xyz_123", "random_rock_456"])
        self.assertEqual(res_garbage["dishes"], [])
        self.assertEqual(res_garbage["message"], "No dishes found")

if __name__ == "__main__":
    unittest.main(verbosity=2)
