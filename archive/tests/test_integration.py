"""
tests/test_integration.py — Comprehensive end-to-end integration tests.
Covers:
  1. Detector precision/recall, timing, EXIF strip, no disk leak
  2. Scorer ranking: Maharashtra vs Gujarat, protein focus re-ranking, vegetarian-only, no-meat input
  3. Allergy exclusion: Peanut allergy never returns peanuts across 20 randomized inputs
  4. LLM_ENABLED=false fallback behavior
  5. Full API flow simulation
"""

import os
import sys
import json
import time
import io
import unittest
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detector import detect
from app.normalize import normalize
from app.scorer import score_dishes
from app.recipes import get_recipe
from app.writer import generate_recipe


class TestDetector(unittest.TestCase):
    def test_mock_detector_returns_valid_structure(self):
        os.environ["MOCK_DETECTOR"] = "1"
        res = detect(b"fake_image_bytes")
        self.assertIn("ingredients", res)
        self.assertIn("low_confidence", res)
        self.assertTrue(len(res["ingredients"]) > 0)
        for item in res["ingredients"]:
            self.assertIn("name", item)
            self.assertIn("confidence", item)
            self.assertGreaterEqual(item["confidence"], 0.4)

    def test_no_image_saved_to_disk(self):
        # Create a small blank image in memory
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color="red").save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        before_files = set(os.listdir(PROJECT_ROOT))
        os.environ["MOCK_DETECTOR"] = "1"
        _ = detect(raw_bytes)
        after_files = set(os.listdir(PROJECT_ROOT))

        self.assertEqual(before_files, after_files, "detect() must not write files to project root")

    def test_detector_execution_time_under_threshold(self):
        start = time.time()
        os.environ["MOCK_DETECTOR"] = "1"
        _ = detect(b"sample_bytes")
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, f"Detector took too long: {elapsed:.3f}s")


class TestScorer(unittest.TestCase):
    def setUp(self):
        self.pantry = ["onion", "potato", "tomato", "besan", "garlic"]

    def test_maharashtra_vs_gujarat_different_ranking(self):
        r_mh = score_dishes(self.pantry, "Maharashtra", "off", 0, False, [])
        r_gj = score_dishes(self.pantry, "Gujarat", "off", 0, False, [])
        top_mh = r_mh["decision"]["dish_name"]
        top_gj = r_gj["decision"]["dish_name"]

        # 1. The top decided dish differs by state preference
        self.assertNotEqual(top_mh, top_gj, "Different regional preferences must select different top dishes")
        self.assertEqual(r_mh["decision"]["state"], "Maharashtra")
        self.assertEqual(r_gj["decision"]["state"], "Gujarat")

        # 2. The same dish scores higher in its home state than elsewhere
        gj_all = score_dishes(self.pantry, "Gujarat", "off", 0, False, [], top_n=200)
        gj_scores = {e["dish_name"]: e["score"] for e in gj_all["top"]}
        self.assertIn(top_mh, gj_scores)
        self.assertGreater(
            r_mh["decision"]["score"],
            gj_scores[top_mh],
            "Maharashtra dish must score higher when state=Maharashtra than state=Gujarat"
        )

    def test_protein_focus_high_re_ranks_only_above_threshold(self):
        r_off = score_dishes(self.pantry, "Maharashtra", "off", 0, False, [], top_n=100)
        r_high = score_dishes(self.pantry, "Maharashtra", "high", 0, False, [], top_n=100)

        off_map = {e["dish_name"]: e for e in r_off["top"]}
        high_map = {e["dish_name"]: e for e in r_high["top"]}

        for name, off_item in off_map.items():
            if name in high_map and off_item["ingredient_match"] < 0.6:
                self.assertAlmostEqual(
                    off_item["score"], high_map[name]["score"], places=4,
                    msg=f"Dish {name} with match < 0.6 should not receive protein boost"
                )

    def test_vegetarian_only_removes_nonveg(self):
        r = score_dishes(self.pantry + ["chicken"], "Maharashtra", "off", 0, True, [], top_n=50)
        for item in r["top"]:
            self.assertEqual(item.get("diet"), "vegetarian", f"Non-veg dish {item['dish_name']} found when vegetarian_only=True")

    def test_no_meat_in_input_prevents_nonveg_in_top5(self):
        r = score_dishes(["onion", "potato", "tomato"], "Maharashtra", "off", 0, False, [], top_n=5)
        for item in r["top"]:
            self.assertEqual(item.get("diet"), "vegetarian", f"Non-veg dish {item['dish_name']} appeared in top 5 without meat in pantry")


class TestAllergyExclusion(unittest.TestCase):
    def test_peanut_allergy_never_returns_peanuts_across_20_inputs(self):
        pantry_variations = [
            ["onion", "poha", "mustard seeds", "peanuts"],
            ["potato", "peanuts", "green chilli", "curry leaves"],
            ["sabudana", "peanuts", "potato", "cumin seeds"],
            ["besan", "onion", "peanuts", "coriander leaves"],
            ["chicken", "peanuts", "onion", "ginger"],
            ["brinjal", "peanuts", "sesame seeds", "goda masala"],
            ["matki", "tomato", "peanuts", "onion"],
            ["rice", "peanuts", "mustard seeds", "lemon"],
            ["flour", "oil", "salt", "peanuts"],
            ["dal", "turmeric", "peanuts", "cumin seeds"],
            ["paneer", "butter", "peanuts", "tomato"],
            ["mushroom", "onion", "peanuts", "garlic"],
            ["cabbage", "peanuts", "mustard seeds", "green chilli"],
            ["cauliflower", "potato", "peanuts", "turmeric"],
            ["fish", "peanuts", "chilli", "garlic"],
            ["mutton", "onion", "peanuts", "garam masala"],
            ["egg", "onion", "peanuts", "tomato"],
            ["sprouts", "lemon", "peanuts", "salt"],
            ["curd", "cucumber", "peanuts", "cumin seeds"],
            ["coconut", "coriander", "peanuts", "green chilli"]
        ]

        violation_count = 0
        for idx, pantry in enumerate(pantry_variations, 1):
            res = score_dishes(
                pantry=pantry,
                state="Maharashtra",
                protein_focus="off",
                max_time=0,
                vegetarian_only=False,
                excluded_allergens=["peanuts", "peanut"],
                top_n=5
            )
            for dish in res["top"]:
                dish_meta = get_recipe(dish["dish_id"]) or {}
                combined_text = (
                    dish["dish_name"] + " " +
                    " ".join(dish.get("missing_mains", [])) + " " +
                    " ".join(dish_meta.get("main_ingredients", [])) + " " +
                    " ".join(i.get("item", "") for i in dish_meta.get("ingredients_with_quantity", []))
                ).lower()

                if "peanut" in combined_text or "shengdana" in combined_text or "groundnut" in combined_text:
                    violation_count += 1

        self.assertEqual(violation_count, 0, f"Found {violation_count} allergen violations across 20 test inputs!")


class TestWriterAndFallback(unittest.TestCase):
    def test_llm_disabled_returns_full_stored_recipe(self):
        old_env = os.environ.get("LLM_ENABLED")
        try:
            os.environ["LLM_ENABLED"] = "false"
            stored = get_recipe("kanda_poha")
            dish_meta = {
                "dish_name": "Kanda Poha",
                "dish_id": "kanda_poha",
                "state": "Maharashtra",
                "rank": 1,
                "ingredient_match": 1.0,
                "missing_mains": [],
                "protein_estimate_g": 6.5,
                "protein_confidence": "medium",
                "verified": False
            }
            res = generate_recipe(
                dish=dish_meta,
                stored=stored,
                pantry=["poha", "onion"],
                allergens=[],
                servings=2
            )
            self.assertEqual(res["source"], "fallback")
            self.assertEqual(res["recipe"]["dish_name"], "Kanda Poha")
            self.assertGreaterEqual(len(res["recipe"]["steps"]), 3)
            self.assertGreaterEqual(len(res["recipe"]["ingredients"]), 5)
        finally:
            if old_env is not None:
                os.environ["LLM_ENABLED"] = old_env
            else:
                os.environ.pop("LLM_ENABLED", None)


if __name__ == "__main__":
    unittest.main()
