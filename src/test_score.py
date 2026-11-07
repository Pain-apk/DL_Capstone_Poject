"""
test_score.py - Comprehensive verification of scoring and protein boosting.
Tests 10 scenarios and outputs Top 5 comparison for Protein Off (0.0) vs Protein High (0.25).
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.score import RecipeScorer

FIXTURE_PATH = os.path.join(PROJECT_ROOT, "tests", "fixtures", "sample_dishes.json")
with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
    SAMPLE_DISH_CATALOG = json.load(f)



def run_tests():
    scorer = RecipeScorer()
    print("=" * 80)
    print("RUNNING 10 TEST CASES FOR RECIPE SCORER")
    print("=" * 80)

    # 1. Diet Filter Test (Non-veg filtered when vegetarian requested)
    t1 = scorer.filter_candidates(SAMPLE_DISH_CATALOG, user_diet="vegetarian", excluded_allergens=[])
    assert all(d["diet"] in ["vegetarian", "vegan"] for d in t1), "Test 1 Failed: Non-veg not filtered!"
    print("[PASS] Test 1: Diet filter correctly removes non-vegetarian dishes.")

    # 2. Allergen Filter Test (Peanut allergy excludes Kanda Poha)
    t2 = scorer.filter_candidates(SAMPLE_DISH_CATALOG, user_diet="vegetarian", excluded_allergens=["peanuts"])
    assert not any("peanuts" in d["dish_name"].lower() or "poha" in d["dish_name"].lower() for d in t2), "Test 2 Failed!"
    print("[PASS] Test 2: Allergen filter removes peanut-containing dishes (Kanda Poha).")

    # 3. Same state region weight
    w_same = scorer.compute_region_weight("Maharashtra", "Maharashtra", "no")
    assert w_same == 1.0, f"Test 3 Failed: expected 1.0, got {w_same}"
    print("[PASS] Test 3: Same state gets region_weight = 1.0.")

    # 4. Neighbour state region weight
    w_nb = scorer.compute_region_weight("Gujarat", "Maharashtra", "no")
    assert w_nb == 0.6, f"Test 4 Failed: expected 0.6, got {w_nb}"
    print("[PASS] Test 4: Neighbour state gets region_weight = 0.6.")

    # 5. Distant non-pan-India region weight
    w_dist = scorer.compute_region_weight("Punjab", "Maharashtra", "no")
    assert w_dist == 0.2, f"Test 5 Failed: expected 0.2, got {w_dist}"
    print("[PASS] Test 5: Distant state gets region_weight = 0.2.")

    # 6. Ingredient Match calculation
    match_val = scorer.compute_ingredient_match(["onion", "potato", "poha"], ["poha", "onion", "potato", "peanuts"])
    assert match_val == 0.75, f"Test 6 Failed: expected 0.75, got {match_val}"
    print(f"[PASS] Test 6: Ingredient match ratio calculated correctly ({match_val}).")

    # 7. Protein boost threshold rule: match < 0.6 => effective_protein_w == 0.0
    dish_low_match = SAMPLE_DISH_CATALOG[2] # Matki Usal
    score_res = scorer.score_dish(dish_low_match, detected_ingredients=["onion"], protein_w=0.25)
    assert score_res["effective_protein_w"] == 0.0, "Test 7 Failed: Protein boost allowed despite match < 0.6!"
    print("[PASS] Test 7: Protein boost strictly suppressed when ingredient_match < 0.6.")

    # 8. Protein boost applied when match >= 0.6
    dish_high_match = SAMPLE_DISH_CATALOG[1] # Pithla
    score_res2 = scorer.score_dish(dish_high_match, detected_ingredients=["besan", "onion", "garlic", "green chilli"], protein_w=0.25)
    assert score_res2["effective_protein_w"] == 0.25, "Test 8 Failed: Protein boost not applied when match >= 0.6!"
    print(f"[PASS] Test 8: Protein boost active (w={score_res2['effective_protein_w']}) when ingredient_match >= 0.6.")

    # 9. Time limit penalty
    fit_ontime = scorer.compute_diet_time_fit(20, 30)
    fit_overtime = scorer.compute_diet_time_fit(45, 30)
    assert fit_ontime == 1.0 and fit_overtime < 1.0, "Test 9 Failed!"
    print(f"[PASS] Test 9: Time limit penalty applied (ontime={fit_ontime}, overtime={fit_overtime:.2f}).")

    # 10. Protein sanity check integration
    calc_besan = scorer.protein_calc.nutrition_db.get_protein_per_100g("besan")
    assert calc_besan and 21.0 <= calc_besan[0] <= 23.5, "Test 10 Failed: Besan nutrition data corrupt!"
    print(f"[PASS] Test 10: Integrated nutrition database returns verified Besan protein ({calc_besan[0]}g).")

    print("\nALL 10 TESTS PASSED!\n")

    # =========================================================================
    # DEMONSTRATION: Top 5 for identical ingredients: Protein Off vs Protein High
    # =========================================================================
    detected = ["onion", "besan", "potato", "poha", "peanuts", "green chilli", "garlic"]
    print("-" * 80)
    print(f"INPUT DETECTED INGREDIENTS: {detected}")
    print("User State: Maharashtra | Diet: Vegetarian | Time Limit: 30 mins")
    print("-" * 80)

    top_off = scorer.rank_candidates(
        SAMPLE_DISH_CATALOG,
        detected_ingredients=detected,
        user_state="Maharashtra",
        user_diet="vegetarian",
        time_limit=30,
        protein_w=0.0,
        top_k=5
    )

    top_high = scorer.rank_candidates(
        SAMPLE_DISH_CATALOG,
        detected_ingredients=detected,
        user_state="Maharashtra",
        user_diet="vegetarian",
        time_limit=30,
        protein_w=0.25,
        top_k=5
    )

    print("\n--- TOP 5 RANKINGS: PROTEIN OFF (protein_w = 0.0) ---")
    print(f"{'Rank':<5}{'Dish Name':<20}{'State':<15}{'Score':<10}{'Match':<8}{'Protein/srv':<12}")
    for idx, d in enumerate(top_off, 1):
        print(f"{idx:<5}{d['dish_name']:<20}{d['state']:<15}{d['final_score']:<10.4f}{d['ingredient_match']:<8.2f}{d['protein_g_per_serving']:<6.2f}g")

    print("\n--- TOP 5 RANKINGS: PROTEIN HIGH (protein_w = 0.25) ---")
    print(f"{'Rank':<5}{'Dish Name':<20}{'State':<15}{'Score':<10}{'Match':<8}{'Protein/srv':<12}{'Effective w':<12}")
    for idx, d in enumerate(top_high, 1):
        print(f"{idx:<5}{d['dish_name']:<20}{d['state']:<15}{d['final_score']:<10.4f}{d['ingredient_match']:<8.2f}{d['protein_g_per_serving']:<6.2f}g     {d['effective_protein_w']:<12.2f}")


if __name__ == "__main__":
    run_tests()

