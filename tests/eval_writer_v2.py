"""
tests/eval_writer_v2.py — Evaluation benchmark for Layer 3 (v2) Recipe Step Writer.
Evaluates 20 catalog cases across 2 configurations:
(a) Stored steps as control
(b) LLM steps (base Qwen2.5-1.5B-Instruct in native bf16/fp16, greedy decoding)

Per config:
- Validation pass %
- Failures by type
- Allergen violations
- Invented-ingredient steps
- Average and max seconds per recipe
- Reviewer rating (left empty for human review)

Saves results to docs/writer_eval_v2.json.
Thresholds for (b):
  Validation >= 80%, 0 allergen violations, 0 invented ingredients, average <= 10s per recipe.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scorer import DishScorer, canonical_name
from src.writer import RecipeWriter, parse_numbered_steps
from src.validators import (
    validate_steps,
    load_catalog_ingredient_vocabulary,
    SPICE_BLENDS,
    normalize_item_name
)

# 20 Diverse catalog cases (identical to eval_writer.py)
EVAL_CASES = [
    {
        "dish_id": "kanda_poha",
        "ingredients": ["poha", "onion", "peanuts", "green chilli"],
        "pantry": ["oil", "mustard seeds", "turmeric", "salt"],
        "servings": 2.0,
        "time_limit": 25,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "batata_poha",
        "ingredients": ["poha", "potato", "onion", "green chilli"],
        "pantry": ["oil", "mustard seeds", "cumin seeds", "turmeric", "salt"],
        "servings": 4.0,
        "time_limit": 30,
        "allergens": ["gluten"]
    },
    {
        "dish_id": "dadpe_pohe",
        "ingredients": ["poha", "onion", "coconut", "lemon", "green chilli"],
        "pantry": ["oil", "mustard seeds", "hing", "salt"],
        "servings": 2.0,
        "time_limit": 20,
        "allergens": ["peanuts"]
    },
    {
        "dish_id": "misal_pav",
        "ingredients": ["sprouted moth beans", "onion", "tomato", "ginger", "garlic"],
        "pantry": ["oil", "mustard seeds", "red chilli powder", "turmeric", "salt"],
        "servings": 4.0,
        "time_limit": 45,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "puneri_misal",
        "ingredients": ["sprouted matki", "onion", "tomato", "ginger", "garlic", "chilli"],
        "pantry": ["oil", "mustard seeds", "turmeric", "salt"],
        "servings": 3.0,
        "time_limit": 40,
        "allergens": ["soy"]
    },
    {
        "dish_id": "batata_vada",
        "ingredients": ["potato", "besan", "green chilli", "ginger", "garlic"],
        "pantry": ["oil", "mustard seeds", "turmeric", "hing", "salt"],
        "servings": 4.0,
        "time_limit": 35,
        "allergens": ["gluten"]
    },
    {
        "dish_id": "vada_pav",
        "ingredients": ["potato", "besan", "pav", "garlic", "green chilli"],
        "pantry": ["oil", "mustard seeds", "turmeric", "salt"],
        "servings": 4.0,
        "time_limit": 40,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "sabudana_vada",
        "ingredients": ["sabudana", "potato", "peanuts", "green chilli"],
        "pantry": ["oil", "cumin seeds", "salt", "sugar"],
        "servings": 3.0,
        "time_limit": 30,
        "allergens": ["gluten"]
    },
    {
        "dish_id": "sabudana_khichdi",
        "ingredients": ["sabudana", "potato", "roasted peanuts", "green chilli"],
        "pantry": ["ghee", "cumin seeds", "salt", "sugar"],
        "servings": 2.0,
        "time_limit": 25,
        "allergens": ["peanuts"]
    },
    {
        "dish_id": "thalipeeth",
        "ingredients": ["bhajani flour", "onion", "coriander leaves", "green chilli"],
        "pantry": ["oil", "turmeric", "salt", "cumin seeds"],
        "servings": 3.0,
        "time_limit": 30,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "upma___sanja",
        "ingredients": ["rava", "onion", "green chilli", "ginger"],
        "pantry": ["oil", "mustard seeds", "curry leaves", "salt"],
        "servings": 2.0,
        "time_limit": 25,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "kothimbir_vadi",
        "ingredients": ["coriander leaves", "besan", "green chilli", "ginger", "garlic"],
        "pantry": ["oil", "turmeric", "sesame seeds", "salt"],
        "servings": 4.0,
        "time_limit": 40,
        "allergens": ["gluten"]
    },
    {
        "dish_id": "alu_vadi___patra",
        "ingredients": ["colocasia leaves", "besan", "tamarind", "jaggery", "green chilli"],
        "pantry": ["oil", "mustard seeds", "sesame seeds", "hing", "salt"],
        "servings": 2.0,
        "time_limit": 35,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "poha_chivda",
        "ingredients": ["thin poha", "peanuts", "roasted chana dal", "green chilli"],
        "pantry": ["oil", "turmeric", "mustard seeds", "hing", "salt", "sugar"],
        "servings": 4.0,
        "time_limit": 20,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "pav_bhaji",
        "ingredients": ["potato", "cauliflower", "green peas", "capsicum", "onion", "tomato"],
        "pantry": ["butter", "pav bhaji masala", "ginger", "garlic", "salt"],
        "servings": 4.0,
        "time_limit": 45,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "sev_pav",
        "ingredients": ["pav", "sev", "onion", "garlic chutney", "coriander leaves"],
        "pantry": ["oil", "salt"],
        "servings": 2.0,
        "time_limit": 20,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "puri_bhaji",
        "ingredients": ["potato", "wheat flour", "onion", "green chilli"],
        "pantry": ["oil", "mustard seeds", "turmeric", "salt"],
        "servings": 3.0,
        "time_limit": 35,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "pithla_bhakri",
        "ingredients": ["besan", "jowar flour", "onion", "garlic", "green chilli"],
        "pantry": ["oil", "mustard seeds", "cumin seeds", "turmeric", "salt"],
        "servings": 2.0,
        "time_limit": 30,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "bharli_vangi",
        "ingredients": ["brinjal", "peanuts", "dry coconut", "onion", "ginger", "garlic"],
        "pantry": ["oil", "godha masala", "turmeric", "red chilli powder", "salt"],
        "servings": 4.0,
        "time_limit": 40,
        "allergens": ["dairy"]
    },
    {
        "dish_id": "kolhapuri_misal",
        "ingredients": ["sprouted matki", "onion", "tomato", "garlic", "ginger", "dry coconut"],
        "pantry": ["oil", "mustard seeds", "kashmiri chilli powder", "salt"],
        "servings": 3.0,
        "time_limit": 40,
        "allergens": ["dairy"]
    }
]


def classify_failure(failure_str: str) -> str:
    s = failure_str.lower()
    if "step count" in s:
        return "step_count"
    elif "words" in s:
        return "word_count"
    elif "unallowed ingredient" in s:
        return "unallowed_ingredient"
    elif "allergen violation" in s:
        return "allergen"
    elif "spice blend" in s:
        return "spice_blend"
    return "other"


def run_benchmark():
    print("=" * 80)
    print("STARTING LAYER 3 (v2) RECIPE WRITER EVALUATION (20 CASES x 2 CONFIGS)")
    print("=" * 80)

    scorer = DishScorer()
    dish_map = {}
    for d in scorer.dishes:
        dish_map[d["id"]] = d

    # Verify and match all 20 cases
    valid_cases = []
    for case in EVAL_CASES:
        did = case["dish_id"]
        if did in dish_map:
            valid_cases.append((case, dish_map[did]))
        else:
            found = False
            for k, d in dish_map.items():
                if did in k or k in did:
                    valid_cases.append((case, d))
                    found = True
                    break
            if not found:
                print(f"[Warning] Dish {did} not matched in catalog!")

    print(f"Matched {len(valid_cases)} / {len(EVAL_CASES)} evaluation cases successfully.")

    writer = RecipeWriter()
    catalog_vocab = load_catalog_ingredient_vocabulary()

    # Pre-warm model for config (b)
    print("\nWarming base model for LLM generation...")
    writer.load_model(mode="base")
    _ = writer.generate_raw_steps("Dish: Warmup\nAllowed Ingredients:\n- water: 1 cup\nInstructions:\n1. Heat water in a pot for five minutes until hot.\n2. Turn off the flame safely.", max_new_tokens=30)
    print("Base model warm and ready.")

    configs = [
        {"id": "stored_steps_control", "name": "Stored steps (control)", "is_llm": False},
        {"id": "llm_steps", "name": "LLM steps (base Qwen2.5-1.5B)", "is_llm": True}
    ]

    all_results = {}

    for cfg in configs:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        is_llm = cfg["is_llm"]

        print(f"\n---> Evaluating Config: [{cfg_name}]")
        case_records = []
        latencies = []
        failures_by_type = {
            "step_count": 0,
            "word_count": 0,
            "unallowed_ingredient": 0,
            "allergen": 0,
            "spice_blend": 0,
            "other": 0
        }
        allergen_violations_count = 0
        invented_ingredients_count = 0
        passed_cases = 0

        for idx, (case, dish) in enumerate(valid_cases, 1):
            dish_name = dish.get("name", case["dish_id"])
            t0 = time.time()

            # Build allowed ingredients
            stored = writer.get_stored_recipe(
                dish,
                servings=case["servings"],
                user_ingredients=case["ingredients"],
                pantry=case["pantry"]
            )
            allowed_ingredients = set(writer.staples)
            allowed_blends = set()
            for u in case["ingredients"]:
                allowed_ingredients.add(canonical_name(u))
            for p in case["pantry"]:
                allowed_ingredients.add(canonical_name(p))
            for ing in stored["ingredients"]:
                allowed_ingredients.add(canonical_name(ing["item"]))
                norm_i = normalize_item_name(ing["item"])
                for b in SPICE_BLENDS:
                    if b in norm_i:
                        allowed_blends.add(b)

            if not is_llm:
                # Config (a): Stored steps control
                steps = stored["steps"]
                t_elapsed = time.time() - t0
                passed, failures = validate_steps(
                    steps=steps,
                    allowed_ingredients=allowed_ingredients,
                    catalog_vocab=catalog_vocab,
                    excluded_allergens=case["allergens"],
                    allowed_catalog_blends=allowed_blends
                )
            else:
                # Config (b): LLM steps
                try:
                    res = writer.generate(
                        dish=dish,
                        user_ingredients=case["ingredients"],
                        pantry=case["pantry"],
                        servings=case["servings"],
                        time_limit=case["time_limit"],
                        allergens=case["allergens"],
                        mode_override="base"
                    )
                    t_elapsed = time.time() - t0
                    passed = (res["source"] == "llm" and res["validation"]["passed"])
                    failures = res["validation"]["failures"]
                    steps = res["recipe"]["steps"]
                except Exception as e:
                    t_elapsed = time.time() - t0
                    passed = False
                    failures = [f"Exception: {str(e)}"]
                    steps = []

            latencies.append(t_elapsed)

            if passed:
                passed_cases += 1
            else:
                for f in failures:
                    ftype = classify_failure(f)
                    failures_by_type[ftype] = failures_by_type.get(ftype, 0) + 1
                    if ftype == "allergen":
                        allergen_violations_count += 1
                    elif ftype == "unallowed_ingredient":
                        invented_ingredients_count += 1

            status_str = "PASS" if passed else f"FAIL ({len(failures)} failures)"
            print(f"  [{idx:02d}/20] {dish_name[:24]:<24} -> {status_str:<18} ({t_elapsed:.2f}s)")

            case_records.append({
                "case_num": idx,
                "dish_id": case["dish_id"],
                "dish_name": dish_name,
                "latency_sec": round(t_elapsed, 2),
                "validation_passed": passed,
                "failures_count": len(failures),
                "failures": failures,
                "step_count": len(steps)
            })

        val_pass_pct = round((passed_cases / len(valid_cases)) * 100.0, 1)
        avg_sec = round(sum(latencies) / len(latencies), 2)
        max_sec = round(max(latencies), 2)

        config_summary = {
            "config_id": cfg_id,
            "config_name": cfg_name,
            "cases_run": len(valid_cases),
            "validation_pass_pct": val_pass_pct,
            "failures_by_type": failures_by_type,
            "allergen_violations": allergen_violations_count,
            "invented_ingredient_steps": invented_ingredients_count,
            "avg_seconds": avg_sec,
            "max_seconds": max_sec,
            "reviewer_rating": "",
            "cases": case_records
        }
        all_results[cfg_id] = config_summary

        print(f"\nSummary for [{cfg_name}]:")
        print(f"  Validation Pass %:        {val_pass_pct}% ({passed_cases}/{len(valid_cases)})")
        print(f"  Failures by Type:         {failures_by_type}")
        print(f"  Allergen Violations:      {allergen_violations_count}")
        print(f"  Invented Ingredient Steps:{invented_ingredients_count}")
        print(f"  Avg Latency:              {avg_sec}s (Max: {max_sec}s)")

    # Save to docs/writer_eval_v2.json
    out_path = os.path.join(PROJECT_ROOT, "docs", "writer_eval_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved full results to {out_path}")

    # Check thresholds for Config (b)
    llm_res = all_results["llm_steps"]
    b_val = llm_res["validation_pass_pct"]
    b_allergens = llm_res["allergen_violations"]
    b_invented = llm_res["invented_ingredient_steps"]
    b_avg_sec = llm_res["avg_seconds"]

    print("\n" + "=" * 80)
    print("THRESHOLD EVALUATION FOR CONFIG (b) [LLM steps]:")
    print(f"- Validation >= 80%:       {b_val}% -> {'PASS' if b_val >= 80.0 else 'FAIL'}")
    print(f"- Allergen Violations == 0: {b_allergens} -> {'PASS' if b_allergens == 0 else 'FAIL'}")
    print(f"- Invented Ingredients == 0: {b_invented} -> {'PASS' if b_invented == 0 else 'FAIL'}")
    print(f"- Avg Latency <= 10.0s:    {b_avg_sec}s -> {'PASS' if b_avg_sec <= 10.0 else 'FAIL'}")
    print("=" * 80)

    passed_all = (b_val >= 80.0 and b_allergens == 0 and b_invented == 0 and b_avg_sec <= 10.0)
    if passed_all:
        print("\nALL THRESHOLDS PASSED: setting llm_mode = 'base' in config.json")
        cfg_file = os.path.join(PROJECT_ROOT, "config.json")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        cdata["llm_mode"] = "base"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cdata, f, indent=2)
    else:
        print("\nLLM V2 FAILED THRESHOLDS: keeping llm_mode = 'off' in config.json")
        cfg_file = os.path.join(PROJECT_ROOT, "config.json")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        cdata["llm_mode"] = "off"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cdata, f, indent=2)


if __name__ == "__main__":
    run_benchmark()

