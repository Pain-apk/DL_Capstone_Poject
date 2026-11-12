"""
tests/eval_writer.py — Evaluation benchmark for Layer 3 Recipe Writer.
Evaluates 20 catalog cases across 4 configurations:
1. Base without method outline
2. Base with outline (DECISION header)
3. Adapter in trained format
4. Adapter with DECISION header

Metrics per config:
- JSON valid %
- Validation pass %
- Allergen violations
- Invented ingredients
- Quantity mismatch %
- Average seconds per recipe
- Reviewer rating (left empty for human review)

Saves results to docs/writer_eval.json.
Checks pass thresholds:
  JSON valid >= 90%, validation >= 80%, 0 allergen violations, 0 invented ingredients, quantity mismatch <= 10%.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scorer import DishScorer, canonical_name, parse_ingredient_line_fixed
from src.writer import RecipeWriter, build_decision_header, format_scaled_quantity
from src.validators import (
    validate_recipe,
    validate_json_schema,
    validate_dish_match,
    validate_allowed_ingredients,
    validate_scaled_quantities,
    validate_no_allergens,
    validate_steps_count,
    validate_time_limit,
    validate_spice_blends,
    normalize_item_name
)

# 20 Diverse catalog cases (vegetarian first, realistic ingredients)
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


def run_benchmark():
    print("=" * 80)
    print("STARTING LAYER 3 RECIPE WRITER EVALUATION (20 CASES x 4 CONFIGS)")
    print("=" * 80)

    scorer = DishScorer()
    dish_map = {}
    for d in scorer.dishes:
        dish_map[d["id"]] = d

    # Verify all 20 test cases match catalog
    valid_cases = []
    for case in EVAL_CASES:
        did = case["dish_id"]
        if did in dish_map:
            valid_cases.append((case, dish_map[did]))
        else:
            # Fallback substring
            found = False
            for k, d in dish_map.items():
                if did in k or k in did:
                    valid_cases.append((case, d))
                    found = True
                    break
            if not found:
                print(f"[Warning] Dish {did} not found in catalog!")

    print(f"Loaded {len(valid_cases)} / {len(EVAL_CASES)} evaluation cases successfully.")

    writer = RecipeWriter()

    configs = [
        {"id": "base_no_outline", "name": "Base without method outline", "model_mode": "base"},
        {"id": "base_with_outline", "name": "Base with outline", "model_mode": "base"},
        {"id": "adapter_trained_format", "name": "Adapter in trained format", "model_mode": "adapter"},
        {"id": "adapter_decision_header", "name": "Adapter with DECISION header", "model_mode": "adapter"}
    ]

    all_results = {}

    for cfg in configs:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        mode = cfg["model_mode"]

        print(f"\n---> Evaluating Config: [{cfg_name}] (model_mode={mode})")
        # Load model for this config
        t_load_start = time.time()
        ok_load = writer.load_model(mode=mode)
        if not ok_load:
            print(f"ERROR: Failed to load model for {cfg_id} in mode {mode}!")
            continue
        print(f"Model {mode} ready in {time.time() - t_load_start:.2f}s.")

        config_case_results = []
        json_valid_count = 0
        val_pass_count = 0
        total_allergen_violations = 0
        total_invented_ingredients = 0
        qty_mismatch_count = 0
        latencies = []

        for idx, (case, dish) in enumerate(valid_cases):
            raw_row = dish.get("raw_row", dish)
            dish_name = dish.get("name") or raw_row.get("dish_name")
            user_ings = case["ingredients"]
            pantry = case.get("pantry", [])
            servings = case.get("servings", 2.0)
            time_limit = case.get("time_limit", 30)
            allergens = case.get("allergens", [])

            # Prepare allowed sets & expected quantities
            base_servings = float(raw_row.get("servings", 2.0)) if str(raw_row.get("servings", "")).replace(".", "").isdigit() else 2.0
            scale = float(servings) / base_servings if base_servings > 0 else 1.0

            allowed_ingredients = set(writer.staples)
            for u in user_ings:
                allowed_ingredients.add(canonical_name(u))
            for p in pantry:
                allowed_ingredients.add(canonical_name(p))

            expected_quantities = {}
            allowed_blends = set()
            iq_lines = [x.strip() for x in str(raw_row.get("ingredients_with_quantity", "")).split("|") if x.strip()]
            for line in iq_lines:
                qty, item = parse_ingredient_line_fixed(line)
                sq = format_scaled_quantity(qty, scale)
                c_item = canonical_name(item)
                allowed_ingredients.add(c_item)
                expected_quantities[c_item] = sq
                norm_item = normalize_item_name(item)
                for blend in ["garam masala", "pav bhaji masala", "godha masala", "sambhar powder", "biryani masala", "chaat masala"]:
                    if blend in norm_item:
                        allowed_blends.add(blend)

            # Construct Prompt according to config
            if cfg_id == "base_no_outline":
                sys_prompt = (
                    "You are an expert Indian culinary assistant. Given detected ingredients, regional context, "
                    "dietary requirements, and time constraints, output the authentic recipe in strict JSON format. "
                    "Output valid JSON only with keys: dish_name, local_name, ingredients (list of objects with 'item' and 'quantity'), "
                    "steps (list of strings), time_minutes (integer), missing_ingredients (list of strings)."
                )
                user_prompt = (
                    f"Dish: {dish_name}\n"
                    f"User Ingredients: {', '.join(user_ings)}\n"
                    f"Servings: {servings}\n"
                    f"Time Limit: {time_limit} minutes\n"
                    f"Excluded Allergens: {', '.join(allergens) if allergens else 'none'}\n"
                    f"Generate the recipe strictly in JSON format."
                )
            elif cfg_id == "base_with_outline":
                header = build_decision_header(
                    dish=dish,
                    user_ingredients=user_ings,
                    pantry=pantry,
                    servings=servings,
                    time_limit=time_limit,
                    allergens=allergens
                )
                sys_prompt = (
                    "You are an expert Indian culinary assistant. Given the following decision constraints, "
                    "output the authentic recipe in strict JSON format. Output valid JSON only with keys: "
                    "dish_name, local_name, ingredients, steps, time_minutes, missing_ingredients."
                )
                user_prompt = f"Decision Header:\n{header}\n\nPlease output the exact recipe in valid JSON format only."
            elif cfg_id == "adapter_trained_format":
                # Exact format adapter was trained on in train.jsonl
                sys_prompt = (
                    "You are an expert Indian culinary assistant. Given detected ingredients, regional context, "
                    "dietary requirements, time constraints, and candidate dishes, select the best matching dish "
                    "and output the authentic recipe in strict JSON format. Output valid JSON only."
                )
                user_prompt = (
                    f"Detected Ingredients: {', '.join(user_ings)}\n"
                    f"Region: {dish.get('state', 'Maharashtra')}\n"
                    f"Diet: {dish.get('diet', 'vegetarian')}\n"
                    f"Time Limit: {time_limit} minutes\n"
                    f"Excluded Allergens: {', '.join(allergens) if allergens else 'none'}\n"
                    f"Candidate Dishes: {dish_name}"
                )
            elif cfg_id == "adapter_decision_header":
                header = build_decision_header(
                    dish=dish,
                    user_ingredients=user_ings,
                    pantry=pantry,
                    servings=servings,
                    time_limit=time_limit,
                    allergens=allergens
                )
                sys_prompt = (
                    "You are an expert Indian culinary assistant. Given the following decision constraints, "
                    "output the authentic recipe in strict JSON format matching all constraints. Output valid JSON only."
                )
                user_prompt = f"Decision Header:\n{header}\n\nPlease generate the exact recipe strictly following the DECISION constraints in valid JSON only."

            # Measure generation latency
            t0 = time.time()
            try:
                raw_output = writer.generate_raw_llm(
                    prompt=user_prompt,
                    system_prompt=sys_prompt,
                    max_new_tokens=350,
                    temperature=0.1
                )
            except Exception as e:
                raw_output = f"ERROR: {str(e)}"
            dt = time.time() - t0
            latencies.append(dt)

            # 1. Check JSON validity
            is_json_valid, parsed_recipe, json_err = validate_json_schema(raw_output)
            case_allergen_viols = 0
            case_invented_count = 0
            case_qty_mismatch = False
            case_val_passed = False
            failures = []

            if is_json_valid:
                json_valid_count += 1
                # Run full validation
                val_ok, rec, val_failures = validate_recipe(
                    recipe_obj=parsed_recipe,
                    expected_dish=dish_name,
                    allowed_ingredients=allowed_ingredients,
                    expected_quantities=expected_quantities,
                    excluded_allergens=allergens,
                    time_limit=time_limit,
                    allowed_catalog_blends=allowed_blends,
                    tolerance=0.10
                )
                case_val_passed = val_ok
                if val_ok:
                    val_pass_count += 1
                failures = val_failures

                # Count specific errors
                for f in val_failures:
                    if "Allergen violation" in f:
                        case_allergen_viols += 1
                    if "Invented ingredient" in f:
                        case_invented_count += 1
                    if "Quantity mismatch" in f:
                        case_qty_mismatch = True
            else:
                failures = [json_err]
                case_qty_mismatch = True

            total_allergen_violations += case_allergen_viols
            total_invented_ingredients += case_invented_count
            if case_qty_mismatch:
                qty_mismatch_count += 1

            config_case_results.append({
                "dish_id": case["dish_id"],
                "dish_name": dish_name,
                "latency_sec": round(dt, 2),
                "json_valid": is_json_valid,
                "validation_passed": case_val_passed,
                "allergen_violations": case_allergen_viols,
                "invented_ingredients": case_invented_count,
                "quantity_mismatch": case_qty_mismatch,
                "failures": failures,
                "raw_output": raw_output[:300] + "..." if len(raw_output) > 300 else raw_output
            })

            print(f"  [{idx+1:02d}/20] {dish_name:<20} | {dt:4.1f}s | JSON: {str(is_json_valid):<5} | Val: {str(case_val_passed):<5} | Failures: {len(failures)}", flush=True)

        total_cases = len(valid_cases)
        json_valid_pct = round((json_valid_count / total_cases) * 100.0, 1)
        val_pass_pct = round((val_pass_count / total_cases) * 100.0, 1)
        qty_mismatch_pct = round((qty_mismatch_count / total_cases) * 100.0, 1)
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        all_results[cfg_id] = {
            "config_id": cfg_id,
            "config_name": cfg_name,
            "model_mode": mode,
            "json_valid_pct": json_valid_pct,
            "validation_pass_pct": val_pass_pct,
            "allergen_violations": total_allergen_violations,
            "invented_ingredients": total_invented_ingredients,
            "quantity_mismatch_pct": qty_mismatch_pct,
            "avg_seconds": avg_latency,
            "reviewer_rating": "",
            "cases": config_case_results
        }

    # Save to docs/writer_eval.json
    docs_dir = os.path.join(PROJECT_ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_file = os.path.join(docs_dir, "writer_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Artifact Saved] Evaluation metrics saved to {out_file}")

    # Print Summary Table
    print("\n" + "=" * 100)
    print("LAYER 3 RECIPE WRITER EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Config Name':<32} | {'JSON %':<7} | {'Val %':<7} | {'Allergens':<9} | {'Invented':<8} | {'Qty Mismatch %':<14} | {'Avg Sec':<7} | {'Reviewer Rating'}")
    print("-" * 100)

    passing_config = None
    for cfg in configs:
        res = all_results.get(cfg["id"])
        if not res:
            continue
        print(f"{res['config_name']:<32} | {res['json_valid_pct']:>6.1f}% | {res['validation_pass_pct']:>6.1f}% | {res['allergen_violations']:>9d} | {res['invented_ingredients']:>8d} | {res['quantity_mismatch_pct']:>13.1f}% | {res['avg_seconds']:>6.2f}s | {res['reviewer_rating']}")

        # Thresholds: JSON valid >= 90%, validation >= 80%, 0 allergen violations, 0 invented ingredients, quantity mismatch <= 10%
        if (
            res["json_valid_pct"] >= 90.0
            and res["validation_pass_pct"] >= 80.0
            and res["allergen_violations"] == 0
            and res["invented_ingredients"] == 0
            and res["quantity_mismatch_pct"] <= 10.0
        ):
            if passing_config is None:
                passing_config = res

    print("-" * 100)
    print("THRESHOLDS: JSON valid >= 90%, validation >= 80%, 0 allergen violations, 0 invented ingredients, quantity mismatch <= 10%")

    config_json_path = os.path.join(PROJECT_ROOT, "config.json")
    current_cfg = {}
    if os.path.exists(config_json_path):
        with open(config_json_path, "r", encoding="utf-8") as f:
            current_cfg = json.load(f)

    if passing_config:
        print(f"\nSUCCESS: Configuration '{passing_config['config_name']}' passed all thresholds!")
        chosen_mode = passing_config["model_mode"]
        current_cfg["llm_mode"] = chosen_mode
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(current_cfg, f, indent=2)
        print(f"Set default LLM_MODE='{chosen_mode}' in config.json.")
    else:
        print("\nLLM FAILED THRESHOLDS: stored recipes are the main answer.")
        current_cfg["llm_mode"] = "off"
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(current_cfg, f, indent=2)
        print("Maintained LLM_MODE='off' in config.json.")

    print("=" * 100)


if __name__ == "__main__":
    run_benchmark()
