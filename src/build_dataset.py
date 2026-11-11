"""
build_dataset.py - Dataset builder and chat-formatting pipeline.

Execution Modes:
--mode pilot: Reads all drafted rows (including verified='no') -> data/processed/pilot_unverified/
--mode final: Reads strictly rows with verified='yes' -> data/processed/final/

STRICT SAFETY GUARDS:
1. Refuses to read tests/fixtures or any test directory.
2. Refuses to read eval/cases.json.
3. Splits 90/5/5 by DISH TITLE with fixed random seed (seed=42).
4. Fixed system prompt across all examples for prefix caching.
"""

import os
import sys
import re
import csv
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FORBIDDEN_PATHS = ["tests", "fixtures", "eval/cases.json", "eval\\cases.json"]

FIXED_SYSTEM_PROMPT = (
    "You are an expert Indian culinary assistant. Given detected ingredients, regional context, "
    "dietary requirements, time constraints, and candidate dishes, select the best matching dish "
    "and output the authentic recipe in strict JSON format. Output valid JSON only."
)

def assert_safe_path(filepath: str):
    norm = os.path.normcase(os.path.abspath(filepath))
    for fb in FORBIDDEN_PATHS:
        if fb in norm:
            raise PermissionError(
                f"[GUARD VIOLATION] build_dataset.py strictly refuses access to test/eval fixture '{filepath}'!"
            )

def parse_ingredients_json(ings_with_qty_str: str) -> List[Dict[str, str]]:
    """
    Parses '1.5 cup poha|1 medium onion|8 curry leaves|salt to taste' into
    [{"item": "poha", "quantity": "1.5 cup"}, {"item": "onion", "quantity": "1 medium"},
     {"item": "curry leaves", "quantity": "8"}, ...]

    Parsing rule:
    - A quantity ends at the last unit token (tsp, tbsp, cup, g, kg, ml, l,
      medium, large, small, pinch, bunch, clove, piece, pieces) or at the
      first number if no unit token follows.
    - Everything after the quantity section is the item name (may be multi-word).
    """
    # Tokens that act as units of measure (not part of the item name).
    UNIT_TOKENS = {
        "tsp", "tbsp", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
        "cup", "cups", "g", "kg", "ml", "l", "oz", "lb", "lbs",
        "medium", "large", "small", "pinch", "bunch", "clove", "cloves",
        "piece", "pieces", "slice", "slices",
    }

    def _split_qty_item(text: str) -> Tuple[str, str]:
        """
        Split a single ingredient string into (quantity, item).
        Strategy: scan tokens left-to-right; quantity = leading number +
        all immediately-following unit tokens; item = remainder.
        """
        tokens = text.split()
        if not tokens:
            return ("1 portion", text)

        # Must start with a numeric token (int, float, or fraction like 1/2)
        num_re = re.compile(r"^\d+([./]\d+)?$")
        if not num_re.match(tokens[0]):
            # No leading number → whole string is the item
            return ("to taste" if "to taste" in text.lower() else "1 portion", text)

        qty_tokens = [tokens[0]]
        rest = tokens[1:]

        # Greedily consume unit tokens
        for tok in rest:
            if tok.lower() in UNIT_TOKENS:
                qty_tokens.append(tok)
            else:
                break

        item_start = len(qty_tokens)
        item_tokens = tokens[item_start:]
        qty_str = " ".join(qty_tokens)
        item_str = " ".join(item_tokens).strip()

        if not item_str:
            # Edge case: just a number with a unit and nothing else
            if len(qty_tokens) > 1:
                # e.g. "2 clove" with no item → item = last unit token
                item_str = qty_tokens[-1]
                qty_str = " ".join(qty_tokens[:-1])
            else:
                item_str = "ingredient"

        return (qty_str, item_str)

    items = []
    for line in ings_with_qty_str.split("|"):
        line = line.strip()
        if not line:
            continue
        if "to taste" in line.lower():
            name = re.sub(r"\bto taste\b", "", line, flags=re.IGNORECASE).strip()
            items.append({"item": name if name else "salt", "quantity": "to taste"})
        else:
            qty, item = _split_qty_item(line)
            items.append({"item": item, "quantity": qty})
    return items

def generate_prompt_variants(dish_row: Dict[str, Any],
                             all_dishes_pool: List[Dict[str, Any]],
                             num_variants: int = 5,
                             seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates 4-6 prompt variations for a dish:
    - User message with detected ingredients, state, diet, time limit, allergens, and 3-5 candidates (including true dish).
    - Assistant message with exact target JSON schema.
    """
    dish_name = dish_row["dish_name"]
    local_name = dish_row.get("romanized_local_name", dish_name)
    state = dish_row.get("state", "Maharashtra")
    diet = dish_row.get("diet", "vegetarian")
    time_min = int(dish_row.get("time_minutes", 25))
    main_ings = [i.strip() for i in dish_row.get("main_ingredients", "").split("|") if i.strip()]
    raw_ings = dish_row.get("ingredients_with_quantity", "")
    parsed_ingredients = parse_ingredients_json(raw_ings)
    steps = [s.strip() for s in dish_row.get("steps", "").split("|") if s.strip()]

    # Other dish candidates from same or nearby states
    other_dishes = [d["dish_name"] for d in all_dishes_pool if d["dish_name"] != dish_name]

    variants = []
    rng = random.Random(seed + hash(dish_name) % 10000)

    allergen_pool = ["nuts", "dairy", "gluten", "mustard", "soy", "peanuts", "none"]

    for v_idx in range(num_variants):
        # 1. Candidate dishes: select 2-4 distractors + true dish
        num_distractors = rng.randint(2, 4)
        distractors = rng.sample(other_dishes, min(num_distractors, len(other_dishes)))
        candidate_list = distractors + [dish_name]
        rng.shuffle(candidate_list)

        # 2. Detected ingredients: main ingredients with occasional missing ingredient
        if len(main_ings) > 2 and v_idx % 2 == 1:
            detected = main_ings[:-1]
            missing = [main_ings[-1]]
        else:
            detected = list(main_ings)
            missing = []

        # Optional extra pantry item in detected
        if rng.random() > 0.5:
            detected.append("salt")

        # 3. Dietary & allergen variations
        chosen_allergen = rng.choice(allergen_pool)
        if chosen_allergen in dish_name.lower() or any(chosen_allergen in i.lower() for i in main_ings):
            chosen_allergen = "none"

        # 4. Time limit variations
        time_limit = time_min + rng.choice([0, 5, 10, 15])

        # Construct User Content
        user_content = (
            f"Detected Ingredients: {', '.join(detected)}\n"
            f"Region: {state}\n"
            f"Diet: {diet}\n"
            f"Time Limit: {time_limit} minutes\n"
            f"Excluded Allergens: {chosen_allergen}\n"
            f"Candidate Dishes: {', '.join(candidate_list)}"
        )

        # Construct Assistant JSON
        assistant_json = {
            "dish_name": dish_name,
            "local_name": local_name,
            "ingredients": parsed_ingredients,
            "steps": steps,
            "time_minutes": time_min,
            "missing_ingredients": missing
        }

        variants.append({
            "messages": [
                {"role": "system", "content": FIXED_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
            ],
            "dish_name": dish_name,
            "variant_id": v_idx
        })

    return variants

def build_dataset(mode: str = "pilot", seed: int = 42):
    random.seed(seed)

    if mode == "pilot":
        input_csv = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")
        output_dir = os.path.join(PROJECT_ROOT, "data", "processed", "pilot_unverified")
    elif mode == "final":
        input_csv = os.path.join(PROJECT_ROOT, "data", "processed", "reviewed_recipes.csv")
        output_dir = os.path.join(PROJECT_ROOT, "data", "processed", "final")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'pilot' or 'final'.")

    assert_safe_path(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Source file {input_csv} does not exist!")

    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Mode-based filtering
    eligible_dishes = []
    for r in reader:
        if r.get("source") == "test_fixture":
            continue
        if mode == "final" and r.get("verified", "").lower() != "yes":
            continue
        # Ensure recipe has valid ingredients and steps
        if r.get("ingredients_with_quantity") and r.get("steps") and not r.get("steps").startswith("BLOCKED"):
            eligible_dishes.append(r)

    print(f"Loaded {len(eligible_dishes)} eligible dishes for mode '{mode}'.")
    if not eligible_dishes:
        raise ValueError("No eligible dishes found to build dataset!")

    # Group uniquely by dish title to guarantee 0 data leakage
    unique_titles = sorted(list({d["dish_name"] for d in eligible_dishes}))
    rng = random.Random(seed)
    rng.shuffle(unique_titles)

    total_dishes = len(unique_titles)
    train_end = int(0.90 * total_dishes)
    val_end = int(0.95 * total_dishes)

    train_titles = set(unique_titles[:train_end])
    val_titles = set(unique_titles[train_end:val_end])
    test_titles = set(unique_titles[val_end:])

    print(f"Dish splits (90/5/5): Train={len(train_titles)}, Val={len(val_titles)}, Test={len(test_titles)}")

    train_data = []
    val_data = []
    test_data = []

    for d in eligible_dishes:
        title = d["dish_name"]
        variants = generate_prompt_variants(d, eligible_dishes, num_variants=5, seed=seed)
        if title in train_titles:
            train_data.extend(variants)
        elif title in val_titles:
            val_data.extend(variants)
        elif title in test_titles:
            test_data.extend(variants)

    # Save to JSONL
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    for path, data in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")

    print(f"\nGenerated dataset files in {output_dir}:")
    print(f"  Train: {len(train_data)} examples ({train_path})")
    print(f"  Val:   {len(val_data)} examples ({val_path})")
    print(f"  Test:  {len(test_data)} examples ({test_path})")

    # Print 5 random examples
    print("\n" + "=" * 80)
    print("5 RANDOMLY SELECTED TRAINING EXAMPLES:")
    print("=" * 80)
    sample_exs = random.sample(train_data, min(5, len(train_data)))
    for idx, ex in enumerate(sample_exs, 1):
        msgs = ex["messages"]
        print(f"\n[EXAMPLE {idx} - Dish: {ex['dish_name']} (Variant {ex['variant_id']})]")
        print(">>> USER:")
        print(msgs[1]["content"])
        print(">>> ASSISTANT:")
        print(msgs[2]["content"])
        print("-" * 60)

    return train_path, val_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot", help="Execution mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    build_dataset(mode=args.mode, seed=args.seed)
