"""
src/validators.py — Recipe validation rules.
NO FastAPI imports allowed in this module.

Validators:
1. JSON valid: Valid JSON with required schema (dish_name, local_name, ingredients, steps, time_minutes, missing_ingredients).
2. Dish matches decision: Generated dish matches the requested catalog dish.
3. Only allowed ingredients: No invented or unauthorized ingredients.
4. Scaled quantities match: Quantities match expected scaled quantities within 10% tolerance.
5. No allergens: Zero occurrences of excluded allergens in ingredients, steps, or missing items.
6. 3-10 steps: Recipe steps count must be between 3 and 10 inclusive.
7. Time within 50% of limit: Cooking time must be within 50% of requested time limit.
8. Spice blends are not free staples: Commercial/complex spice blends cannot be hallucinated without being in the allowed ingredients.
"""

import json
import re
from typing import Dict, Any, List, Set, Tuple, Optional, Union

# Common spice blends that are NOT basic free staples
SPICE_BLENDS = {
    "garam masala", "pav bhaji masala", "godha masala", "goda masala",
    "sambhar powder", "sambar powder", "biryani masala", "chaat masala",
    "kitchen king", "chana masala powder", "rasam powder", "curry powder",
    "tandoori masala", "panch phoron", "kashmiri masala"
}

# Rich allergen synonyms and derivative ingredients
ALLERGEN_TERMS = {
    "dairy": [
        "dairy", "milk", "butter", "cheese", "ghee", "paneer", "curd", "yogurt",
        "yoghurt", "cream", "malai", "whey", "makkhan", "khoya", "mawa"
    ],
    "gluten": [
        "gluten", "wheat", "flour", "maida", "atta", "suji", "sooji", "semolina",
        "rava", "bread", "roti", "chapati", "naan", "puri", "bhatura"
    ],
    "peanuts": [
        "peanut", "peanuts", "groundnut", "groundnuts", "shengdana", "singdana", "moongphali"
    ],
    "peanut": [
        "peanut", "peanuts", "groundnut", "groundnuts", "shengdana", "singdana", "moongphali"
    ],
    "nuts": [
        "nuts", "nut", "cashew", "kaju", "almond", "badam", "walnut", "akhrot",
        "pistachio", "pista"
    ],
    "tree nuts": [
        "nuts", "nut", "cashew", "kaju", "almond", "badam", "walnut", "akhrot",
        "pistachio", "pista"
    ],
    "soy": [
        "soy", "soya", "tofu", "edamame", "soy sauce", "soya chunks"
    ],
    "egg": [
        "egg", "eggs", "anda", "egg yolk", "egg white"
    ],
    "eggs": [
        "egg", "eggs", "anda", "egg yolk", "egg white"
    ],
    "seafood": [
        "fish", "prawn", "prawns", "shrimp", "shrimps", "surmai", "pomfret",
        "crab", "lobster", "bombil", "rawas", "mackerel", "bangda", "seafood"
    ],
    "fish": [
        "fish", "surmai", "pomfret", "bombil", "rawas", "mackerel", "bangda", "catfish"
    ],
    "mustard": [
        "mustard", "mustard seeds", "rai", "sarson", "mustard oil"
    ],
    "sesame": [
        "sesame", "sesame seeds", "til"
    ]
}


def parse_quantity_number(qty_str: str) -> Optional[float]:
    """
    Extract numeric float from quantity string.
    Supports mixed fractions ('1 1/2'), simple fractions ('1/2'), decimals ('1.5'), ints ('2').
    Returns None for qualitative descriptions ('to taste', 'as needed', 'pinch').
    """
    if not qty_str:
        return None
    s = str(qty_str).lower().strip()
    if any(q in s for q in ["to taste", "as needed", "pinch", "a pinch"]):
        return None

    # Mixed fraction: e.g. "1 1/2"
    m_mixed = re.match(r"^(\d+)\s+(\d+)/(\d+)", s)
    if m_mixed:
        whole, num, den = m_mixed.groups()
        if float(den) != 0:
            return float(whole) + float(num) / float(den)

    # Simple fraction: e.g. "1/2" or "3/4"
    m_frac = re.match(r"^(\d+)/(\d+)", s)
    if m_frac:
        num, den = m_frac.groups()
        if float(den) != 0:
            return float(num) / float(den)

    # Decimal or integer: e.g. "1.5" or "2"
    m_num = re.match(r"^(\d+(?:\.\d+)?)", s)
    if m_num:
        return float(m_num.group(1))

    return None


def normalize_item_name(name: str) -> str:
    """Normalize ingredient name for matching."""
    s = str(name).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def validate_json_schema(recipe_obj: Union[str, Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Validator 1: Validate JSON structure and required keys."""
    if isinstance(recipe_obj, str):
        raw = recipe_obj.strip()
        # Handle code fences
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            parsed = json.loads(raw)
        except Exception as e:
            return False, None, f"JSON parse error: {str(e)}"
    elif isinstance(recipe_obj, dict):
        parsed = recipe_obj
    else:
        return False, None, "Expected JSON string or dict"

    if not isinstance(parsed, dict):
        return False, None, "Recipe root must be a JSON object"

    required_keys = ["dish_name", "local_name", "ingredients", "steps", "time_minutes", "missing_ingredients"]
    for k in required_keys:
        if k not in parsed:
            return False, None, f"Missing required JSON key: '{k}'"

    if not isinstance(parsed["dish_name"], str) or not parsed["dish_name"].strip():
        return False, None, "'dish_name' must be a non-empty string"
    if not isinstance(parsed["local_name"], str):
        return False, None, "'local_name' must be a string"
    if not isinstance(parsed["ingredients"], list):
        return False, None, "'ingredients' must be a list"
    for idx, ing in enumerate(parsed["ingredients"]):
        if not isinstance(ing, dict) or "item" not in ing or "quantity" not in ing:
            return False, None, f"Ingredient at index {idx} must be an object with 'item' and 'quantity'"
    if not isinstance(parsed["steps"], list) or len(parsed["steps"]) == 0:
        return False, None, "'steps' must be a non-empty list of instructions"
    if not isinstance(parsed["time_minutes"], (int, float)) or parsed["time_minutes"] <= 0:
        return False, None, "'time_minutes' must be a positive number"
    if not isinstance(parsed["missing_ingredients"], list):
        return False, None, "'missing_ingredients' must be a list"

    return True, parsed, None


def validate_dish_match(recipe: Dict[str, Any], expected_dish: str) -> Tuple[bool, Optional[str]]:
    """Validator 2: Verify generated dish matches the decision target dish."""
    gen_dish = normalize_item_name(recipe.get("dish_name", ""))
    exp_dish = normalize_item_name(expected_dish)

    if not gen_dish or not exp_dish:
        return False, f"Dish mismatch: expected '{expected_dish}', got '{recipe.get('dish_name')}'"

    # Exact or sub-token match
    if gen_dish == exp_dish or gen_dish in exp_dish or exp_dish in gen_dish:
        return True, None

    # Check words overlap
    gen_words = set(gen_dish.split())
    exp_words = set(exp_dish.split())
    if gen_words and exp_words and len(gen_words.intersection(exp_words)) >= min(len(gen_words), len(exp_words)):
        return True, None

    return False, f"Dish mismatch: expected '{expected_dish}', got '{recipe.get('dish_name')}'"


def validate_allowed_ingredients(
    recipe: Dict[str, Any],
    allowed_ingredients: Set[str]
) -> Tuple[bool, List[str]]:
    """
    Validator 3: Verify only allowed ingredients are present. No invented ingredients.
    allowed_ingredients must contain normalized ingredient names.
    """
    failures = []
    normalized_allowed = {normalize_item_name(x) for x in allowed_ingredients}

    for ing in recipe.get("ingredients", []):
        raw_item = ing.get("item", "")
        norm_item = normalize_item_name(raw_item)
        if not norm_item:
            continue

        # Check direct or token match
        matched = False
        if norm_item in normalized_allowed:
            matched = True
        else:
            for allowed in normalized_allowed:
                if norm_item in allowed or allowed in norm_item:
                    matched = True
                    break
                # Check stems
                if norm_item.endswith("s") and norm_item[:-1] in allowed:
                    matched = True
                    break
                if allowed.endswith("s") and allowed[:-1] in norm_item:
                    matched = True
                    break

        if not matched:
            failures.append(f"Invented ingredient not allowed: '{raw_item}'")

    return (len(failures) == 0), failures


def validate_scaled_quantities(
    recipe: Dict[str, Any],
    expected_quantities: Dict[str, str],
    tolerance: float = 0.10
) -> Tuple[bool, List[str]]:
    """
    Validator 4: Verify scaled quantities match expected quantities within tolerance (default 10%).
    expected_quantities is a mapping from normalized ingredient name to expected quantity string.
    """
    failures = []
    norm_expected = {}
    for k, v in expected_quantities.items():
        norm_expected[normalize_item_name(k)] = (v, parse_quantity_number(v))

    for ing in recipe.get("ingredients", []):
        item = normalize_item_name(ing.get("item", ""))
        got_qty_str = str(ing.get("quantity", ""))
        got_num = parse_quantity_number(got_qty_str)

        # Find best matching expected ingredient
        exp_match = None
        if item in norm_expected:
            exp_match = norm_expected[item]
        else:
            for k_exp, val in norm_expected.items():
                if item in k_exp or k_exp in item:
                    exp_match = val
                    break

        if exp_match is None:
            continue

        exp_qty_str, exp_num = exp_match
        if exp_num is not None:
            if got_num is None:
                failures.append(f"Quantity mismatch for '{ing.get('item')}': expected ~{exp_qty_str}, got non-numeric '{got_qty_str}'")
            else:
                diff = abs(got_num - exp_num)
                relative_error = diff / max(exp_num, 1e-6)
                if relative_error > tolerance:
                    failures.append(f"Quantity mismatch for '{ing.get('item')}': expected ~{exp_qty_str} ({exp_num}), got '{got_qty_str}' ({got_num}, diff={relative_error*100:.1f}%)")

    return (len(failures) == 0), failures


def validate_no_allergens(
    recipe: Dict[str, Any],
    excluded_allergens: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validator 5: Zero occurrences of excluded allergens in ingredients, steps, or missing items.
    """
    failures = []
    if not excluded_allergens:
        return True, []

    # Compile search texts
    ing_texts = [f"{ing.get('item', '')} {ing.get('quantity', '')}" for ing in recipe.get("ingredients", [])]
    steps_texts = [str(s) for s in recipe.get("steps", [])]
    missing_texts = [str(m) for m in recipe.get("missing_ingredients", [])]
    combined_text = " ".join(ing_texts + steps_texts + missing_texts).lower()

    for allergen in excluded_allergens:
        a_clean = allergen.lower().strip()
        if not a_clean:
            continue

        # Get synonym terms
        terms = ALLERGEN_TERMS.get(a_clean, [a_clean])
        if a_clean not in terms:
            terms.append(a_clean)

        for term in terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, combined_text):
                failures.append(f"Allergen violation: contains '{term}' matching excluded allergen '{allergen}'")
                break

    return (len(failures) == 0), failures


def validate_steps_count(recipe: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validator 6: Steps count must be between 3 and 10 inclusive."""
    steps = recipe.get("steps", [])
    if not isinstance(steps, list):
        return False, "Steps must be a list"
    count = len(steps)
    if count < 3 or count > 10:
        return False, f"Step count must be 3-10, got {count}"
    for idx, s in enumerate(steps):
        if not isinstance(s, str) or len(s.strip()) < 5:
            return False, f"Step {idx+1} is empty or too short"
    return True, None


def validate_time_limit(recipe: Dict[str, Any], time_limit: Optional[int]) -> Tuple[bool, Optional[str]]:
    """Validator 7: Cooking time within 50% of the limit."""
    if not time_limit or time_limit <= 0:
        return True, None

    time_val = recipe.get("time_minutes")
    if not isinstance(time_val, (int, float)):
        return False, f"Invalid 'time_minutes': {time_val}"

    min_time = 0.5 * time_limit
    max_time = 1.5 * time_limit

    if time_val < min_time or time_val > max_time:
        return False, f"Cooking time {time_val}m not within 50% of limit {time_limit}m (allowed: {min_time:.1f}m - {max_time:.1f}m)"

    return True, None


def validate_spice_blends(
    recipe: Dict[str, Any],
    allowed_catalog_blends: Set[str]
) -> Tuple[bool, List[str]]:
    """
    Validator 8: Spice blends are not free staples.
    Complex spice blends cannot be included unless they were in the allowed catalog ingredients.
    """
    failures = []
    norm_allowed_blends = {normalize_item_name(x) for x in allowed_catalog_blends}

    for ing in recipe.get("ingredients", []):
        item_norm = normalize_item_name(ing.get("item", ""))
        for blend in SPICE_BLENDS:
            if blend in item_norm:
                if not any(blend in allowed for allowed in norm_allowed_blends):
                    failures.append(f"Spice blend '{blend}' is not a free staple and was not in allowed catalog ingredients")

    return (len(failures) == 0), failures


def validate_recipe(
    recipe_obj: Union[str, Dict[str, Any]],
    expected_dish: str,
    allowed_ingredients: Set[str],
    expected_quantities: Dict[str, str],
    excluded_allergens: Optional[List[str]] = None,
    time_limit: Optional[int] = None,
    allowed_catalog_blends: Optional[Set[str]] = None,
    tolerance: float = 0.10
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Master validator combining all 8 checks.
    Returns (is_valid, parsed_recipe_dict, list_of_failure_reasons).
    """
    failures: List[str] = []

    # 1. JSON Valid
    valid_json, recipe, json_err = validate_json_schema(recipe_obj)
    if not valid_json:
        return False, {}, [json_err]

    # 2. Dish matches decision
    ok_dish, dish_err = validate_dish_match(recipe, expected_dish)
    if not ok_dish and dish_err:
        failures.append(dish_err)

    # 3. Only allowed ingredients
    ok_allowed, invent_errs = validate_allowed_ingredients(recipe, allowed_ingredients)
    failures.extend(invent_errs)

    # 4. Scaled quantities match
    ok_qty, qty_errs = validate_scaled_quantities(recipe, expected_quantities, tolerance=tolerance)
    failures.extend(qty_errs)

    # 5. No allergens
    ok_allergens, allergen_errs = validate_no_allergens(recipe, excluded_allergens or [])
    failures.extend(allergen_errs)

    # 6. 3-10 steps
    ok_steps, step_err = validate_steps_count(recipe)
    if not ok_steps and step_err:
        failures.append(step_err)

    # 7. Time within 50% of limit
    ok_time, time_err = validate_time_limit(recipe, time_limit)
    if not ok_time and time_err:
        failures.append(time_err)

    # 8. Spice blends not free staples
    ok_spices, spice_errs = validate_spice_blends(recipe, allowed_catalog_blends or set())
    failures.extend(spice_errs)

    return (len(failures) == 0), recipe, failures

