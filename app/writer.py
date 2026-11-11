"""
app/writer.py — Decision header builder, LLM generation, and 9 strict code checks.
Python 3.8-compatible. Loads Qwen2.5-1.5B-Instruct + QLoRA adapter in 4-bit.
"""

import os
import sys
import json
import re
import time
import threading
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.recipes import get_recipe
from app.scorer import STAPLES, NON_VEG_INGREDIENTS

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    os.path.join(PROJECT_ROOT, "outputs", "pilot_unverified", "adapter")
)

FIXED_SYSTEM_PROMPT = (
    "You are an expert Indian culinary assistant. Given detected ingredients, regional context, "
    "dietary requirements, time constraints, and candidate dishes, select the best matching dish "
    "and output the authentic recipe in strict JSON format. Output valid JSON only."
)

# Concurrency lock: one generation request at a time
_model_lock = threading.Lock()

# Module-level singleton
_model = None
_tokenizer = None
_model_load_attempted = False
_vram_after_load_mb = 0.0


def is_llm_enabled() -> bool:
    val = os.environ.get("LLM_ENABLED", "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def load_writer_model() -> Tuple[Any, Any]:
    """Lazy-load the base model + LoRA adapter in 4-bit NF4 once."""
    global _model, _tokenizer, _model_load_attempted, _vram_after_load_mb

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if _model_load_attempted and _model is None:
        raise RuntimeError("Previous attempt to load writer model failed.")

    _model_load_attempted = True

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )

    _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    _model.eval()

    if torch.cuda.is_available():
        _vram_after_load_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    return _model, _tokenizer


def scale_quantity(qty_str: str, base_servings: float, target_servings: float) -> str:
    """Scale a quantity string linearly according to servings ratio."""
    if not qty_str or base_servings <= 0 or target_servings <= 0:
        return qty_str
    ratio = target_servings / base_servings
    if abs(ratio - 1.0) < 0.05:
        return qty_str

    # Match leading numbers or fractions like 1, 1.5, 1/2, 2/3
    def _repl(m):
        val_str = m.group(0)
        try:
            if "/" in val_str:
                num, denom = val_str.split("/")
                val = float(num) / float(denom)
            else:
                val = float(val_str)
            scaled = val * ratio
            # format cleanly
            if abs(scaled - round(scaled)) < 0.05:
                return str(int(round(scaled)))
            elif abs(scaled * 2 - round(scaled * 2)) < 0.05:
                return f"{int(round(scaled * 2)) / 2:g}"
            else:
                return f"{scaled:.2f}".rstrip("0").rstrip(".")
        except Exception:
            return val_str

    return re.sub(r"^\d+(?:/\d+)?(?:\.\d+)?", _repl, qty_str)


def build_decision_header(
    dish: Dict[str, Any],
    stored: Dict[str, Any],
    user_ingredients: List[str],
    allergens: List[str],
    servings: int,
    max_time: int
) -> str:
    dish_name = dish.get("dish_name", "")
    state = dish.get("state", "Maharashtra")
    rank = dish.get("rank", 1)
    match_score = dish.get("ingredient_match", 1.0)
    missing = dish.get("missing_mains", [])
    missing_str = ", ".join(missing) if missing else "none"
    avoid_str = ", ".join(allergens) if allergens else "none"

    base_servings = float(stored.get("servings", 2.0))
    allowed_list = []
    for item in stored.get("ingredients_with_quantity", []):
        raw_qty = item.get("qty", "")
        item_name = item.get("item", "")
        scaled_q = scale_quantity(raw_qty, base_servings, float(servings))
        allowed_list.append(f"{item_name} {scaled_q}".strip())

    allowed_str = "; ".join(allowed_list)

    header = (
        "[DECISION - made by code, do not change]\n"
        f"dish: {dish_name} (state: {state}, rank {rank} of 5, ingredient match {match_score:.2f})\n"
        f"user ingredients: {', '.join(user_ingredients)}\n"
        f"missing for this dish: {missing_str}\n"
        f"servings: {servings}   time limit: {max_time if max_time > 0 else 'none'}\n"
        f"avoid: {avoid_str}\n"
        f"allowed ingredients with quantities: {allowed_str}\n"
        "style: short, plain, modest. Use only the allowed ingredients. If something is missing, "
        "say it in missing_ingredients. Do not claim authenticity, health benefits or nutrition numbers.\n"
        "[END DECISION]\n"
    )
    return header


def build_trained_user_message(
    dish: Dict[str, Any],
    user_ingredients: List[str],
    allergens: List[str],
    max_time: int
) -> str:
    dish_name = dish.get("dish_name", "")
    state = dish.get("state", "Maharashtra")
    diet = dish.get("diet", "vegetarian")
    time_limit = f"{max_time if max_time > 0 else 30} minutes"
    allergen_str = ", ".join(allergens) if allergens else "none"

    body = (
        f"Detected Ingredients: {', '.join(user_ingredients)}\n"
        f"Region: {state}\n"
        f"Diet: {diet}\n"
        f"Time Limit: {time_limit}\n"
        f"Excluded Allergens: {allergen_str}\n"
        f"Candidate Dishes: {dish_name}"
    )
    return body


def compute_missing_ingredients(stored: Dict[str, Any], pantry: List[str]) -> List[str]:
    """Compute true missing ingredients deterministically from stored mains vs user pantry + staples."""
    effective = set(p.lower().strip() for p in pantry) | STAPLES
    missing = []
    for main in stored.get("main_ingredients", []):
        m_clean = main.lower().strip()
        if m_clean in STAPLES:
            continue
        if not any(m_clean in p or p in m_clean for p in effective):
            missing.append(main)
    return missing


def run_9_checks(
    raw_output: str,
    decided_dish: str,
    stored_recipe: Dict[str, Any],
    user_ingredients: List[str],
    pantry: List[str],
    allergens: List[str],
    servings: int
) -> Tuple[bool, Optional[Dict[str, Any]], List[str], float]:
    """
    Run 9 strict code checks on raw output.
    Returns: (is_passed, parsed_recipe_or_none, failed_checks_list, quantity_mismatch_rate)
    """
    failed = []
    qty_mismatch_rate = 0.0

    # 1. JSON parses and has all required keys; steps has 3-12 non-empty items
    required_keys = {"dish_name", "local_name", "ingredients", "steps", "time_minutes", "missing_ingredients"}
    try:
        recipe = json.loads(raw_output)
    except Exception as e:
        return False, None, [f"Check 1: JSON parse failure: {e}"], 0.0

    if not isinstance(recipe, dict) or not required_keys.issubset(recipe.keys()):
        missing_keys = list(required_keys - set(recipe.keys() if isinstance(recipe, dict) else []))
        return False, None, [f"Check 1: Missing keys: {missing_keys}"], 0.0

    steps = recipe.get("steps", [])
    if not isinstance(steps, list) or len(steps) < 3 or len(steps) > 12 or not all(isinstance(s, str) and s.strip() for s in steps):
        failed.append(f"Check 1: Steps count ({len(steps)}) outside [3, 12] or has empty step")

    # 2. dish_name equals the decided dish
    llm_dish = str(recipe.get("dish_name", "")).strip().lower()
    decided_dish_clean = decided_dish.strip().lower()
    if llm_dish != decided_dish_clean:
        failed.append(f"Check 2: dish_name '{recipe.get('dish_name')}' != decided '{decided_dish}'")

    # 3. Every ingredient in allowed ingredients, user's list, pantry list, or staples
    stored_items = {
        ing.get("item", "").strip().lower()
        for ing in stored_recipe.get("ingredients_with_quantity", [])
    }
    allowed_item_set = (
        stored_items
        | set(u.strip().lower() for u in user_ingredients)
        | set(p.strip().lower() for p in pantry)
        | STAPLES
    )

    llm_ingredients = recipe.get("ingredients", [])
    invented = []
    for ing in llm_ingredients:
        item = str(ing.get("item", "")).strip().lower()
        # relaxed match: item token in allowed set
        match = any(item in a or a in item for a in allowed_item_set)
        if not match:
            invented.append(item)
    if invented:
        failed.append(f"Check 3: Invented ingredients not in allowed set: {invented}")

    # 4. Quantities match stored quantities scaled to servings
    base_servings = float(stored_recipe.get("servings", 2.0))
    stored_qty_map = {
        ing.get("item", "").strip().lower(): scale_quantity(ing.get("qty", ""), base_servings, float(servings))
        for ing in stored_recipe.get("ingredients_with_quantity", [])
    }
    mismatches = 0
    total_compared = 0
    for ing in llm_ingredients:
        item = str(ing.get("item", "")).strip().lower()
        llm_q = str(ing.get("quantity", "")).strip().lower()
        if item in stored_qty_map:
            total_compared += 1
            expected_q = stored_qty_map[item].strip().lower()
            if llm_q != expected_q:
                mismatches += 1

    if total_compared > 0:
        qty_mismatch_rate = round(mismatches / total_compared, 3)
    # Check 4 is a warning/metric per spec; records if mismatch_rate > 0.30

    # 5. No allergen or excluded item anywhere, including steps
    full_text = json.dumps(recipe).lower()
    allergens_clean = [a.strip().lower() for a in allergens if a.strip()]
    for alg in allergens_clean:
        pattern = r"\b" + re.escape(alg) + r"\b"
        if re.search(pattern, full_text):
            failed.append(f"Check 5: Allergen '{alg}' detected in recipe text")

    # 6. Every known ingredient name mentioned in the steps is in the allowed set
    all_steps_text = " ".join(steps).lower()
    for inv in invented:
        pattern = r"\b" + re.escape(inv) + r"\b"
        if re.search(pattern, all_steps_text):
            failed.append(f"Check 6: Invented ingredient '{inv}' used in cooking steps")

    # 7. time_minutes within 50% of stored time
    stored_time = int(stored_recipe.get("time_minutes", 30))
    llm_time = recipe.get("time_minutes")
    try:
        llm_time_int = int(llm_time)
        if abs(llm_time_int - stored_time) > 0.5 * stored_time:
            failed.append(f"Check 7: time_minutes ({llm_time_int}) deviates > 50% from stored ({stored_time})")
    except (TypeError, ValueError):
        failed.append(f"Check 7: time_minutes '{llm_time}' is not a valid integer")

    # 8. Meat dishes: steps must themselves mention cooking through
    meat_type = stored_recipe.get("meat_type", "none").strip().lower()
    if meat_type != "none":
        cook_through_keywords = ("tender", "cooked through", "cooked thoroughly", "safe internal", "simmer")
        if not any(kw in all_steps_text for kw in cook_through_keywords):
            failed.append("Check 8: Non-veg dish steps missing cook-through safety step")

    # 9. Replace missing_ingredients with code-computed list
    code_missing = compute_missing_ingredients(stored_recipe, pantry)
    recipe["missing_ingredients"] = code_missing

    is_passed = (len(failed) == 0)
    return is_passed, recipe, failed, qty_mismatch_rate


def generate_recipe(
    dish: Dict[str, Any],
    stored: Dict[str, Any],
    pantry: List[str],
    allergens: Optional[List[str]] = None,
    servings: int = 2,
    max_time: int = 0,
    prompt_version: str = "B",
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate recipe from model with decision header, execute 9 checks, fallback on error.
    """
    allergens = allergens or []
    dish_name = dish.get("dish_name", "")
    dish_id = dish.get("dish_id", dish_name.lower().replace(" ", "_"))

    # Compute code missing ingredients upfront
    code_missing = compute_missing_ingredients(stored, pantry)

    # Base fallback response structure
    fallback_recipe = {
        "dish_name": stored.get("dish_name", dish_name),
        "local_name": stored.get("local_name", dish_name),
        "ingredients": [
            {"item": ing["item"], "quantity": scale_quantity(ing["qty"], float(stored.get("servings", 2)), float(servings))}
            for ing in stored.get("ingredients_with_quantity", [])
        ],
        "steps": stored.get("steps", []),
        "time_minutes": stored.get("time_minutes", 30),
        "missing_ingredients": code_missing
    }

    base_response = {
        "recipe": fallback_recipe,
        "source": "fallback",
        "failed_checks": [],
        "prompt_header": None,
        "protein_estimate_g": dish.get("protein_estimate_g", 0.0),
        "protein_confidence": dish.get("protein_confidence", "low"),
        "verified": dish.get("verified", False),
        "quantity_mismatch_rate": 0.0
    }

    if not is_llm_enabled():
        base_response["failed_checks"] = ["LLM_ENABLED is False (running in fallback mode)"]
        return base_response

    # Build prompt
    header = build_decision_header(dish, stored, pantry, allergens, servings, max_time)
    trained_body = build_trained_user_message(dish, pantry, allergens, max_time)

    if prompt_version == "B":
        full_user_msg = header + trained_body
    else:
        full_user_msg = trained_body

    if debug:
        base_response["prompt_header"] = header

    messages = [
        {"role": "system", "content": FIXED_SYSTEM_PROMPT},
        {"role": "user", "content": full_user_msg}
    ]

    # Acquire lock and run generation with 30s timeout
    acquired = _model_lock.acquire(timeout=30.0)
    if not acquired:
        base_response["source"] = "fallback"
        base_response["failed_checks"] = ["Timeout waiting for model lock (server busy)"]
        return base_response

    raw_response = ""
    try:
        model, tokenizer = load_writer_model()
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        import torch
        inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        raw_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    except Exception as e:
        base_response["failed_checks"] = [f"Model generation error: {e}"]
        return base_response
    finally:
        _model_lock.release()

    # Run the 9 strict checks
    passed, llm_recipe, failed_checks, qty_mismatch = run_9_checks(
        raw_output=raw_response,
        decided_dish=dish_name,
        stored_recipe=stored,
        user_ingredients=pantry,
        pantry=pantry,
        allergens=allergens,
        servings=servings
    )

    base_response["quantity_mismatch_rate"] = qty_mismatch

    if passed and llm_recipe is not None:
        base_response["recipe"] = llm_recipe
        base_response["source"] = "llm"
        base_response["failed_checks"] = []
    else:
        base_response["source"] = "fallback"
        base_response["failed_checks"] = failed_checks

    return base_response

