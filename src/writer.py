"""
src/writer.py — Recipe Writer & LLM Integrator.
NO FastAPI imports allowed in this module.

Provides:
- build_decision_header: constructs decision header by code from catalog row.
- format_scaled_quantity: scales numeric ingredient quantities by servings factor.
- RecipeWriter: handles lazy LLM loading (4-bit Qwen2.5-1.5B-Instruct + adapter),
  JSON generation, validation, one-shot retry, and guaranteed stored catalog fallback.
"""

import os
import sys
import re
import json
import time
import logging
import traceback
from typing import Dict, Any, List, Set, Tuple, Optional, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scorer import canonical_name, parse_ingredient_line_fixed
from src.validators import (
    validate_recipe,
    parse_quantity_number,
    normalize_item_name
)

logger = logging.getLogger("src.writer")


def format_scaled_quantity(original_qty_str: str, scale: float) -> str:
    """
    Scales the numeric portion of a quantity string by scale factor.
    E.g. '1.5 cup' with scale=2.0 -> '3 cup'
         '1/2 tsp' with scale=2.0 -> '1 tsp'
         'to taste' with scale=2.0 -> 'to taste'
    """
    if not original_qty_str:
        return ""
    s = str(original_qty_str).strip()
    if any(q in s.lower() for q in ["to taste", "as needed", "pinch"]):
        return s

    def _fmt(val: float) -> str:
        if abs(val - round(val)) < 1e-4:
            return str(int(round(val)))
        for frac_val, frac_str in [(0.25, "1/4"), (0.5, "1/2"), (0.75, "3/4"), (0.33, "1/3"), (0.67, "2/3")]:
            if abs(val - frac_val) < 0.05:
                return frac_str
        return f"{val:.2f}".rstrip("0").rstrip(".")

    # Mixed fraction: e.g. "1 1/2 tsp"
    m_mixed = re.match(r"^(\d+)\s+(\d+)/(\d+)\s*(.*)$", s)
    if m_mixed:
        whole, num, den, unit = m_mixed.groups()
        if float(den) != 0:
            val = (float(whole) + float(num) / float(den)) * scale
            return f"{_fmt(val)} {unit}".strip()

    # Simple fraction: e.g. "1/2 tsp"
    m_frac = re.match(r"^(\d+)/(\d+)\s*(.*)$", s)
    if m_frac:
        num, den, unit = m_frac.groups()
        if float(den) != 0:
            val = (float(num) / float(den)) * scale
            return f"{_fmt(val)} {unit}".strip()

    # Float/integer: e.g. "1.5 cup" or "2"
    m_num = re.match(r"^(\d+(?:\.\d+)?)\s*(.*)$", s)
    if m_num:
        num, unit = m_num.groups()
        val = float(num) * scale
        return f"{_fmt(val)} {unit}".strip()

    return s


def build_decision_header(
    dish: Dict[str, Any],
    user_ingredients: List[str],
    pantry: Optional[List[str]] = None,
    servings: float = 2.0,
    time_limit: Optional[int] = None,
    allergens: Optional[List[str]] = None
) -> str:
    """
    Constructs the DECISION header strictly from code using catalog row data.
    Format:
    [DECISION dish, user ingredients, missing items, servings, time limit, avoid list, allowed ingredients with quantities scaled to servings, style END DECISION]
    """
    raw_row = dish.get("raw_row", dish)
    dish_name = dish.get("name") or raw_row.get("dish_name", "Unknown Dish")

    # Base servings from catalog row
    base_servings = 2.0
    s_raw = str(raw_row.get("servings", "2")).strip()
    if s_raw.replace(".", "").isdigit() and float(s_raw) > 0:
        base_servings = float(s_raw)
    scale = float(servings) / base_servings if base_servings > 0 else 1.0

    pantry_set = {canonical_name(x) for x in (pantry or [])}
    user_set = {canonical_name(x) for x in user_ingredients}

    # Parse and scale catalog ingredients
    iq_str = raw_row.get("ingredients_with_quantity", "")
    iq_lines = [x.strip() for x in str(iq_str).split("|") if x.strip()]
    allowed_items = []
    for line in iq_lines:
        qty, item = parse_ingredient_line_fixed(line)
        scaled_q = format_scaled_quantity(qty, scale)
        allowed_items.append(f"{item}: {scaled_q}")

    # Determine missing non-staple ingredients
    main_raw = raw_row.get("main_ingredients", "")
    mains = [canonical_name(x) for x in re.split(r"[|,]", main_raw) if x.strip()]
    missing = [m for m in mains if m not in user_set and m not in pantry_set]

    user_ing_str = ", ".join(user_ingredients) if user_ingredients else "none"
    missing_str = ", ".join(missing) if missing else "none"
    avoid_str = ", ".join(allergens) if allergens else "none"
    effective_time = time_limit if time_limit and time_limit > 0 else int(dish.get("time_minutes", 30))
    allowed_str = ", ".join(allowed_items)

    servings_display = int(servings) if servings == int(servings) else servings
    header = (
        f"[DECISION dish: {dish_name}, "
        f"user ingredients: {user_ing_str}, "
        f"missing items: {missing_str}, "
        f"servings: {servings_display}, "
        f"time limit: {effective_time} minutes, "
        f"avoid list: {avoid_str}, "
        f"allowed ingredients with quantities scaled to servings: {allowed_str}, "
        f"style: concise step-by-step END DECISION]"
    )
    return header


class RecipeWriter:
    """
    Recipe Writer with lazy LLM loading, validator integration,
    one-shot retry on validation failure, and guaranteed stored recipe fallback.
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str = os.path.join(PROJECT_ROOT, "outputs", "pilot_unverified", "adapter"),
        staples_path: str = os.path.join(PROJECT_ROOT, "data", "staples.json")
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.staples_path = staples_path
        self.staples = self._load_staples()

        # Model state (loaded lazily)
        self.tokenizer = None
        self.model = None
        self._current_loaded_mode = None

    def _load_staples(self) -> Set[str]:
        if os.path.exists(self.staples_path):
            try:
                with open(self.staples_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {canonical_name(x) for x in data}
            except Exception:
                pass
        return {"salt", "oil", "water", "turmeric", "red chilli powder", "mustard seeds", "cumin seeds", "hing", "sugar", "ghee"}

    def get_stored_recipe(
        self,
        dish: Dict[str, Any],
        servings: float = 2.0,
        user_ingredients: Optional[List[str]] = None,
        pantry: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Constructs stored catalog recipe with quantities scaled to requested servings."""
        raw_row = dish.get("raw_row", dish)
        dish_name = dish.get("name") or raw_row.get("dish_name", "Unknown Dish")
        local_name = (
            dish.get("local_name")
            or raw_row.get("romanized_local_name")
            or raw_row.get("native_name")
            or dish_name
        )

        base_servings = 2.0
        s_raw = str(raw_row.get("servings", "2")).strip()
        if s_raw.replace(".", "").isdigit() and float(s_raw) > 0:
            base_servings = float(s_raw)
        scale = float(servings) / base_servings if base_servings > 0 else 1.0

        # Ingredients with scaled quantities
        iq_str = raw_row.get("ingredients_with_quantity", "")
        iq_lines = [x.strip() for x in str(iq_str).split("|") if x.strip()]
        ingredients = []
        for line in iq_lines:
            qty, item = parse_ingredient_line_fixed(line)
            scaled_q = format_scaled_quantity(qty, scale)
            ingredients.append({"item": item, "quantity": scaled_q})

        # Steps
        steps_raw = raw_row.get("steps", "")
        steps = [s.strip() for s in str(steps_raw).split("|") if s.strip()]
        if not steps:
            steps = ["Prepare ingredients according to standard regional method.", "Cook thoroughly and serve hot."]

        # Time
        time_mins = int(dish.get("time_minutes", 30)) if str(dish.get("time_minutes", "")).isdigit() else 30

        # Missing ingredients
        user_set = {canonical_name(x) for x in (user_ingredients or [])}
        pantry_set = {canonical_name(x) for x in (pantry or [])}
        main_raw = raw_row.get("main_ingredients", "")
        mains = [canonical_name(x) for x in re.split(r"[|,]", main_raw) if x.strip()]
        missing = [m for m in mains if m not in user_set and m not in pantry_set]

        return {
            "dish_name": dish_name,
            "local_name": local_name,
            "ingredients": ingredients,
            "steps": steps,
            "time_minutes": time_mins,
            "missing_ingredients": missing
        }

    def load_model(self, mode: str = "adapter") -> bool:
        """
        Lazily loads base or adapter model in 4-bit.
        Returns True on success, False on failure.
        """
        mode = mode.lower().strip()
        if mode == "off":
            return False

        if self.model is not None and self._current_loaded_mode == mode:
            return True

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        try:
            logger.info(f"Loading tokenizer: {self.base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

            logger.info(f"Loading base model in 4-bit: {self.base_model_name}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            if mode == "adapter":
                if not os.path.exists(self.adapter_path):
                    logger.error(f"Adapter path not found: {self.adapter_path}")
                    return False
                from peft import PeftModel
                logger.info(f"Loading LoRA adapter from {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                self.model = base_model

            self._current_loaded_mode = mode
            logger.info(f"Model successfully loaded in mode: {mode}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LLM model: {traceback.format_exc()}")
            self.model = None
            self.tokenizer = None
            self._current_loaded_mode = None
            return False

    def generate_raw_llm(self, prompt: str, system_prompt: Optional[str] = None, max_new_tokens: int = 360, temperature: float = 0.3) -> str:
        """Invokes LLM and returns raw generated text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        import torch
        sys_p = system_prompt or (
            "You are an expert Indian culinary assistant. Given the following decision constraints, "
            "output the authentic recipe in strict JSON format. Output valid JSON only, with no markdown fences, "
            "no preamble, and no explanation. The JSON must have keys: dish_name, local_name, ingredients "
            "(list of objects with 'item' and 'quantity'), steps (list of 3 to 10 strings), time_minutes (integer), "
            "missing_ingredients (list of strings)."
        )

        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": eos_ids
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def generate(
        self,
        dish: Dict[str, Any],
        user_ingredients: List[str],
        pantry: Optional[List[str]] = None,
        servings: float = 2.0,
        time_limit: Optional[int] = None,
        allergens: Optional[List[str]] = None,
        vegetarian_only: bool = False,
        mode_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Primary generation pipeline:
        1. If LLM_MODE is 'off', returns stored catalog recipe directly.
        2. If LLM_MODE is 'base' or 'adapter', builds decision header and runs LLM.
        3. Validates output; on failure retries once.
        4. If still failing, model missing, or CUDA error: returns stored catalog recipe.
        Guaranteed: Never raises unhandled exception.
        """
        raw_row = dish.get("raw_row", dish)
        demo_ready = str(raw_row.get("demo_ready", "")).strip().lower()
        recipe_status = "reviewed" if demo_ready == "yes" else "unreviewed"

        # Mode determination
        llm_mode = (mode_override or os.getenv("LLM_MODE", "off")).lower().strip()

        # If LLM_MODE is 'off', return stored recipe directly
        if llm_mode == "off":
            stored = self.get_stored_recipe(dish, servings=servings, user_ingredients=user_ingredients, pantry=pantry)
            return {
                "recipe": stored,
                "source": "stored",
                "recipe_status": recipe_status,
                "validation": {"passed": True, "failures": []},
                "warnings": ["LLM_MODE is off: returning stored catalog recipe (unreviewed)."]
            }

        # Build decision header and validator constraints
        header = build_decision_header(
            dish=dish,
            user_ingredients=user_ingredients,
            pantry=pantry,
            servings=servings,
            time_limit=time_limit,
            allergens=allergens
        )

        # Build allowed ingredients set & expected quantities
        raw_iq = raw_row.get("ingredients_with_quantity", "")
        iq_lines = [x.strip() for x in str(raw_iq).split("|") if x.strip()]

        base_servings = 2.0
        s_raw = str(raw_row.get("servings", "2")).strip()
        if s_raw.replace(".", "").isdigit() and float(s_raw) > 0:
            base_servings = float(s_raw)
        scale = float(servings) / base_servings if base_servings > 0 else 1.0

        expected_quantities = {}
        allowed_ingredients = set(self.staples)
        allowed_catalog_blends = set()

        for u in user_ingredients:
            allowed_ingredients.add(canonical_name(u))
        for p in (pantry or []):
            allowed_ingredients.add(canonical_name(p))

        for line in iq_lines:
            qty, item = parse_ingredient_line_fixed(line)
            scaled_q = format_scaled_quantity(qty, scale)
            c_item = canonical_name(item)
            allowed_ingredients.add(c_item)
            expected_quantities[c_item] = scaled_q
            # Check if this item was a catalog spice blend
            norm_item = normalize_item_name(item)
            for blend in ["garam masala", "pav bhaji masala", "godha masala", "sambhar powder", "biryani masala", "chaat masala"]:
                if blend in norm_item:
                    allowed_catalog_blends.add(blend)

        dish_name = dish.get("name") or raw_row.get("dish_name", "Unknown Dish")

        # Try loading model
        try:
            loaded = self.load_model(mode=llm_mode)
            if not loaded or self.model is None:
                logger.warning(f"Could not load LLM in mode '{llm_mode}'. Falling back to stored catalog recipe.")
                stored = self.get_stored_recipe(dish, servings=servings, user_ingredients=user_ingredients, pantry=pantry)
                return {
                    "recipe": stored,
                    "source": "stored",
                    "recipe_status": recipe_status,
                    "validation": {"passed": False, "failures": [f"Model loading failed for mode '{llm_mode}'"]},
                    "warnings": [f"Model unavailable for '{llm_mode}'. Fallback to stored recipe."]
                }
        except Exception as e:
            logger.error(f"Exception loading model: {traceback.format_exc()}")
            stored = self.get_stored_recipe(dish, servings=servings, user_ingredients=user_ingredients, pantry=pantry)
            return {
                "recipe": stored,
                "source": "stored",
                "recipe_status": recipe_status,
                "validation": {"passed": False, "failures": [f"Model loading exception: {str(e)}"]},
                "warnings": ["Model exception during load. Fallback to stored recipe."]
            }

        # Attempt 1: Generate with decision header
        prompt_1 = (
            f"Decision Header:\n{header}\n\n"
            f"Please generate the authentic recipe for '{dish_name}' in strict JSON format. "
            f"Follow all constraints, strictly use the allowed ingredients with their scaled quantities, "
            f"and adhere to the avoid list."
        )

        validation_failures = []
        try:
            raw_out_1 = self.generate_raw_llm(prompt_1, temperature=0.2)
            passed_1, parsed_recipe_1, failures_1 = validate_recipe(
                recipe_obj=raw_out_1,
                expected_dish=dish_name,
                allowed_ingredients=allowed_ingredients,
                expected_quantities=expected_quantities,
                excluded_allergens=allergens,
                time_limit=time_limit,
                allowed_catalog_blends=allowed_catalog_blends,
                tolerance=0.10
            )
            if passed_1:
                return {
                    "recipe": parsed_recipe_1,
                    "source": "llm",
                    "recipe_status": recipe_status,
                    "validation": {"passed": True, "failures": []},
                    "warnings": []
                }
            validation_failures = failures_1
            logger.info(f"Attempt 1 validation failed for {dish_name}: {failures_1}. Retrying once...")
        except Exception as e:
            logger.warning(f"Attempt 1 generation exception: {traceback.format_exc()}")
            validation_failures = [f"Generation exception: {str(e)}"]

        # Attempt 2: One-shot retry with error feedback
        try:
            prompt_2 = (
                f"Decision Header:\n{header}\n\n"
                f"Your previous recipe generation for '{dish_name}' failed validation due to the following errors:\n"
                + "\n".join(f"- {f}" for f in validation_failures) + "\n\n"
                f"Please fix all errors and output the corrected recipe in valid JSON format only."
            )
            raw_out_2 = self.generate_raw_llm(prompt_2, temperature=0.1)
            passed_2, parsed_recipe_2, failures_2 = validate_recipe(
                recipe_obj=raw_out_2,
                expected_dish=dish_name,
                allowed_ingredients=allowed_ingredients,
                expected_quantities=expected_quantities,
                excluded_allergens=allergens,
                time_limit=time_limit,
                allowed_catalog_blends=allowed_catalog_blends,
                tolerance=0.10
            )
            if passed_2:
                return {
                    "recipe": parsed_recipe_2,
                    "source": "llm",
                    "recipe_status": recipe_status,
                    "validation": {"passed": True, "failures": []},
                    "warnings": ["Recipe generated by LLM after 1 correction retry."]
                }
            validation_failures = failures_2
            logger.warning(f"Attempt 2 validation also failed for {dish_name}: {failures_2}. Falling back to stored.")
        except Exception as e:
            logger.warning(f"Attempt 2 generation exception: {traceback.format_exc()}")
            validation_failures.append(f"Retry exception: {str(e)}")

        # Fallback to stored recipe
        stored = self.get_stored_recipe(dish, servings=servings, user_ingredients=user_ingredients, pantry=pantry)
        return {
            "recipe": stored,
            "source": "stored",
            "recipe_status": recipe_status,
            "validation": {"passed": False, "failures": validation_failures},
            "warnings": ["LLM generation failed validation: returning stored catalog recipe (unreviewed)."]
        }
