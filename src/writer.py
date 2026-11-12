"""
src/writer.py — Layer 3 Recipe Writer (v2).
NO FastAPI imports allowed in this module.

Redesign:
- Code builds ingredients (scaled quantities), time_minutes, missing_ingredients,
  dish_name, local_name from the stored catalog row. The LLM never outputs these.
- LLM output = plain text, one numbered step per line. No JSON. Parse by splitting lines.
- Model: base Qwen2.5-1.5B-Instruct, NO adapter, fp16/bf16, greedy decoding,
  max_new_tokens 350. Load lazily once.
- Prompt: dish name, allowed ingredients with quantities, stored steps as reference,
  and avoid list. Asks for 4-8 clear steps using ONLY the listed ingredients.
- LLM_MODE = off | base. Default off.
- Final recipe JSON assembled by code, identical response fields as before.
- Fallback: any failure, missing model, CUDA error, or LLM_MODE=off returns
  stored recipe with source "stored". Never returns 500.
"""

import os
import re
import json
import logging
import traceback
from fractions import Fraction
from typing import Dict, Any, List, Set, Tuple, Optional, Union

from src.scorer import canonical_name, parse_ingredient_line_fixed
from src.validators import (
    validate_steps,
    load_catalog_ingredient_vocabulary,
    SPICE_BLENDS,
    ALLERGEN_TERMS,
    normalize_item_name
)

logger = logging.getLogger("src.writer")


def format_scaled_quantity(qty_str: str, scale: float) -> str:
    """Scale numeric quantity by scale factor."""
    if not qty_str:
        return ""
    s = str(qty_str).strip()
    if any(q in s.lower() for q in ["to taste", "as needed", "pinch", "a pinch"]):
        return s

    m_mixed = re.match(r"^(\d+)\s+(\d+)/(\d+)\s*(.*)$", s)
    if m_mixed:
        whole, num, den, unit = m_mixed.groups()
        val = (float(whole) + float(num) / float(den)) * scale
        frac = Fraction(val).limit_denominator(8)
        if frac.denominator == 1:
            q_str = str(frac.numerator)
        elif frac.numerator > frac.denominator:
            w = frac.numerator // frac.denominator
            rem = frac.numerator % frac.denominator
            q_str = f"{w} {rem}/{frac.denominator}"
        else:
            q_str = f"{frac.numerator}/{frac.denominator}"
        return f"{q_str} {unit}".strip()

    m_frac = re.match(r"^(\d+)/(\d+)\s*(.*)$", s)
    if m_frac:
        num, den, unit = m_frac.groups()
        val = (float(num) / float(den)) * scale
        frac = Fraction(val).limit_denominator(8)
        if frac.denominator == 1:
            q_str = str(frac.numerator)
        else:
            q_str = f"{frac.numerator}/{frac.denominator}"
        return f"{q_str} {unit}".strip()

    m_num = re.match(r"^([\d\.]+)\s*(.*)$", s)
    if m_num:
        val_str, unit = m_num.groups()
        try:
            val = float(val_str) * scale
            val_round = round(val, 2)
            if val_round == int(val_round):
                q_str = str(int(val_round))
            else:
                q_str = f"{val_round:g}"
            return f"{q_str} {unit}".strip()
        except ValueError:
            pass

    return s


def parse_numbered_steps(raw_text: str) -> List[str]:
    """
    Parses plain text LLM output into a clean list of individual steps.
    Expects one numbered step per line.
    """
    if not raw_text:
        return []
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    steps = []
    for line in lines:
        # Match lines like "1. ", "1) ", "Step 1: ", "1: ", "- 1. ", "5 - "
        m = re.match(r"^(?:[-*]\s*)?(?:step\s*)?\d+\s*[\.\)\:\-]?\s+(.*)$", line, re.IGNORECASE)
        if m:
            clean = m.group(1).strip()
            # Remove any wrapping quotes or markdown bold
            clean = re.sub(r"^\*\*|\*\*$", "", clean).strip()
            if clean:
                steps.append(clean)
        elif steps and not re.match(r"^(?:note|tip|enjoy|serve|here\s+are)", line, re.IGNORECASE):
            # Continuation line of previous step if short
            clean = re.sub(r"^\*\*|\*\*$", "", line).strip()
            if len(steps[-1].split()) + len(clean.split()) <= 40:
                steps[-1] = f"{steps[-1]} {clean}"

    return steps


class RecipeWriter:
    """
    Layer 3 Recipe Writer (v2).
    Generates plain-text numbered steps using base Qwen2.5-1.5B-Instruct in fp16/bf16,
    or falls back to verified stored catalog recipes.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        staples_path: str = "data/staples.json",
        catalog_path: str = "data/raw/maharashtra_drafts.csv"
    ):
        self.base_model_name = base_model_name
        self.staples_path = staples_path
        self.catalog_path = catalog_path
        self.staples = self._load_staples()
        self.catalog_vocab = load_catalog_ingredient_vocabulary(self.catalog_path)

        # Lazy model state
        self.model = None
        self.tokenizer = None
        self._current_loaded_mode = None

    def _load_staples(self) -> Set[str]:
        if os.path.exists(self.staples_path):
            try:
                with open(self.staples_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {canonical_name(x) for x in data if x}
            except Exception as e:
                logger.warning(f"Error loading staples from {self.staples_path}: {e}")
        return {
            "salt", "oil", "water", "turmeric", "red chilli powder", "mustard seeds",
            "cumin seeds", "green chilli", "ginger", "garlic", "curry leaves",
            "coriander leaves", "lemon juice", "hing", "sugar", "ghee"
        }

    def load_model(self, mode: str = "base") -> bool:
        """
        Lazily loads base model in native fp16/bf16.
        Returns True on success, False on failure.
        """
        mode = mode.lower().strip()
        if mode == "off":
            return False

        if self.model is not None and self._current_loaded_mode == mode:
            return True

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            logger.info(f"Loading tokenizer: {self.base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                local_files_only=True
            )

            logger.info(f"Loading base model in {dtype} on {device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True
            )

            self._current_loaded_mode = mode
            logger.info(f"Base model loaded successfully on {device}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load LLM base model: {traceback.format_exc()}")
            self.model = None
            self.tokenizer = None
            self._current_loaded_mode = None
            return False

    def generate_raw_steps(self, prompt: str, max_new_tokens: int = 350) -> str:
        """Invokes LLM with greedy decoding and returns plain text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Indian chef. Write cooking steps for the given dish using "
                    "strictly the allowed ingredients. Follow all constraints precisely."
                )
            },
            {"role": "user", "content": prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_ids
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def get_stored_recipe(
        self,
        dish: Dict[str, Any],
        servings: float = 2.0,
        user_ingredients: Optional[List[str]] = None,
        pantry: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Assembles verified recipe from catalog row.
        Scales quantities to requested servings.
        Provides robust default steps if catalog steps are missing/nan.
        """
        raw_row = dish.get("raw_row", {})
        dish_name = dish.get("name") or raw_row.get("dish_name", "Unknown Dish")
        local_name = raw_row.get("romanized_local_name") or dish.get("local_name") or dish_name

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
            # High-quality default steps adhering to 5-40 words rule
            steps = [
                f"Clean, wash, and prepare all the listed ingredients for {dish_name} according to standard regional method.",
                "Heat the pan or kadai with cooking oil and temper with the spices until aromatic.",
                "Add the primary seasoned ingredients and cook covered on low to medium flame until tender and thoroughly infused.",
                "Garnish with freshly chopped coriander leaves and serve warm with authentic accompaniment."
            ]

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
        Layer 3 recipe generation flow:
        - Code builds ingredients, time_minutes, missing_ingredients, dish_name, local_name.
        - If LLM_MODE == 'off': returns stored recipe immediately.
        - If LLM_MODE == 'base': LLM generates plain-text numbered steps.
        - Validates steps (3-10 steps, 5-40 words, no unallowed ingredients, no allergens, spice blends only if allowed).
        - On validation pass: returns assembled recipe with source 'llm'.
        - On failure, missing model, CUDA error, or exception: returns stored recipe with source 'stored'. Never 500.
        """
        raw_row = dish.get("raw_row", {})
        demo_ready = str(raw_row.get("demo_ready", "")).strip().lower()
        recipe_status = "reviewed" if demo_ready == "yes" else "unreviewed"

        # Code-built stored recipe
        stored_recipe = self.get_stored_recipe(
            dish=dish,
            servings=servings,
            user_ingredients=user_ingredients,
            pantry=pantry
        )

        llm_mode = (mode_override or os.getenv("LLM_MODE", "off")).lower().strip()

        # If forced failure is requested for testing resilience
        if llm_mode in ["force_failure", "force_model_failure"]:
            logger.warning("Forced model failure requested for verification.")
            return {
                "recipe": stored_recipe,
                "source": "stored",
                "recipe_status": recipe_status,
                "validation": {"passed": False, "failures": ["Simulated model failure: RuntimeError('CUDA Out Of Memory')"]},
                "warnings": ["Model unavailable or error encountered: returning stored catalog recipe (unreviewed)."]
            }

        # If LLM_MODE is off, return stored recipe directly
        if llm_mode == "off":
            return {
                "recipe": stored_recipe,
                "source": "stored",
                "recipe_status": recipe_status,
                "validation": {"passed": True, "failures": []},
                "warnings": ["LLM_MODE is off: returning stored catalog recipe (unreviewed)."]
            }

        # Build allowed ingredients set & catalog spice blends
        dish_name = stored_recipe["dish_name"]
        allowed_ingredients = set(self.staples)
        allowed_catalog_blends = set()

        for u in user_ingredients:
            allowed_ingredients.add(canonical_name(u))
        for p in (pantry or []):
            allowed_ingredients.add(canonical_name(p))
        for ing in stored_recipe["ingredients"]:
            c_name = canonical_name(ing["item"])
            allowed_ingredients.add(c_name)
            # Check if this item is a spice blend
            norm_item = normalize_item_name(ing["item"])
            for blend in SPICE_BLENDS:
                if blend in norm_item:
                    allowed_catalog_blends.add(blend)

        # Try loading base model
        try:
            loaded = self.load_model(mode=llm_mode)
            if not loaded or self.model is None:
                logger.warning(f"Could not load LLM in mode '{llm_mode}'. Falling back to stored catalog recipe.")
                return {
                    "recipe": stored_recipe,
                    "source": "stored",
                    "recipe_status": recipe_status,
                    "validation": {"passed": False, "failures": [f"Model loading failed for mode '{llm_mode}'"]},
                    "warnings": ["Model unavailable or error encountered: returning stored catalog recipe (unreviewed)."]
                }
        except Exception as e:
            logger.error(f"Exception loading model: {traceback.format_exc()}")
            return {
                "recipe": stored_recipe,
                "source": "stored",
                "recipe_status": recipe_status,
                "validation": {"passed": False, "failures": [f"Model loading exception: {str(e)}"]},
                "warnings": ["Model unavailable or error encountered: returning stored catalog recipe (unreviewed)."]
            }

        # Build prompt for LLM steps
        ing_lines = [f"- {i['item']}: {i['quantity']}" for i in stored_recipe["ingredients"]]
        ref_steps = "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(stored_recipe["steps"]))
        avoid_str = ", ".join(allergens) if allergens else "None"

        prompt = (
            f"Dish: {dish_name}\n\n"
            f"Allowed Ingredients (with quantities):\n" + "\n".join(ing_lines) + "\n\n"
            f"Reference Steps:\n{ref_steps}\n\n"
            f"Avoid List: {avoid_str}\n\n"
            f"Instructions:\n"
            f"1. Write 4 to 8 clear, numbered cooking steps (one step per line).\n"
            f"2. Each step must be between 5 and 40 words.\n"
            f"3. Use ONLY the listed allowed ingredients. Do NOT introduce or invent any new ingredients.\n"
            f"4. Do NOT use or mention any ingredient from the avoid list.\n"
            f"5. Output ONLY the numbered steps (1., 2., ...). Stop after the last step. Do NOT output any headings, JSON, or extra text."
        )

        try:
            raw_text = self.generate_raw_steps(prompt, max_new_tokens=350)
            parsed_steps = parse_numbered_steps(raw_text)

            # Validate steps
            passed, failures = validate_steps(
                steps=parsed_steps,
                allowed_ingredients=allowed_ingredients,
                catalog_vocab=self.catalog_vocab,
                excluded_allergens=allergens,
                allowed_catalog_blends=allowed_catalog_blends
            )

            if passed:
                # Assemble final recipe JSON with LLM steps
                llm_recipe = dict(stored_recipe)
                llm_recipe["steps"] = parsed_steps
                return {
                    "recipe": llm_recipe,
                    "source": "llm",
                    "recipe_status": recipe_status,
                    "validation": {"passed": True, "failures": []},
                    "warnings": []
                }
            else:
                logger.warning(f"LLM steps failed validation for {dish_name}: {failures}. Falling back to stored.")
                return {
                    "recipe": stored_recipe,
                    "source": "stored",
                    "recipe_status": recipe_status,
                    "validation": {"passed": False, "failures": failures},
                    "warnings": ["LLM generation failed validation: returning stored catalog recipe (unreviewed)."]
                }
        except Exception as e:
            logger.warning(f"LLM generation exception for {dish_name}: {traceback.format_exc()}")
            return {
                "recipe": stored_recipe,
                "source": "stored",
                "recipe_status": recipe_status,
                "validation": {"passed": False, "failures": [f"Generation exception: {str(e)}"]},
                "warnings": ["Model unavailable or error encountered: returning stored catalog recipe (unreviewed)."]
            }
