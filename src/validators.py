"""
src/validators.py — Step-focused recipe validators for Layer 3 (v2).
NO FastAPI imports allowed in this module.

Validators for recipe steps:
1. 3-10 steps: Recipe steps count must be between 3 and 10 inclusive.
2. 5-40 words per step: Each individual step must contain between 5 and 40 words.
3. No unallowed catalog ingredients: No ingredient name from the catalog vocabulary
   appears in a step unless it is in the allowed list (dish ingredients + staples).
4. No allergen words: Zero occurrences of excluded allergen terms in any step.
5. Spice blends only if allowed: Commercial/complex spice blends cannot appear in any step
   unless they are present in the dish's catalog ingredients.
"""

import os
import re
import pandas as pd
from typing import Dict, Any, List, Set, Tuple, Optional, Union

# Common commercial/complex spice blends that are NOT free staples
SPICE_BLENDS = {
    "garam masala", "pav bhaji masala", "godha masala", "goda masala",
    "sambhar powder", "sambar powder", "biryani masala", "chaat masala",
    "kitchen king", "chana masala powder", "rasam powder", "curry powder",
    "tandoori masala", "panch phoron", "kashmiri masala", "kala masala"
}

# Rich allergen synonyms and derivative food terms
ALLERGEN_TERMS = {
    "dairy": [
        "dairy", "milk", "butter", "cheese", "ghee", "paneer", "curd", "yogurt",
        "yoghurt", "cream", "malai", "whey", "makkhan", "khoya", "mawa", "buttermilk"
    ],
    "gluten": [
        "gluten", "wheat", "flour", "maida", "atta", "suji", "sooji", "semolina",
        "rava", "bread", "roti", "chapati", "naan", "puri", "bhatura", "pav"
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

# Cached catalog vocabulary
_CATALOG_VOCAB_CACHE: Optional[Set[str]] = None


def normalize_item_name(name: str) -> str:
    """Normalize ingredient name for matching."""
    if not name:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[\(\)\[\],]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_catalog_ingredient_vocabulary(
    csv_path: str = "data/raw/maharashtra_drafts.csv"
) -> Set[str]:
    """
    Builds the vocabulary of known ingredient names from the catalog.
    Extracts distinct ingredient terms across all catalog dishes.
    """
    global _CATALOG_VOCAB_CACHE
    if _CATALOG_VOCAB_CACHE is not None:
        return _CATALOG_VOCAB_CACHE

    vocab = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                # From main ingredients
                for m in str(r.get("main_ingredients", "")).split("|"):
                    m_clean = normalize_item_name(m)
                    if m_clean and len(m_clean) > 2:
                        vocab.add(m_clean)
                # From ingredients with quantity
                iq_str = str(r.get("ingredients_with_quantity", ""))
                for line in iq_str.split("|"):
                    parts = line.split(":", 1)
                    item = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                    item_clean = normalize_item_name(item)
                    if item_clean and len(item_clean) > 2:
                        vocab.add(item_clean)
        except Exception:
            pass

    # Core base ingredient nouns that must be recognized
    core_nouns = {
        "potato", "potatoes", "tomato", "tomatoes", "onion", "onions", "paneer",
        "chicken", "fish", "mutton", "pork", "egg", "eggs", "poha", "rava",
        "sooji", "suji", "besan", "rice", "matki", "moong", "chana", "peanuts",
        "peanut", "coconut", "farsan", "jowar", "atta", "cashew", "cashews",
        "brinjal", "brinjals", "eggplant", "capsicum", "carrot", "carrots",
        "peas", "tamarind", "sabudana", "curd", "yogurt", "milk", "cheese",
        "butter", "ghee", "garlic", "ginger", "pav", "colocasia", "sev",
        "mustard seeds", "cumin seeds", "curry leaves", "coriander leaves",
        "green chillies", "green chilli", "turmeric", "red chilli powder",
        "oil", "salt", "sugar", "water"
    }
    vocab.update(core_nouns)

    # Filter out common adjectives or cooking terms
    non_ingredients = {
        "chopped", "finely", "fresh", "dry", "boiled", "sliced", "crushed",
        "grated", "roasted", "fried", "slit", "large", "medium", "small",
        "minced", "thinly", "pureed", "diced", "peeled", "tender", "round",
        "sprouted", "soaked", "prepared", "toasting", "deep", "shallow", "pan",
        "leaves", "seeds", "powder", "paste", "thick", "thin", "soft", "sweet",
        "sour", "raw", "hot", "warm", "cold", "white", "purple", "ripe"
    }
    vocab = {x for x in vocab if x not in non_ingredients and len(x) > 2}

    _CATALOG_VOCAB_CACHE = vocab
    return _CATALOG_VOCAB_CACHE


def validate_steps_count(steps: List[str]) -> Tuple[bool, Optional[str]]:
    """Validator: 3-10 steps."""
    if not isinstance(steps, list):
        return False, "Steps must be a list of strings"
    count = len(steps)
    if count < 3 or count > 10:
        return False, f"Step count must be between 3 and 10, got {count}"
    return True, None


def validate_step_word_counts(steps: List[str]) -> Tuple[bool, List[str]]:
    """Validator: each step must be between 5 and 40 words."""
    failures = []
    for idx, s in enumerate(steps):
        if not isinstance(s, str):
            failures.append(f"Step {idx + 1} is not a string")
            continue
        words = s.strip().split()
        word_count = len(words)
        if word_count < 5:
            failures.append(f"Step {idx + 1} has only {word_count} words; must be 5-40 words")
        elif word_count > 40:
            failures.append(f"Step {idx + 1} has {word_count} words; exceeds 40 words maximum")
    return (len(failures) == 0), failures


def validate_step_ingredients(
    steps: List[str],
    allowed_ingredients: Set[str],
    catalog_vocab: Optional[Set[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validator: no catalog ingredient name appears in a step unless it is in the allowed list.
    allowed_ingredients includes dish ingredients + staples + user pantry.
    """
    failures = []
    if catalog_vocab is None:
        catalog_vocab = load_catalog_ingredient_vocabulary()

    norm_allowed = {normalize_item_name(x) for x in allowed_ingredients if x}
    # Universal kitchen elements always permissible in cooking methods
    norm_allowed.update({"water", "salt"})

    # Sort vocabulary by length descending so compound items match first
    sorted_vocab = sorted(catalog_vocab, key=lambda x: len(x), reverse=True)

    for idx, step in enumerate(steps):
        step_lower = step.lower()
        found_unallowed_in_step = set()

        for term in sorted_vocab:
            # Match word boundary
            # Support plural forms ending in s or es
            pattern = r"\b" + re.escape(term) + r"(?:es|s)?\b"
            if re.search(pattern, step_lower):
                # Check if term or stem is allowed
                is_allowed = False
                for a in norm_allowed:
                    if term in a or a in term:
                        is_allowed = True
                        break
                    if term.endswith("s") and term[:-1] in a:
                        is_allowed = True
                        break
                    if a.endswith("s") and a[:-1] in term:
                        is_allowed = True
                        break

                if not is_allowed:
                    # Avoid flagging substring if a longer compound was already matched
                    if not any(term in already for already in found_unallowed_in_step):
                        found_unallowed_in_step.add(term)
                        failures.append(
                            f"Step {idx + 1} contains unallowed ingredient: '{term}' not in allowed ingredients"
                        )

    return (len(failures) == 0), failures


def validate_step_allergens(
    steps: List[str],
    excluded_allergens: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Validator: zero allergen words appear in any step."""
    failures = []
    if not excluded_allergens:
        return True, []

    for idx, step in enumerate(steps):
        step_lower = step.lower()
        for allergen in excluded_allergens:
            a_clean = allergen.lower().strip()
            if not a_clean:
                continue
            terms = ALLERGEN_TERMS.get(a_clean, [a_clean])
            if a_clean not in terms:
                terms.append(a_clean)

            for term in terms:
                pattern = r"\b" + re.escape(term) + r"(?:es|s)?\b"
                if re.search(pattern, step_lower):
                    failures.append(
                        f"Allergen violation in step {idx + 1}: contains '{term}' matching excluded allergen '{allergen}'"
                    )
                    break

    return (len(failures) == 0), failures


def validate_step_spice_blends(
    steps: List[str],
    allowed_catalog_blends: Optional[Set[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validator: spice blends appear only if allowed.
    Commercial spice blends cannot appear unless they were in the dish's catalog ingredients.
    """
    failures = []
    norm_allowed_blends = {normalize_item_name(x) for x in (allowed_catalog_blends or set())}

    for idx, step in enumerate(steps):
        step_lower = step.lower()
        for blend in SPICE_BLENDS:
            pattern = r"\b" + re.escape(blend) + r"\b"
            if re.search(pattern, step_lower):
                if not any(blend in allowed for allowed in norm_allowed_blends):
                    failures.append(
                        f"Step {idx + 1} mentions spice blend '{blend}' which is not in allowed catalog ingredients"
                    )

    return (len(failures) == 0), failures


def validate_steps(
    steps: List[str],
    allowed_ingredients: Set[str],
    catalog_vocab: Optional[Set[str]] = None,
    excluded_allergens: Optional[List[str]] = None,
    allowed_catalog_blends: Optional[Set[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Master step validator combining all 5 step checks:
    1. 3-10 steps
    2. Each 5-40 words
    3. No unallowed catalog ingredients
    4. No allergen words
    5. Spice blends only if allowed
    """
    failures: List[str] = []

    # 1. Step count (3-10)
    ok_count, count_err = validate_steps_count(steps)
    if not ok_count and count_err:
        failures.append(count_err)

    # 2. Word count per step (5-40)
    ok_words, word_errs = validate_step_word_counts(steps)
    failures.extend(word_errs)

    # 3. No unallowed catalog ingredients
    ok_ing, ing_errs = validate_step_ingredients(steps, allowed_ingredients, catalog_vocab=catalog_vocab)
    failures.extend(ing_errs)

    # 4. No allergen words
    ok_allergens, allergen_errs = validate_step_allergens(steps, excluded_allergens=excluded_allergens)
    failures.extend(allergen_errs)

    # 5. Spice blends only if allowed
    ok_spices, spice_errs = validate_step_spice_blends(steps, allowed_catalog_blends=allowed_catalog_blends)
    failures.extend(spice_errs)

    return (len(failures) == 0), failures
