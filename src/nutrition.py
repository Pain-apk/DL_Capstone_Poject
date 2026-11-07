"""
nutrition.py - Protein data lookup backed by verified IFCT 2017 & USDA tables.
Rules:
- No guessed values.
- Source tracking for every single item ('IFCT 2017' or 'USDA').
- Export to data/processed/protein_lookup.csv.
- Unmatched items logged to data/processed/unmatched_protein_items.csv.
"""

import os
import sys
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Optional, Dict, Tuple, Set

# Verified nutritional records (protein in grams per 100g edible portion)
# Sources:
# - IFCT 2017: Indian Food Composition Tables (ICMR - National Institute of Nutrition, Hyderabad)
# - USDA: USDA FoodData Central
REFERENCE_PROTEIN_DATA = [
    # Cereals & Millets
    ("rice", 7.94, "IFCT 2017", "Raw milled rice"),
    ("poha", 6.72, "IFCT 2017", "Flattened rice / aval"),
    ("atta", 10.57, "IFCT 2017", "Whole wheat flour"),
    ("wheat flour", 10.57, "IFCT 2017", "Whole wheat flour"),
    ("maida", 10.36, "IFCT 2017", "Refined wheat flour"),
    ("suji", 10.40, "IFCT 2017", "Semolina / rava"),
    ("rava", 10.40, "IFCT 2017", "Semolina"),
    ("jowar", 10.40, "IFCT 2017", "Sorghum grain / flour"),
    ("bajra", 11.60, "IFCT 2017", "Pearl millet"),
    ("ragi", 7.30, "IFCT 2017", "Finger millet / nachni"),

    # Pulses & Legumes
    ("besan", 22.30, "IFCT 2017", "Bengal gram flour dry"),
    ("chana dal", 21.55, "IFCT 2017", "Split Bengal gram"),
    ("toor dal", 22.30, "IFCT 2017", "Red gram / split pigeon pea"),
    ("moong dal", 23.86, "IFCT 2017", "Split green gram"),
    ("urad dal", 22.96, "IFCT 2017", "Split black gram"),
    ("masoor dal", 24.30, "IFCT 2017", "Split red lentil"),
    ("chana", 20.50, "IFCT 2017", "Chickpeas / kabuli chana raw"),
    ("kala chana", 22.00, "IFCT 2017", "Brown chickpea"),
    ("rajma", 22.90, "IFCT 2017", "Kidney beans raw"),
    ("matki", 22.50, "IFCT 2017", "Moth beans raw"),
    ("moong whole", 24.00, "IFCT 2017", "Whole green gram"),
    ("chawli", 24.10, "IFCT 2017", "Cowpea / black-eyed peas"),
    ("kulith", 22.00, "IFCT 2017", "Horse gram"),
    ("soya chunks", 52.00, "IFCT 2017", "Defatted soy granules/chunks"),

    # Nuts & Oilseeds
    ("peanuts", 25.80, "IFCT 2017", "Groundnut raw"),
    ("roasted peanuts", 26.20, "IFCT 2017", "Groundnut roasted"),
    ("sesame seeds", 18.30, "IFCT 2017", "Til"),
    ("cashew", 18.20, "IFCT 2017", "Kaju raw"),
    ("coconut", 3.33, "IFCT 2017", "Fresh mature coconut kernel"),
    ("dry coconut", 6.88, "IFCT 2017", "Copra / dry coconut"),

    # Dairy
    ("milk", 3.26, "IFCT 2017", "Cow milk whole"),
    ("buffalo milk", 4.34, "IFCT 2017", "Buffalo milk whole"),
    ("curd", 3.10, "IFCT 2017", "Cow milk curd / dahi"),
    ("paneer", 18.86, "IFCT 2017", "Fresh cottage cheese"),
    ("butter", 0.85, "USDA", "Salted/unsalted butter"),
    ("ghee", 0.00, "IFCT 2017", "Clarified butter pure fat"),

    # Vegetables & Tubers
    ("potato", 1.60, "IFCT 2017", "Potato fresh tuber"),
    ("onion", 1.10, "IFCT 2017", "Red onion bulb"),
    ("tomato", 0.88, "IFCT 2017", "Ripe red tomato"),
    ("green peas", 7.20, "IFCT 2017", "Fresh green peas"),
    ("cauliflower", 1.90, "IFCT 2017", "Fresh florets"),
    ("cabbage", 1.30, "IFCT 2017", "Fresh cabbage"),
    ("spinach", 2.00, "IFCT 2017", "Palak fresh"),
    ("fenugreek leaves", 4.40, "IFCT 2017", "Methi leaves fresh"),
    ("bottle gourd", 0.60, "IFCT 2017", "Lauki / dudhi"),
    ("bitter gourd", 1.65, "IFCT 2017", "Karela"),
    ("okra", 1.90, "IFCT 2017", "Bhindi / lady finger"),
    ("eggplant", 1.40, "IFCT 2017", "Baingan / brinjal"),
    ("capsicum", 1.30, "IFCT 2017", "Green bell pepper"),

    # Herbs, Spices & Seasonings
    ("green chilli", 1.87, "IFCT 2017", "Green chillies"),
    ("garlic", 6.36, "IFCT 2017", "Garlic cloves"),
    ("ginger", 1.82, "IFCT 2017", "Ginger rhizome fresh"),
    ("coriander leaves", 3.33, "IFCT 2017", "Fresh cilantro"),
    ("curry leaves", 6.10, "IFCT 2017", "Fresh curry leaves"),
    ("mustard seeds", 20.00, "IFCT 2017", "Rai / mustard seeds"),
    ("cumin seeds", 18.70, "IFCT 2017", "Jeera"),
    ("turmeric", 6.30, "IFCT 2017", "Turmeric powder / haldi"),
    ("red chilli powder", 13.40, "IFCT 2017", "Dried red chilli powder"),
    ("asafoetida", 4.00, "IFCT 2017", "Hing compound"),
    ("coriander powder", 12.30, "IFCT 2017", "Dhania powder"),
    ("lemon juice", 0.35, "USDA", "Fresh lemon juice"),
    ("lemon", 1.10, "USDA", "Whole lemon with pulp"),

    # Pantry & Staples (zero/negligible protein)
    ("oil", 0.00, "IFCT 2017", "Cooking vegetable oil"),
    ("salt", 0.00, "IFCT 2017", "Sodium chloride table salt"),
    ("sugar", 0.00, "IFCT 2017", "Refined white sugar"),
    ("jaggery", 0.40, "IFCT 2017", "Cane jaggery / gur"),
    ("water", 0.00, "USDA", "Potable water"),

    # Non-Vegetarian (for tiered/regional recipes)
    ("egg", 13.00, "IFCT 2017", "Whole hen egg"),
    ("chicken", 21.80, "IFCT 2017", "Broiler chicken meat"),
    ("mutton", 21.40, "IFCT 2017", "Goat meat"),
    ("fish", 19.50, "IFCT 2017", "Fresh fish generic edible flesh"),
    ("prawns", 19.30, "IFCT 2017", "Fresh marine prawns"),
]


class NutritionDatabase:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self.lookup_path = os.path.join(processed_dir, "protein_lookup.csv")
        self.unmatched_path = os.path.join(processed_dir, "unmatched_protein_items.csv")
        self.lookup: Dict[str, Tuple[float, str, str]] = {}
        self.unmatched: Set[str] = set()
        self._init_database()

    def _init_database(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        for item, prot, src, note in REFERENCE_PROTEIN_DATA:
            self.lookup[item.lower().strip()] = (prot, src, note)
        self.save_lookup_csv()

    def save_lookup_csv(self):
        with open(self.lookup_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ingredient", "protein_per_100g", "source", "notes"])
            for ing, (prot, src, note) in sorted(self.lookup.items()):
                writer.writerow([ing, f"{prot:.2f}", src, note])

    def record_unmatched(self, raw_item: str):
        item_clean = raw_item.lower().strip()
        if item_clean and item_clean not in self.lookup:
            self.unmatched.add(item_clean)
            self._save_unmatched()

    def _save_unmatched(self):
        with open(self.unmatched_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["unmatched_ingredient"])
            for item in sorted(self.unmatched):
                writer.writerow([item])

    def get_protein_per_100g(self, ingredient_name: str) -> Optional[Tuple[float, str]]:
        """
        Returns (protein_per_100g, source) or None if no match.
        Never guesses unknown values.
        """
        key = ingredient_name.lower().strip()
        if key in self.lookup:
            prot, src, _ = self.lookup[key]
            return prot, src
        self.record_unmatched(ingredient_name)
        return None


# Module-level singleton
_NUTRITION_DB: Optional[NutritionDatabase] = None

def get_nutrition_db(processed_dir: str = "data/processed") -> NutritionDatabase:
    global _NUTRITION_DB
    if _NUTRITION_DB is None:
        _NUTRITION_DB = NutritionDatabase(processed_dir=processed_dir)
    return _NUTRITION_DB


if __name__ == "__main__":
    db = NutritionDatabase()
    print(f"Saved {len(db.lookup)} verified protein entries to {db.lookup_path}")
    # Sanity check
    besan_match = db.get_protein_per_100g("besan")
    print(f"Besan check: {besan_match}")
