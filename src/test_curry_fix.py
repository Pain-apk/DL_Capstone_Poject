import csv
import re

CSV_PATH = "data/raw/maharashtra_drafts.csv"

# In Phase 0, the check was:
# "Count rows where an ingredient was split wrongly (e.g. item='leaves', quantity='8 curry'). Fix the parser in src/build_dataset.py. Do not retrain."
# When ingredients were parsed with the buggy regex:
# regex: r"^(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)\s*([a-zA-Z]+)?\s*(.*)$"
# On "8 curry leaves":
# group 1 (number): "8"
# group 2 (unit): "curry"
# group 3 (name): "leaves"
# item = "leaves", quantity = "8 curry"

def check_broken_parser(iq_str):
    """Returns True if buggy regex would parse an ingredient as item='leaves', qty='... curry'"""
    lines = [x.strip() for x in str(iq_str).split("|") if x.strip()]
    for line in lines:
        m = re.match(r"^(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)\s*([a-zA-Z]+)?\s*(.*)$", line)
        if m:
            qty_num, unit, name = m.groups()
            if unit and unit.lower() == "curry" and name and name.lower() == "leaves":
                return True
            if name and name.lower() == "leaves":
                return True
    return False

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

before_count = sum(1 for r in rows if check_broken_parser(r.get("ingredients_with_quantity", "")))
print(f"Broken rows with old parser: {before_count}")

# Robust parser function:
UNIT_TOKENS = {
    "tsp", "tbsp", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
    "cup", "cups", "g", "kg", "ml", "l", "oz", "lb", "lbs",
    "medium", "large", "small", "pinch", "bunch", "clove", "cloves",
    "piece", "pieces", "slice", "slices",
}

def parse_ingredient_line_fixed(line):
    line = line.strip()
    if not line:
        return ("", "")
    if "to taste" in line.lower():
        name = re.sub(r"\bto taste\b", "", line, flags=re.IGNORECASE).strip()
        return ("to taste", name if name else "salt")
    tokens = line.split()
    num_re = re.compile(r"^\d+([./]\d+)?$")
    if not num_re.match(tokens[0]):
        return ("1 portion", line)
    qty_tokens = [tokens[0]]
    rest = tokens[1:]
    for tok in rest:
        if tok.lower() in UNIT_TOKENS:
            qty_tokens.append(tok)
        else:
            break
    item_tokens = tokens[len(qty_tokens):]
    qty_str = " ".join(qty_tokens)
    item_str = " ".join(item_tokens).strip()
    if not item_str:
        if len(qty_tokens) > 1:
            item_str = qty_tokens[-1]
            qty_str = " ".join(qty_tokens[:-1])
        else:
            item_str = "ingredient"
    return (qty_str, item_str)

def check_broken_fixed(iq_str):
    lines = [x.strip() for x in str(iq_str).split("|") if x.strip()]
    for line in lines:
        qty, item = parse_ingredient_line_fixed(line)
        if item.lower() == "leaves" or qty.lower().endswith("curry"):
            return True
    return False

after_count = sum(1 for r in rows if check_broken_fixed(r.get("ingredients_with_quantity", "")))
print(f"Broken rows with fixed parser: {after_count}")
