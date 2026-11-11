"""
Quick regression test for parse_ingredients_json in src/build_dataset.py.
Run: cd DL_Capstone_Poject; .\venv\Scripts\python.exe scratch\test_parser.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.build_dataset import parse_ingredients_json

CASES = [
    # (input_str,  expected_qty, expected_item)
    ("8 curry leaves",          "8",          "curry leaves"),
    ("1/2 tsp cumin seeds",     "1/2 tsp",    "cumin seeds"),
    ("4 clove garlic",          "4 clove",    "garlic"),
    ("3 tbsp roasted peanuts",  "3 tbsp",     "roasted peanuts"),
    ("1 medium onion",          "1 medium",   "onion"),
]

failures = []
for raw, exp_qty, exp_item in CASES:
    result = parse_ingredients_json(raw)
    assert len(result) == 1, f"Expected 1 item, got {len(result)} for '{raw}'"
    got_qty  = result[0]["quantity"]
    got_item = result[0]["item"]
    if got_qty != exp_qty or got_item != exp_item:
        failures.append(
            f"  '{raw}' → qty={got_qty!r}, item={got_item!r}  "
            f"(expected qty={exp_qty!r}, item={exp_item!r})"
        )

if failures:
    print("FAILED:")
    for f in failures:
        print(f)
else:
    print("PARSER TEST PASSED")
