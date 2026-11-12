import os
import json
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt")

m = YOLO(WEIGHTS_PATH)
names = m.names

target_20 = [
    "onion", "tomato", "potato", "garlic", "ginger", "green chilli", "carrot",
    "brinjal", "capsicum", "cabbage", "okra", "cucumber", "spinach",
    "coriander leaves", "lemon", "green peas", "cauliflower", "egg", "paneer", "curd"
]

print("Total classes in Candidate A:", len(names))
print("\nAll 63 classes:")
for k, v in names.items():
    print(f"  {k}: {v}")

print("\n--- Matching against the 20 target ingredients ---")
covered = {}
missing = []

for target in target_20:
    matches = []
    for k, v in names.items():
        v_low = v.lower()
        if target == "onion" and "onion" in v_low and "spring" not in v_low:
            matches.append(v)
        elif target == "tomato" and "tomato" in v_low:
            matches.append(v)
        elif target == "potato" and "potato" in v_low and "sweet" not in v_low:
            matches.append(v)
        elif target == "garlic" and "garlic" in v_low:
            matches.append(v)
        elif target == "ginger" and "ginger" in v_low:
            matches.append(v)
        elif target == "green chilli" and ("chili" in v_low or "chilli" in v_low):
            matches.append(v)
        elif target == "carrot" and "carrot" in v_low:
            matches.append(v)
        elif target == "brinjal" and ("eggplant" in v_low or "aubergine" in v_low or "brinjal" in v_low):
            matches.append(v)
        elif target == "capsicum" and ("bell pepper" in v_low or "capsicum" in v_low):
            matches.append(v)
        elif target == "cucumber" and ("cucumber" in v_low or "zucchini" in v_low):
            matches.append(v)
        elif target == "lemon" and "lemon" in v_low:
            matches.append(v)
        elif target == "green peas" and "pea" in v_low:
            matches.append(v)
        elif target == "cauliflower" and "cauliflower" in v_low:
            matches.append(v)
    if matches:
        covered[target] = matches
    else:
        missing.append(target)

print("\nCovered Target Ingredients (13):")
for t, m_list in covered.items():
    print(f"  - {t:<18} -> {m_list}")

print("\nMissing Target Ingredients (7):")
for t in missing:
    print(f"  - {t}")

