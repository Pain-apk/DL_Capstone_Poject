import os
import sys
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.protein import ProteinCalculator

t0 = time.time()
calc = ProteinCalculator()
df = pd.read_csv("data/raw/maharashtra_drafts.csv")

print(f"Loaded ProteinCalculator in {(time.time() - t0)*1000:.2f} ms")

t1 = time.time()
prot_estimates = {}
for i, row in df.iterrows():
    mains = [x.strip() for x in str(row.get("main_ingredients", "")).split("|") if x.strip()]
    iqs = [x.strip() for x in str(row.get("ingredients_with_quantity", "")).split("|") if x.strip()]
    servings = float(row.get("servings", 2.0))
    prot, conf, _ = calc.compute_recipe_protein(iqs, mains, servings=servings)
    prot_estimates[row["dish_name"]] = (prot, conf)

print(f"Computed protein for {len(df)} dishes in {(time.time() - t1)*1000:.2f} ms")
for name in list(prot_estimates.keys())[:5]:
    p, c = prot_estimates[name]
    print(f"  {name}: {p:.2f}g ({c})")
