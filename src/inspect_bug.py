import pandas as pd
import json

df = pd.read_csv("data/raw/maharashtra_drafts.csv")
print("Total rows:", len(df))

# Check Phase 0 context:
# "Count rows where an ingredient was split wrongly (e.g. item='leaves', quantity='8 curry'). Fix the parser in src/build_dataset.py. Do not retrain."
# Let's inspect where 'leaves' or '8 curry' occurs across all columns
broken_count = 0
for i, r in df.iterrows():
    m = str(r.get("main_ingredients", ""))
    p = str(r.get("pantry_items", ""))
    iq = str(r.get("ingredients_with_quantity", ""))
    
    # Check if 'leaves' is an isolated ingredient name in main or pantry
    m_items = [x.strip().lower() for x in m.split(",") if x.strip()]
    p_items = [x.strip().lower() for x in p.split(",") if x.strip()]
    
    is_broken = False
    if "leaves" in m_items or "leaves" in p_items or "curry" in m_items or "curry" in p_items:
        is_broken = True
    elif "8 curry" in iq.lower() or "curry, leaves" in m.lower() or "curry, leaves" in p.lower():
        is_broken = True
    elif any(item == "leaves" or item == "curry" for item in m_items + p_items):
        is_broken = True

    if is_broken:
        broken_count += 1
        if broken_count <= 10:
            print(f"Row {i} ({r['dish_name']}):")
            print(f"  main_ingredients: {m}")
            print(f"  pantry_items: {p}")
            print(f"  iq: {iq}")

print(f"\nTotal broken rows identified: {broken_count}")
