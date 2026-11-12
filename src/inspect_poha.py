import csv

with open("data/raw/maharashtra_drafts.csv", "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        name = row["dish_name"].lower()
        if "poha" in name or "pohe" in name or "kanda" in name:
            print(f"Dish: {row['dish_name']} ({row['state']})")
            print(f"  main_ingredients: {row['main_ingredients']}")
            print(f"  iq: {row['ingredients_with_quantity'][:70]}")
