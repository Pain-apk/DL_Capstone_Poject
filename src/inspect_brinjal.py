import pandas as pd

df = pd.read_csv("data/raw/maharashtra_drafts.csv")
print("Unique main_ingredients:")
print(df["main_ingredients"].value_counts())
