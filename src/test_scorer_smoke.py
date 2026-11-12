import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scorer import DishScorer

scorer = DishScorer()

print("\n--- Test 1: Empty input ---")
res1 = scorer.score_dishes([])
print(res1)
assert res1["dishes"] == []
assert res1["message"] == "No dishes found"

print("\n--- Test 2: Tomato + Cucumber (Kanda Poha must NOT be in top 5) ---")
res2 = scorer.score_dishes(["tomato", "cucumber"])
print("Dishes returned:", len(res2["dishes"]))
for d in res2["dishes"]:
    print(f"  {d['name']} | score={d['score']} | matched={d['matched']}")
    assert "kanda poha" not in d["name"].lower()
    assert "batata poha" not in d["name"].lower()

print("\n--- Test 3: Poha + Onion + Peanuts (Kanda Poha should appear) ---")
res3 = scorer.score_dishes(["poha", "onion", "peanuts"])
print("Dishes returned:", len(res3["dishes"]))
for d in res3["dishes"]:
    print(f"  {d['name']} | score={d['score']} | matched={d['matched']}")
assert any("poha" in d["name"].lower() for d in res3["dishes"])

print("\n--- Test 4: Pantry-only input (must give No dishes found) ---")
res4 = scorer.score_dishes([], pantry=["salt", "oil", "turmeric", "water"])
print(res4)
assert res4["dishes"] == []

print("\n[+] Smoke test passed successfully!")

