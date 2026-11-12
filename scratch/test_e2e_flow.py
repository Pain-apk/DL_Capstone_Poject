"""
scratch/test_e2e_flow.py — Simulates the full Web UI lifecycle through FastAPI TestClient.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from app.main import app

def test_full_pipeline():
    client = TestClient(app)

    print("=== 1. TEST GET /config ===")
    r = client.get("/config")
    assert r.status_code == 200, f"Config failed: {r.text}"
    cfg = r.json()
    print(f"  Available states: {len(cfg['states'])}, pantry items: {len(cfg['pantry_options'])}")

    print("\n=== 2. TEST POST /detect ===")
    with open("tests/images/tomato_01.jpg", "rb") as f:
        r = client.post("/detect", files={"image": ("tomato.jpg", f, "image/jpeg")})
    assert r.status_code == 200, f"Detect failed: {r.text}"
    det = r.json()
    print("  Detection Output:", det)
    detected_items = [i["canonical"] for i in det.get("ingredients", [])]
    assert len(detected_items) > 0, "Expected at least 1 detected item"

    print("\n=== 3. TEST POST /dishes ===")
    dishes_payload = {
        "ingredients": detected_items + ["onion", "potato"],
        "pantry": ["oil", "salt", "water", "turmeric", "mustard seeds", "cumin seeds"],
        "state": "Maharashtra",
        "protein_focus": "off",
        "max_time": 45,
        "servings": 2,
        "vegetarian_only": True,
        "excluded_allergens": []
    }
    r = client.post("/dishes", json=dishes_payload)
    assert r.status_code == 200, f"Dishes failed: {r.text}"
    dishes_res = r.json()
    top_candidates = dishes_res.get("top", [])
    decision = dishes_res.get("decision", {})
    print(f"  Top Decision: {decision.get('dish_name')} (Match: {decision.get('ingredient_match')*100:.0f}%, State: {decision.get('state')})")
    print(f"  Top 5 candidates: {[c['dish_name'] for c in top_candidates]}")

    print("\n=== 4. TEST POST /recipe/generate ===")
    gen_payload = {
        "dish_id": decision.get("dish_id"),
        "dish": decision,
        "ingredients": detected_items + ["onion", "potato"],
        "pantry": ["oil", "salt", "water", "turmeric", "mustard seeds", "cumin seeds"],
        "state": "Maharashtra",
        "allergens": [],
        "servings": 2,
        "max_time": 45,
        "debug": True
    }
    r = client.post("/recipe/generate", json=gen_payload)
    assert r.status_code == 200, f"Generate failed: {r.text}"
    gen_res = r.json()
    print(f"  Recipe Source: {gen_res.get('source')}")
    print(f"  Recipe Title:  {gen_res.get('recipe', {}).get('dish_name')}")
    print(f"  Ingredients:   {len(gen_res.get('recipe', {}).get('ingredients', []))} items")
    print(f"  Steps:         {len(gen_res.get('recipe', {}).get('steps', []))} steps")

    print("\n=== 5. TEST GET /recipe/{dish_id} ===")
    r = client.get(f"/recipe/{decision.get('dish_id')}")
    assert r.status_code == 200, f"Fetch reference failed: {r.text}"
    ref = r.json()
    print(f"  Stored Reference Dish: {ref.get('dish_name')} ({ref.get('time_minutes')}m, {ref.get('servings')} servings)")

    print("\n" + "=" * 60)
    print("ALL 5 ENDPOINTS PASSED END-TO-END PIPELINE VERIFICATION!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_pipeline()
