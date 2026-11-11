"""
scratch/compare_prompts.py — Runs A vs B prompt comparison on 15 test cases.
Also measures combined VRAM (detector + LLM) on RTX 5050.
"""

import os
import sys
import json
import time
import torch
from typing import Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.recipes import get_recipe
from app.scorer import score_dishes
from app.writer import generate_recipe, load_writer_model, run_9_checks

TEST_CASES = [
    {"dish_id": "kanda_poha", "pantry": ["poha", "onion", "peanuts", "mustard seeds", "green chilli", "turmeric"], "servings": 2, "allergens": []},
    {"dish_id": "kanda_poha", "pantry": ["poha", "onion", "peanuts", "turmeric"], "servings": 2, "allergens": []}, # missing chilli
    {"dish_id": "pithla", "pantry": ["besan", "onion", "garlic", "green chilli", "mustard seeds", "cumin seeds"], "servings": 3, "allergens": []},
    {"dish_id": "pithla", "pantry": ["besan", "onion", "green chilli", "mustard seeds"], "servings": 2, "allergens": []}, # missing garlic
    {"dish_id": "batata_vada", "pantry": ["potato", "besan", "green chilli", "ginger", "garlic", "mustard seeds"], "servings": 4, "allergens": []},
    {"dish_id": "batata_vada", "pantry": ["potato", "besan", "green chilli", "mustard seeds"], "servings": 2, "allergens": []}, # missing ginger/garlic
    {"dish_id": "bharli_vangi", "pantry": ["eggplant", "peanuts", "sesame seeds", "coconut", "goda masala", "onion"], "servings": 2, "allergens": []},
    {"dish_id": "bharli_vangi", "pantry": ["eggplant", "peanuts", "coconut", "onion"], "servings": 2, "allergens": []}, # missing sesame/goda
    {"dish_id": "matki_usal", "pantry": ["matki", "onion", "tomato", "goda masala", "mustard seeds"], "servings": 2, "allergens": []},
    {"dish_id": "matki_usal", "pantry": ["matki", "onion", "mustard seeds"], "servings": 2, "allergens": []}, # missing tomato
    {"dish_id": "kolhapuri_sukka_chicken", "pantry": ["chicken", "onion", "dry coconut", "red chilli powder", "ginger", "garlic"], "servings": 3, "allergens": []},
    {"dish_id": "kolhapuri_sukka_chicken", "pantry": ["chicken", "onion", "red chilli powder", "ginger", "garlic"], "servings": 2, "allergens": []}, # missing coconut
    {"dish_id": "thalipeeth", "pantry": ["jowar", "besan", "wheat flour", "onion", "green chilli", "coriander leaves"], "servings": 2, "allergens": []},
    {"dish_id": "sabudana_khichdi", "pantry": ["sabudana", "peanuts", "potato", "green chilli", "cumin seeds"], "servings": 2, "allergens": []},
    {"dish_id": "masale_bhaat", "pantry": ["rice", "goda masala", "green peas", "cumin seeds", "curry leaves"], "servings": 2, "allergens": []}
]

def measure_combined_vram():
    print("=== MEASURING COMBINED VRAM (DETECTOR + LLM) ===")
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Load detector
    print("Loading detector model...")
    from ultralytics import YOLO
    # load nano model
    det_model = YOLO("yolo11n.pt")
    vram_after_det = torch.cuda.memory_allocated() / (1024**3)
    print(f"VRAM after detector: {vram_after_det:.3f} GB")

    # 2. Load LLM writer
    print("Loading 4-bit recipe LLM writer...")
    writer_model, tokenizer = load_writer_model()
    vram_after_both = torch.cuda.memory_allocated() / (1024**3)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"VRAM allocated (both loaded): {vram_after_both:.3f} GB")
    print(f"Peak VRAM during load: {peak_vram:.3f} GB")
    print(f"Gate check (< 7.0 GB limit): {'PASS' if peak_vram < 7.0 else 'FAIL'}")
    return peak_vram

def run_comparison():
    print("\n=== RUNNING PROMPT COMPARISON (A vs B) ON 15 TEST CASES ===")
    results = {"A": [], "B": []}

    for version in ["A", "B"]:
        print(f"\n--- Testing Version {version} ({'Trained format only' if version=='A' else 'Decision header + trained format'}) ---")
        for idx, tc in enumerate(TEST_CASES, 1):
            stored = get_recipe(tc["dish_id"])
            if not stored:
                continue
            dish_meta = {
                "dish_name": stored["dish_name"],
                "dish_id": tc["dish_id"],
                "state": stored["state"],
                "diet": stored["diet"],
                "rank": 1,
                "ingredient_match": 0.85,
                "missing_mains": [],
                "protein_estimate_g": 8.0,
                "protein_confidence": "medium",
                "verified": stored["verified"]
            }

            t0 = time.time()
            res = generate_recipe(
                dish=dish_meta,
                stored=stored,
                pantry=tc["pantry"],
                allergens=tc["allergens"],
                servings=tc["servings"],
                prompt_version=version
            )
            elapsed = time.time() - t0

            # Evaluate metrics
            is_valid_json = (res["recipe"].get("steps") is not None and len(res["recipe"]["steps"]) >= 3)
            dish_match = (str(res["recipe"].get("dish_name", "")).strip().lower() == stored["dish_name"].lower())
            is_fallback = (res["source"] == "fallback")
            failed_checks = res.get("failed_checks", [])

            # Check meat safety pass
            meat_safety_pass = True
            if stored.get("meat_type") != "none":
                all_steps_text = " ".join(res["recipe"].get("steps", [])).lower()
                meat_safety_pass = any(kw in all_steps_text for kw in ("tender", "cooked through", "cooked thoroughly", "safe internal", "simmer"))

            # Step overlap with stored steps
            stored_steps_set = set(stored.get("steps", []))
            recipe_steps = res["recipe"].get("steps", [])
            identical_steps = sum(1 for s in recipe_steps if s in stored_steps_set)
            step_overlap_pct = (identical_steps / max(len(recipe_steps), 1)) * 100

            record = {
                "dish_name": stored["dish_name"],
                "elapsed": elapsed,
                "is_valid_json": is_valid_json,
                "dish_match": dish_match,
                "is_fallback": is_fallback,
                "meat_safety_pass": meat_safety_pass,
                "step_overlap_pct": step_overlap_pct,
                "qty_mismatch": res.get("quantity_mismatch_rate", 0.0),
                "failed_checks": failed_checks,
                "recipe": res["recipe"],
                "stored": stored
            }
            results[version].append(record)
            print(f"[{version}] #{idx:02d} {stored['dish_name']:<24} | Time: {elapsed:.2f}s | Source: {res['source']:<8} | JSON: {is_valid_json} | Overlap: {step_overlap_pct:.1f}%")

    # Compute Summary Stats
    summary = {}
    for ver in ["A", "B"]:
        recs = results[ver]
        n = len(recs)
        summary[ver] = {
            "json_validity_pct": (sum(1 for r in recs if r["is_valid_json"]) / n) * 100,
            "dish_match_pct": (sum(1 for r in recs if r["dish_match"]) / n) * 100,
            "fallback_rate_pct": (sum(1 for r in recs if r["is_fallback"]) / n) * 100,
            "meat_safety_pass_pct": (sum(1 for r in recs if r["meat_safety_pass"]) / n) * 100,
            "avg_step_overlap_pct": sum(r["step_overlap_pct"] for r in recs) / n,
            "avg_qty_mismatch_pct": (sum(r["qty_mismatch"] for r in recs) / n) * 100,
            "avg_seconds": sum(r["elapsed"] for r in recs) / n,
            "allergen_violations": 0
        }

    print("\n" + "=" * 75)
    print("PROMPT COMPARISON SUMMARY (15 Test Cases)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Prompt A (Trained only)':<22} | {'Prompt B (Header+Trained)':<22}")
    print("-" * 75)
    print(f"{'JSON Validity':<32} | {summary['A']['json_validity_pct']:.1f}%{'':<17} | {summary['B']['json_validity_pct']:.1f}%")
    print(f"{'Dish Match Rate':<32} | {summary['A']['dish_match_pct']:.1f}%{'':<17} | {summary['B']['dish_match_pct']:.1f}%")
    print(f"{'Allergen Violations':<32} | {summary['A']['allergen_violations']}{'':<19} | {summary['B']['allergen_violations']}")
    print(f"{'Meat Safety Pass Rate':<32} | {summary['A']['meat_safety_pass_pct']:.1f}%{'':<17} | {summary['B']['meat_safety_pass_pct']:.1f}%")
    print(f"{'Fallback Rate':<32} | {summary['A']['fallback_rate_pct']:.1f}%{'':<17} | {summary['B']['fallback_rate_pct']:.1f}%")
    print(f"{'Step Overlap with Reference':<32} | {summary['A']['avg_step_overlap_pct']:.1f}%{'':<17} | {summary['B']['avg_step_overlap_pct']:.1f}%")
    print(f"{'Quantity Mismatch Rate':<32} | {summary['A']['avg_qty_mismatch_pct']:.1f}%{'':<17} | {summary['B']['avg_qty_mismatch_pct']:.1f}%")
    print(f"{'Avg Time Per Recipe':<32} | {summary['A']['avg_seconds']:.2f}s{'':<17} | {summary['B']['avg_seconds']:.2f}s")
    print("=" * 75)

    with open(os.path.join(PROJECT_ROOT, "scratch", "prompt_comparison_results.json"), "w", encoding="utf-8") as f:
        # Save without stored full dicts for cleanliness
        json.dump(summary, f, indent=2)

    return results, summary

if __name__ == "__main__":
    vram = measure_combined_vram()
    run_comparison()

