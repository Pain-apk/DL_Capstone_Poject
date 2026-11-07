"""
apply_reviews.py - Applies human reviewer verdicts and corrections from review/review_sheet.csv
to produce verified dataset records.

Rules:
- Supports 'ok', 'fix', and 'reject' verdicts.
- verified = "yes" ONLY for rows explicitly approved as 'ok' or successfully corrected and approved as 'fix'.
- Unreviewed rows remain verified = "no" and review_flag = "pending_human_review".
- Rejected rows remain verified = "no" and review_flag = "rejected".
- Never overwrites or deletes original files. Saves output to data/processed/reviewed_recipes.csv.
"""

import os
import sys
import csv
import json
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRAFTS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")
REVIEW_SHEET = os.path.join(PROJECT_ROOT, "review", "review_sheet.csv")
OUTPUT_REVIEWED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "reviewed_recipes.csv")

def apply_reviews():
    if not os.path.exists(DRAFTS_CSV):
        print(f"Error: {DRAFTS_CSV} not found.")
        return
    if not os.path.exists(REVIEW_SHEET):
        print(f"Error: {REVIEW_SHEET} not found.")
        return

    os.makedirs(os.path.dirname(OUTPUT_REVIEWED_CSV), exist_ok=True)

    # Read review sheet
    reviews: Dict[str, Dict[str, str]] = {}
    with open(REVIEW_SHEET, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dish = r["dish"].strip()
            reviews[dish] = r

    # Read drafted recipes
    with open(DRAFTS_CSV, "r", encoding="utf-8") as f:
        drafts = list(csv.DictReader(f))

    updated_rows = []
    stats = {"ok": 0, "fix": 0, "reject": 0, "unreviewed": 0}

    for row in drafts:
        dish = row["dish_name"].strip()
        review_entry = reviews.get(dish)

        new_row = dict(row)

        if review_entry:
            verdict = review_entry.get("verdict", "").strip().lower()
            reviewer = review_entry.get("reviewer_name", "").strip()
            corrections = review_entry.get("corrections", "").strip()

            new_row["reviewer_name"] = reviewer
            new_row["review_verdict"] = verdict

            if verdict == "ok":
                new_row["verified"] = "yes"
                new_row["review_flag"] = "ok"
                stats["ok"] += 1
            elif verdict == "fix":
                # Apply corrections if provided (format: key=val or replacement text)
                if corrections:
                    new_row["reviewer_corrections"] = corrections
                    # If reviewer provided specific field corrections (e.g., ingredients=...)
                    if "ingredients=" in corrections:
                        m = re.search(r"ingredients=(.*?)(;|$)", corrections)
                        if m:
                            new_row["ingredients_with_quantity"] = m.group(1).strip()
                    if "steps=" in corrections:
                        m = re.search(r"steps=(.*?)(;|$)", corrections)
                        if m:
                            new_row["steps"] = m.group(1).strip()
                    if "servings=" in corrections:
                        m = re.search(r"servings=(.*?)(;|$)", corrections)
                        if m:
                            new_row["servings"] = m.group(1).strip()

                new_row["verified"] = "yes"
                new_row["review_flag"] = "fix_applied"
                stats["fix"] += 1
            elif verdict == "reject":
                new_row["verified"] = "no"
                new_row["review_flag"] = "rejected"
                stats["reject"] += 1
            else:
                new_row["verified"] = "no"
                new_row["review_flag"] = "pending_human_review"
                stats["unreviewed"] += 1
        else:
            new_row["verified"] = "no"
            new_row["review_flag"] = "unreviewed"
            stats["unreviewed"] += 1

        updated_rows.append(new_row)

    # Save output to data/processed/reviewed_recipes.csv preserving original drafts
    if updated_rows:
        fieldnames = list(updated_rows[0].keys())
        with open(OUTPUT_REVIEWED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    print(f"Applied reviews to {len(updated_rows)} dishes.")
    print(f"Results saved to: {OUTPUT_REVIEWED_CSV}")
    print(f"Verdicts breakdown: {stats}")

if __name__ == "__main__":
    apply_reviews()

