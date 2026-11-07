"""
rigorous_source_verifier.py - Implements strict real-source verification:
1. Real HTTP GET request for every candidate URL.
2. Extracts real <title> and verifies it actually matches the requested dish.
3. Extracts first 12 words of real ingredient list from the HTML.
4. Caches raw text to data/cache/<dish>_source_<idx>.txt.
5. Logs all attempts to logs/source_fetch_log.csv.
6. Re-drafts data/raw/maharashtra_drafts.csv with new columns:
   - servings
   - time_notes
   - source_urls
   - meat_type (chicken, mutton, fish, pork, beef, none)
7. Dishes with fewer than 3 verified opened pages are strictly marked BLOCKED with review_flag=check.
"""

import os
import sys
import re
import csv
import datetime
import urllib.request
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
FETCH_LOG = os.path.join(LOGS_DIR, "source_fetch_log.csv")
DRAFTS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Pilot dish specifications and their candidate URLs to test
PILOT_CANDIDATES = [
    {
        "dish_name": "Kanda Poha",
        "romanized_local_name": "Kande Pohe",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "10m prep + 10m cook (quick breakfast)",
        "candidate_urls": [
            "https://hebbarskitchen.com/poha-recipe-2-ways-kanda-poha-aloo-poha/",
            "https://www.indianhealthyrecipes.com/poha-recipe-kanda-batata-poha/",
            "https://en.wikibooks.org/wiki/Cookbook:Poha"
        ]
    },
    {
        "dish_name": "Misal Pav",
        "romanized_local_name": "Misal Pav",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "excludes 2 days sprouting of moth beans (matki)",
        "candidate_urls": [
            "https://hebbarskitchen.com/spicy-maharashtrian-misal-pav-recipe/",
            "https://www.indianhealthyrecipes.com/misal-pav-recipe/",
            "https://en.wikipedia.org/wiki/Misal_pav"
        ]
    },
    {
        "dish_name": "Thalipeeth",
        "romanized_local_name": "Thalipeeth",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "15m prep + 15m cook (excludes roasting grains for traditional bhajani)",
        "candidate_urls": [
            "https://hebbarskitchen.com/thalipeeth-recipe-maharashtrian-thalipit/",
            "https://www.vegrecipesofindia.com/thalipeeth-recipe/",
            "https://www.tarladalal.com/thalipeeth-2708r"
        ]
    },
    {
        "dish_name": "Pithla",
        "romanized_local_name": "Pithla",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "no",
        "time_notes": "5m prep + 10m cook (quick single pot comfort meal)",
        "candidate_urls": [
            "https://hebbarskitchen.com/pithla-recipe-maharashtrian-pitla-bhaji/",
            "https://www.vegrecipesofindia.com/pitla-recipe-maharashtrian-pitla/",
            "https://www.archanaskitchen.com/maharashtrian-pithla-recipe"
        ]
    },
    {
        "dish_name": "Matki Usal",
        "romanized_local_name": "Matki Usal",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "excludes 1-2 days soaking and sprouting of matki",
        "candidate_urls": [
            "https://hebbarskitchen.com/spicy-maharashtrian-misal-pav-recipe/",
            "https://www.vegrecipesofindia.com/matki-usal-recipe/",
            "https://www.tarladalal.com/matki-chi-usal-3870r"
        ]
    },
    {
        "dish_name": "Bharli Vangi",
        "romanized_local_name": "Bharli Vangi",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "15m stuffing prep + 15m simmering",
        "candidate_urls": [
            "https://hebbarskitchen.com/bharli-vangi-recipe-stuffed-brinjal/",
            "https://www.vegrecipesofindia.com/bharli-vangi-recipe-stuffed-aubergines/",
            "https://www.archanaskitchen.com/bharli-vangi-recipe-maharashtrian-style-stuffed-eggplant"
        ]
    },
    {
        "dish_name": "Kolhapuri Tambda Rassa",
        "romanized_local_name": "Tambda Rassa",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "non-vegetarian",
        "meat_type": "mutton",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "includes 25m mutton pressure cooking for stock extraction",
        "candidate_urls": [
            "https://madhurasrecipe.com/non-veg/kolhapuri-tambda-rassa/",
            "https://www.sanjeevkapoor.com/Recipe/Kolhapuri-Tambda-Rassa-Sirke-Wale-Pyaaz-KhaanaKhazana.html",
            "https://deccantiffin.co.uk/blogs/recipes/kolhapuri-tambda-rassa"
        ]
    },
    {
        "dish_name": "Surmai Fry",
        "romanized_local_name": "Surmai Rava Fry",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "non-vegetarian",
        "meat_type": "fish",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "15m marination + 8m shallow frying",
        "candidate_urls": [
            "https://hebbarskitchen.com/surmai-fish-fry-recipe-seer-fish-fry/",
            "https://www.archanaskitchen.com/malvani-surmai-fry-recipe-seer-fish-fry",
            "https://kunalsfoodjourney.com/crispy-surmai-rava-fry-recipe-malvani-style/"
        ]
    },
    {
        "dish_name": "Jowar Bhakri",
        "romanized_local_name": "Jowarichi Bhakri",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "requires boiling hot water for kneading non-gluten sorghum flour",
        "candidate_urls": [
            "https://hebbarskitchen.com/jowar-roti-recipe-jowar-bhakri-jolada/",
            "https://www.vegrecipesofindia.com/jowar-roti-recipe/",
            "https://www.archanaskitchen.com/jowar-bhakri-recipe-sorghum-flatbread"
        ]
    },
    {
        "dish_name": "Masale Bhaat",
        "romanized_local_name": "Masale Bhaat",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "includes 20m rice soaking + 15m steam cooking",
        "candidate_urls": [
            "https://hebbarskitchen.com/maharashtrian-masale-bhat-recipe/",
            "https://www.vegrecipesofindia.com/masale-bhaat-recipe-maharashtrian-masale-bhaat/",
            "https://www.archanaskitchen.com/traditional-maharashtrian-masale-baath-spiced-vegetable-rice-recipe"
        ]
    }
]

def http_get(url: str) -> Tuple[int, str, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.geturl(), resp.read().decode("utf-8", errors="ignore"), ""
    except urllib.error.HTTPError as e:
        return e.code, url, "", str(e)
    except Exception as e:
        return 0, url, "", str(e)

def extract_recipe_data(html: str):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
    title = re.sub(r"\s+", " ", title).strip()

    # Servings
    servings = "2"
    serv_elem = soup.find(class_=re.compile(r"wprm-recipe-servings|recipe-yield|servings", re.I))
    if serv_elem:
        m = re.search(r"\d+", serv_elem.get_text())
        if m:
            servings = m.group(0)

    # Ingredients extraction
    ingredients = []
    wprm_ings = soup.find_all(class_=re.compile(r"wprm-recipe-ingredient", re.I))
    if wprm_ings:
        for item in wprm_ings:
            t = item.get_text(" ", strip=True)
            if t:
                ingredients.append(t)
    else:
        # Check headings for ingredients
        for heading in soup.find_all(["h2", "h3", "h4", "span"]):
            if "ingredient" in heading.get_text().lower():
                parent = heading.find_parent(["div", "section"]) or heading
                ul = parent.find_next(["ul", "ol"])
                if ul:
                    ingredients = [li.get_text(" ", strip=True) for li in ul.find_all("li") if li.get_text(strip=True)]
                    if ingredients:
                        break

    ing_joined = " | ".join(ingredients) if ingredients else ""
    first_12 = " ".join(re.findall(r"\b[\w/½⅓¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞\.]+\b", ing_joined)[:12]) if ing_joined else "No ingredients identified"

    # Steps extraction
    steps = []
    wprm_steps = soup.find_all(class_=re.compile(r"wprm-recipe-instruction", re.I))
    if wprm_steps:
        for s in wprm_steps:
            txt = s.get_text(" ", strip=True)
            if txt:
                steps.append(txt)
    else:
        for heading in soup.find_all(["h2", "h3", "h4"]):
            if any(k in heading.get_text().lower() for k in ["instruction", "method", "directions", "how to make"]):
                parent = heading.find_next(["ol", "ul"])
                if parent:
                    steps = [li.get_text(" ", strip=True) for li in parent.find_all("li") if li.get_text(strip=True)]
                    break

    steps_joined = " | ".join(steps) if steps else ""
    full_text = soup.get_text()

    return title, servings, ing_joined, first_12, steps_joined, full_text

def run():
    fetch_log_rows = []
    draft_rows = []

    print("Executing strict verification across all 10 pilot dishes...\n")

    for dish in PILOT_CANDIDATES:
        name = dish["dish_name"]
        print(f"--- Checking {name} ---")
        verified_sources = []
        slug = re.sub(r"\W+", "_", name.lower()).strip("_")

        for idx, url in enumerate(dish["candidate_urls"], 1):
            now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status, final_url, html, err = http_get(url)
            print(f"  Attempting: {url} -> Status: {status}")

            title = "FAILED"
            first_12 = "FAILED"
            is_valid_match = False

            if status == 200 and html:
                title, servings, ings, first_12, steps, full_text = extract_recipe_data(html)
                print(f"    Real Title: {title[:60]}")
                print(f"    First 12 words: {first_12}")

                # Check if title genuinely matches dish
                name_tokens = [t.lower() for t in re.split(r"\W+", name) if len(t) > 3]
                title_lower = title.lower()
                matches_dish = (
                    any(t in title_lower for t in name_tokens) or
                    ("poha" in name.lower() and ("poha" in title_lower or "pohe" in title_lower)) or
                    ("misal" in name.lower() and "misal" in title_lower) or
                    ("bhakri" in name.lower() and ("bhakri" in title_lower or "rotti" in title_lower or "roti" in title_lower))
                )

                if matches_dish:
                    is_valid_match = True
                    # Cache page text
                    cache_file = os.path.join(CACHE_DIR, f"{slug}_source_{len(verified_sources)+1}.txt")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(f"DISH: {name}\n")
                        f.write(f"URL: {final_url}\n")
                        f.write(f"TITLE: {title}\n")
                        f.write(f"FETCHED_AT: {now_iso}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(full_text)

                    verified_sources.append({
                        "url": final_url,
                        "title": title,
                        "servings": servings,
                        "ingredients": ings,
                        "first_12": first_12,
                        "steps": steps
                    })
                else:
                    print(f"    [MISMATCH REJECTED] Page title '{title[:40]}' does not match dish '{name}'")
                    first_12 = f"[MISMATCH] Title was: {title[:40]}"

            fetch_log_rows.append({
                "dish": name,
                "url": final_url,
                "http_status": status,
                "real_title": title,
                "fetched_at": now_iso,
                "first_12_ingredient_words": first_12
            })

        count_verified = len(verified_sources)
        print(f"  --> {count_verified}/3 verified opened sources for {name}")

        # Strict check rule
        if count_verified >= 3:
            review_flag = "pending_pilot_approval"
            status_val = "verified"
            # Build draft strictly from verified sources
            raw_ings = verified_sources[0]["ingredients"]
            raw_steps = verified_sources[0]["steps"]
            servings_val = verified_sources[0]["servings"]
            s1 = verified_sources[0]["url"]
            s2 = verified_sources[1]["url"]
            s3 = verified_sources[2]["url"]
            source_urls = f"{s1} | {s2} | {s3}"
        else:
            review_flag = "check"
            status_val = "blocked"
            raw_ings = f"BLOCKED: Only {count_verified}/3 independent sources could be opened via HTTP. Memory drafting forbidden."
            raw_steps = f"BLOCKED: Need 3 verified opened sources. Review flag set to 'check'."
            servings_val = verified_sources[0]["servings"] if count_verified > 0 else "0"
            s1 = verified_sources[0]["url"] if count_verified >= 1 else ""
            s2 = verified_sources[1]["url"] if count_verified >= 2 else ""
            s3 = ""
            source_urls = " | ".join([s["url"] for s in verified_sources]) if verified_sources else "None"

        draft_rows.append({
            "dish_name": name,
            "romanized_local_name": dish["romanized_local_name"],
            "native_name": "",
            "native_name_checked": "no",
            "state": "Maharashtra",
            "tier": "Tier 1",
            "diet": dish["diet"],
            "meat_type": dish["meat_type"],
            "contains_egg": dish["contains_egg"],
            "pan_india": dish["pan_india"],
            "servings": servings_val,
            "time_notes": dish["time_notes"],
            "source_urls": source_urls,
            "ingredients_with_quantity": raw_ings,
            "steps": raw_steps,
            "source_1": s1,
            "source_2": s2,
            "source_3": s3,
            "url_checked": "yes" if count_verified >= 1 else "no",
            "verified": "no",
            "review_flag": review_flag,
            "status": status_val
        })

    # Save logs/source_fetch_log.csv
    with open(FETCH_LOG, "w", newline="", encoding="utf-8") as f:
        fields = ["dish", "url", "http_status", "real_title", "fetched_at", "first_12_ingredient_words"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fetch_log_rows)

    # Save data/raw/maharashtra_drafts.csv
    draft_fields = [
        "dish_name", "romanized_local_name", "native_name", "native_name_checked",
        "state", "tier", "diet", "meat_type", "contains_egg", "pan_india",
        "servings", "time_notes", "source_urls",
        "ingredients_with_quantity", "steps",
        "source_1", "source_2", "source_3",
        "url_checked", "verified", "review_flag", "status"
    ]
    with open(DRAFTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=draft_fields)
        writer.writeheader()
        writer.writerows(draft_rows)

    print(f"\nSaved source fetch log to: {FETCH_LOG}")
    print(f"Saved pilot drafts to: {DRAFTS_CSV}")

if __name__ == "__main__":
    run()
