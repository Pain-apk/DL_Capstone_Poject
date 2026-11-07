"""
verify_and_draft_pilot.py - Automatically searches for real recipe URLs on accessible domains,
fetches the full HTML via real HTTP GET requests, extracts exact <title> and real ingredient text,
caches the text in data/cache/, logs to logs/source_fetch_log.csv, and produces verified drafts.
"""

import os
import sys
import re
import csv
import time
import datetime
import urllib.request
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
FETCH_LOG = os.path.join(LOGS_DIR, "source_fetch_log.csv")
DRAFTS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Candidate query terms for the 10 pilot dishes
PILOT_QUERIES = [
    {
        "dish_name": "Kanda Poha",
        "romanized_local_name": "Kande Pohe",
        "search_term": "kanda poha",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "10m prep + 10m cook (excludes optional peanut roasting)"
    },
    {
        "dish_name": "Misal Pav",
        "romanized_local_name": "Misal Pav",
        "search_term": "misal pav",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "excludes 2 days sprouting of moth beans (matki)"
    },
    {
        "dish_name": "Thalipeeth",
        "romanized_local_name": "Thalipeeth",
        "search_term": "thalipeeth",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "15m prep + 15m cook (excludes making homemade bhajani flour)"
    },
    {
        "dish_name": "Pithla",
        "romanized_local_name": "Pithla",
        "search_term": "pithla",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "no",
        "time_notes": "5m prep + 10m cook (quick single-pot dish)"
    },
    {
        "dish_name": "Matki Usal",
        "romanized_local_name": "Matki Usal",
        "search_term": "matki usal",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "excludes 1-2 days soaking and sprouting of matki"
    },
    {
        "dish_name": "Bharli Vangi",
        "romanized_local_name": "Bharli Vangi",
        "search_term": "bharli vangi",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "15m stuffing prep + 15m simmering"
    },
    {
        "dish_name": "Kolhapuri Tambda Rassa",
        "romanized_local_name": "Tambda Rassa",
        "search_term": "tambda rassa",
        "diet": "non-vegetarian",
        "meat_type": "mutton",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "includes 25m mutton pressure cooking for stock extraction"
    },
    {
        "dish_name": "Surmai Fry",
        "romanized_local_name": "Surmai Rava Fry",
        "search_term": "surmai fry",
        "diet": "non-vegetarian",
        "meat_type": "fish",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "15m marination + 8m shallow frying"
    },
    {
        "dish_name": "Jowar Bhakri",
        "romanized_local_name": "Jowarichi Bhakri",
        "search_term": "jowar roti bhakri",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "requires boiling hot water for kneading non-gluten dough"
    },
    {
        "dish_name": "Masale Bhaat",
        "romanized_local_name": "Masale Bhaat",
        "search_term": "masale bhaat",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_notes": "includes 20m rice soaking + 15m steam cooking"
    }
]

# Trusted food portals with accessible search endpoints
SEARCH_SITES = [
    {
        "name": "Hebbars Kitchen",
        "search_url": "https://hebbarskitchen.com/?s={query}",
        "pattern": r"hebbarskitchen\.com/[a-z0-9\-]+(?:recipe|pohe|poha|misal|thalipeeth|pithla|usal|vangi|rassa|fry|bhakri|bhaat)[a-z0-9\-]*/?"
    },
    {
        "name": "Indian Healthy Recipes",
        "search_url": "https://www.indianhealthyrecipes.com/?s={query}",
        "pattern": r"indianhealthyrecipes\.com/[a-z0-9\-]+(?:recipe|pohe|poha|misal|thalipeeth|pithla|usal|vangi|rassa|fry|bhakri|bhaat)[a-z0-9\-]*/?"
    },
    {
        "name": "Spice Up The Curry",
        "search_url": "https://www.spiceupthecurry.com/?s={query}",
        "pattern": r"spiceupthecurry\.com/[a-z0-9\-]+(?:recipe|pohe|poha|misal|thalipeeth|pithla|usal|vangi|rassa|fry|bhakri|bhaat)[a-z0-9\-]*/?"
    },
    {
        "name": "Cook With Manali",
        "search_url": "https://www.cookwithmanali.com/?s={query}",
        "pattern": r"cookwithmanali\.com/[a-z0-9\-]+(?:recipe|pohe|poha|misal|thalipeeth|pithla|usal|vangi|rassa|fry|bhakri|bhaat)[a-z0-9\-]*/?"
    },
    {
        "name": "Whisk Affair",
        "search_url": "https://www.whiskaffair.com/?s={query}",
        "pattern": r"whiskaffair\.com/[a-z0-9\-]+(?:recipe|pohe|poha|misal|thalipeeth|pithla|usal|vangi|rassa|fry|bhakri|bhaat)[a-z0-9\-]*/?"
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
            status = resp.status
            final_url = resp.geturl()
            html = resp.read().decode("utf-8", errors="ignore")
            return status, final_url, html, ""
    except urllib.error.HTTPError as e:
        return e.code, url, "", str(e)
    except Exception as e:
        return 0, url, "", str(e)

def find_candidate_urls(dish_info: Dict[str, str]) -> List[str]:
    term = dish_info["search_term"]
    found = []
    seen = set()

    for site in SEARCH_SITES:
        encoded = urllib.parse.quote_plus(term)
        s_url = site["search_url"].format(query=encoded)
        status, final_url, html, err = http_get(s_url)
        if status == 200 and html:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0].rstrip("/") + "/"
                if re.search(site["pattern"], href, re.I):
                    if href not in seen and not any(skip in href for skip in ["/tag/", "/category/", "/page/", "/feed/"]):
                        seen.add(href)
                        found.append(href)
                        if len(found) >= 3:
                            return found
        time.sleep(0.5)

    return found

def extract_recipe_content(html: str) -> Tuple[str, str, str, str, str]:
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
        for heading in soup.find_all(["h2", "h3", "h4"]):
            if "ingredient" in heading.get_text().lower():
                parent = heading.find_next(["ul", "ol"])
                if parent:
                    ingredients = [li.get_text(" ", strip=True) for li in parent.find_all("li") if li.get_text(strip=True)]
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

def run_verification():
    log_rows = []
    draft_rows = []

    print("================================================================================")
    print("STARTING REAL LIVE HTTP FETCH & VERIFICATION FOR 10 PILOT DISHES")
    print("================================================================================\n")

    for dish in PILOT_QUERIES:
        name = dish["dish_name"]
        print(f"\n>>> Resolving 3 verified sources for: {name} ...")
        candidate_urls = find_candidate_urls(dish)
        print(f"    Discovered candidates: {candidate_urls}")

        verified_sources = []
        slug = re.sub(r"\W+", "_", name.lower()).strip("_")

        for idx, url in enumerate(candidate_urls, 1):
            now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status, final_url, html, err = http_get(url)
            print(f"    Fetching Source {idx}: {final_url} -> Status: {status}")

            if status == 200 and html:
                title, servings, ings, first_12, steps, full_text = extract_recipe_content(html)
                print(f"      Title: {title[:60]}")
                print(f"      First 12 words: {first_12}")

                # Cache full page text
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
                    "steps": steps,
                    "fetched_at": now_iso
                })

                log_rows.append({
                    "dish": name,
                    "url": final_url,
                    "http_status": status,
                    "real_title": title,
                    "fetched_at": now_iso,
                    "first_12_ingredient_words": first_12
                })

                if len(verified_sources) >= 3:
                    break
            else:
                log_rows.append({
                    "dish": name,
                    "url": final_url,
                    "http_status": status,
                    "real_title": f"HTTP Error {status}: {err}",
                    "fetched_at": now_iso,
                    "first_12_ingredient_words": "None"
                })

        num_opened = len(verified_sources)
        print(f"    [Result for {name}]: {num_opened}/3 verified sources opened.")

        # If fewer than 3 opened, mark it blocked / check
        review_flag = "pending_pilot_approval" if num_opened >= 3 else "check"
        status_flag = "verified" if num_opened >= 3 else "blocked"

        s1 = verified_sources[0]["url"] if num_opened >= 1 else ""
        s2 = verified_sources[1]["url"] if num_opened >= 2 else ""
        s3 = verified_sources[2]["url"] if num_opened >= 3 else ""
        source_urls_str = " | ".join([s["url"] for s in verified_sources])
        servings_val = verified_sources[0]["servings"] if num_opened >= 1 else "2"

        # Construct draft recipe only if we have verified sources
        raw_ings = verified_sources[0]["ingredients"] if num_opened >= 1 else "BLOCKED: insufficient opened sources"
        raw_steps = verified_sources[0]["steps"] if num_opened >= 1 else "BLOCKED: insufficient opened sources"

        draft_rows.append({
            "dish_name": name,
            "romanized_local_name": dish["romanized_local_name"],
            "native_name": "", # populated only if copied directly from verified page
            "native_name_checked": "no",
            "state": "Maharashtra",
            "tier": "Tier 1",
            "diet": dish["diet"],
            "meat_type": dish["meat_type"],
            "contains_egg": dish["contains_egg"],
            "pan_india": dish["pan_india"],
            "servings": servings_val,
            "time_notes": dish["time_notes"],
            "source_urls": source_urls_str,
            "ingredients_with_quantity": raw_ings[:500], # excerpt
            "steps": raw_steps[:500], # excerpt
            "source_1": s1,
            "source_2": s2,
            "source_3": s3,
            "url_checked": "yes" if num_opened >= 1 else "no",
            "verified": "no",
            "review_flag": review_flag,
            "status": status_flag
        })

    # Save logs/source_fetch_log.csv
    with open(FETCH_LOG, "w", newline="", encoding="utf-8") as f:
        fields = ["dish", "url", "http_status", "real_title", "fetched_at", "first_12_ingredient_words"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)

    # Save data/raw/maharashtra_drafts.csv with all requested columns
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

    print(f"\nSaved fetch log to: {FETCH_LOG}")
    print(f"Saved pilot drafts to: {DRAFTS_CSV}")

if __name__ == "__main__":
    run_verification()

