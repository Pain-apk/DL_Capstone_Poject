"""
fetch_pilot_sources.py - Performs real HTTP requests to verify source URLs,
extracts real <title>, logs to logs/source_fetch_log.csv, and caches text in data/cache/.
"""

import os
import sys
import re
import csv
import time
import datetime
import urllib.request
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
FETCH_LOG = os.path.join(LOGS_DIR, "source_fetch_log.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Candidate real URLs to test for each pilot dish
CANDIDATE_URLS = {
    "Kanda Poha": [
        "https://hebbarskitchen.com/kanda-poha-recipe-maharashtrian-poha/",
        "https://www.vegrecipesofindia.com/kanda-poha-recipe-how-to-make-kanda-poha/",
        "https://www.tarladalal.com/kanda-poha-3765r"
    ],
    "Misal Pav": [
        "https://hebbarskitchen.com/kolhapuri-misal-pav-recipe/",
        "https://www.vegrecipesofindia.com/misal-pav-recipe-puneri-misal-pav/",
        "https://www.archanaskitchen.com/maharashtrian-kolhapuri-misal-pav-recipe"
    ],
    "Thalipeeth": [
        "https://www.vegrecipesofindia.com/thalipeeth-recipe/",
        "https://hebbarskitchen.com/instant-thalipeeth-recipe-bhajani/",
        "https://www.tarladalal.com/thalipeeth-2708r"
    ],
    "Pithla": [
        "https://hebbarskitchen.com/pithla-recipe-pitla-bhakri/",
        "https://www.vegrecipesofindia.com/pitla-recipe-maharashtrian-pitla/",
        "https://www.archanaskitchen.com/maharashtrian-pithla-recipe"
    ],
    "Matki Usal": [
        "https://www.vegrecipesofindia.com/matki-usal-recipe/",
        "https://hebbarskitchen.com/matki-chi-usal-recipe-matki-ki-sabji/",
        "https://www.tarladalal.com/matki-chi-usal-3870r"
    ],
    "Bharli Vangi": [
        "https://hebbarskitchen.com/bharli-vangi-recipe-stuffed-brinjal/",
        "https://www.vegrecipesofindia.com/bharli-vangi-recipe-stuffed-aubergines/",
        "https://www.archanaskitchen.com/bharli-vangi-recipe-maharashtrian-style-stuffed-eggplant"
    ],
    "Kolhapuri Tambda Rassa": [
        "https://madhurasrecipe.com/non-veg/kolhapuri-tambda-rassa/",
        "https://www.sanjeevkapoor.com/Recipe/Kolhapuri-Tambda-Rassa-Sirke-Wale-Pyaaz-KhaanaKhazana.html",
        "https://deccantiffin.co.uk/blogs/recipes/kolhapuri-tambda-rassa"
    ],
    "Surmai Fry": [
        "https://hebbarskitchen.com/surmai-fish-fry-recipe-seer-fish-fry/",
        "https://www.archanaskitchen.com/malvani-surmai-fry-recipe-seer-fish-fry",
        "https://kunalsfoodjourney.com/crispy-surmai-rava-fry-recipe-malvani-style/"
    ],
    "Jowar Bhakri": [
        "https://hebbarskitchen.com/jowar-roti-recipe-jowar-bhakri/",
        "https://www.vegrecipesofindia.com/jowar-roti-recipe/",
        "https://www.archanaskitchen.com/jowar-bhakri-recipe-sorghum-flatbread"
    ],
    "Masale Bhaat": [
        "https://www.vegrecipesofindia.com/masale-bhaat-recipe-maharashtrian-masale-bhaat/",
        "https://hebbarskitchen.com/masale-bhaat-recipe-masala-rice/",
        "https://www.archanaskitchen.com/traditional-maharashtrian-masale-baath-spiced-vegetable-rice-recipe"
    ]
}

def fetch_url(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            status = response.status
            real_url = response.geturl()
            html = response.read().decode("utf-8", errors="ignore")
            return status, real_url, html, None
    except urllib.error.HTTPError as e:
        return e.code, url, "", str(e)
    except Exception as e:
        return 0, url, "", str(e)

def extract_recipe_details(html: str):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
    # Clean whitespace in title
    title = re.sub(r"\s+", " ", title).strip()

    # Look for ingredient elements
    ingredient_text = ""
    # Look for wprm-recipe-ingredient or li elements in recipe container
    wprm_ings = soup.find_all(class_=re.compile(r"wprm-recipe-ingredient|recipe-ingredient|ingredient", re.I))
    if wprm_ings:
        items = [i.get_text(strip=True) for i in wprm_ings[:8]]
        ingredient_text = " ".join(items)
    else:
        # Fallback to li elements under an ingredients heading
        for heading in soup.find_all(["h2", "h3", "h4"]):
            if "ingredient" in heading.get_text().lower():
                parent = heading.find_next(["ul", "ol"])
                if parent:
                    items = [li.get_text(strip=True) for li in parent.find_all("li")[:6]]
                    ingredient_text = " ".join(items)
                    break

    # If still empty, grab first text snippet mentioning poha/rice/dal/onion
    if not ingredient_text:
        text_full = soup.get_text()
        m = re.search(r"ingredients[:\s]+(.*?)(instructions|method|directions)", text_full, re.I | re.S)
        if m:
            ingredient_text = m.group(1).strip()
        else:
            ingredient_text = " ".join(soup.stripped_strings)[:200]

    # Clean and extract first 12 words
    clean_words = re.findall(r"\b[\w/½⅓¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞\.]+\b", ingredient_text)
    first_12 = " ".join(clean_words[:12]) if clean_words else "No ingredients detected"

    return title, first_12, soup.get_text()

def main():
    log_rows = []
    print("Beginning real HTTP requests for 10 pilot dishes (3 sources each)...\n")

    for dish, urls in CANDIDATE_URLS.items():
        print(f"--- Fetching sources for {dish} ---")
        dish_slug = re.sub(r"\W+", "_", dish.lower()).strip("_")
        for idx, url in enumerate(urls, 1):
            now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{idx}/3] Requesting {url} ...", end=" ")
            status, final_url, html, err = fetch_url(url)
            print(f"Status: {status}")

            title = "FAILED"
            first_12 = "FAILED"
            if status == 200 and html:
                title, first_12, full_text = extract_recipe_details(html)
                # Cache page text
                cache_file = os.path.join(CACHE_DIR, f"{dish_slug}_source_{idx}.txt")
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(f"URL: {final_url}\n")
                    f.write(f"TITLE: {title}\n")
                    f.write(f"FETCHED_AT: {now_iso}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(full_text)
                print(f"     Title: {title[:60]}")
                print(f"     First 12 words: {first_12}")

            log_rows.append({
                "dish": dish,
                "url": final_url,
                "http_status": status,
                "real_title": title,
                "fetched_at": now_iso,
                "first_12_ingredient_words": first_12
            })
            time.sleep(1) # respectful pacing

    # Write logs/source_fetch_log.csv
    with open(FETCH_LOG, "w", newline="", encoding="utf-8") as f:
        fields = ["dish", "url", "http_status", "real_title", "fetched_at", "first_12_ingredient_words"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nCompleted fetch. Log saved to {FETCH_LOG}")

if __name__ == "__main__":
    main()

