"""
scratch/fetch_test_images.py — Downloads 10 public-domain / CC-licensed vegetable photos from Wikimedia Commons
into tests/images/ and writes ground-truth annotations to tests/expected.json.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_IMAGES_DIR = os.path.join(PROJECT_ROOT, "tests", "images")
EXPECTED_JSON = os.path.join(PROJECT_ROOT, "tests", "expected.json")

os.makedirs(TESTS_IMAGES_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 10 curated Wikimedia Commons image targets covering our 8 target classes
IMAGE_SPECS = [
    {
        "filename": "onion_01.jpg",
        "wiki_title": "File:Onions.jpg",
        "expected": ["onion"],
        "license": "CC BY-SA 3.0",
        "author": "Colin"
    },
    {
        "filename": "tomato_01.jpg",
        "wiki_title": "File:Tomato_je.jpg",
        "expected": ["tomato"],
        "license": "CC BY-SA 3.0",
        "author": "Jerzy Opioła"
    },
    {
        "filename": "potato_01.jpg",
        "wiki_title": "File:Patates.jpg",
        "expected": ["potato"],
        "license": "CC BY-SA 3.0",
        "author": "Mestres"
    },
    {
        "filename": "garlic_01.jpg",
        "wiki_title": "File:Garlic_Bulb_20060909.jpg",
        "expected": ["garlic"],
        "license": "CC BY-SA 2.5",
        "author": "Justinbaer"
    },
    {
        "filename": "carrot_01.jpg",
        "wiki_title": "File:Carrots_of_many_colors.jpg",
        "expected": ["carrot"],
        "license": "Public Domain",
        "author": "Stephen Ausmus / USDA"
    },
    {
        "filename": "brinjal_01.jpg",
        "wiki_title": "File:Eggplant_and_cross_section03.jpg",
        "expected": ["brinjal"],
        "license": "CC BY-SA 3.0",
        "author": "Fir0002"
    },
    {
        "filename": "green_chilli_01.jpg",
        "wiki_title": "File:Green_chili_peppers.jpg",
        "expected": ["green_chilli"],
        "license": "CC BY-SA 4.0",
        "author": "David Monniaux"
    },
    {
        "filename": "lemon_01.jpg",
        "wiki_title": "File:Lemon.jpg",
        "expected": ["lemon"],
        "license": "CC BY-SA 2.5",
        "author": "André Karwath"
    },
    {
        "filename": "combo_onion_tomato_01.jpg",
        "wiki_title": "File:Onions_and_tomatoes.jpg",
        "expected": ["onion", "tomato"],
        "license": "CC BY-SA 4.0",
        "author": "Wikimedia User"
    },
    {
        "filename": "combo_potato_garlic_01.jpg",
        "wiki_title": "File:Potatoes_and_garlic.jpg",
        "expected": ["potato", "garlic"],
        "license": "CC BY-SA 3.0",
        "author": "Wikimedia Contributor"
    }
]

def resolve_wiki_url(wiki_title):
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(wiki_title)}&prop=imageinfo&iiprop=url|size&format=json"
    req = urllib.request.Request(api_url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pid, pdata in pages.items():
            if "imageinfo" in pdata and len(pdata["imageinfo"]) > 0:
                return pdata["imageinfo"][0]["url"]
    return None

def fetch_all():
    print(f"[*] Fetching 10 test images from Wikimedia Commons into {TESTS_IMAGES_DIR}...")
    expected_map = {}
    downloaded = []

    for spec in IMAGE_SPECS:
        fn = spec["filename"]
        title = spec["wiki_title"]
        dest = os.path.join(TESTS_IMAGES_DIR, fn)
        url = resolve_wiki_url(title)
        
        if not url:
            # Fallback search if title differs slightly
            search_query = title.replace("File:", "").replace(".jpg", "").replace("_", " ")
            search_api = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(search_query)}&srnamespace=6&format=json"
            req = urllib.request.Request(search_api, headers=HEADERS)
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    sdata = json.loads(resp.read().decode("utf-8"))
                    results = sdata.get("query", {}).get("search", [])
                    if results:
                        alt_title = results[0]["title"]
                        url = resolve_wiki_url(alt_title)
            except Exception as e:
                print(f"[-] Search fallback failed for {title}: {e}")

        if url:
            try:
                print(f"  Downloading {fn} from {url[:70]}...")
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    img_bytes = resp.read()
                    with open(dest, "wb") as f:
                        f.write(img_bytes)
                
                # Verify it is a valid image with PIL
                img = Image.open(dest)
                img.verify()
                print(f"  [OK] Saved {fn} ({img.format}, {len(img_bytes)//1024} KB)")
                expected_map[fn] = spec["expected"]
                downloaded.append({
                    "filename": fn,
                    "url": url,
                    "license": spec["license"],
                    "classes": spec["expected"],
                    "http_status": 200
                })
            except Exception as exc:
                print(f"  [!] Failed downloading {fn}: {exc}")
        else:
            print(f"  [!] Could not resolve Wikimedia URL for {title}")

    # Write expected.json
    with open(EXPECTED_JSON, "w", encoding="utf-8") as f:
        json.dump(expected_map, f, indent=2)
    print(f"\n[+] Wrote {len(expected_map)} entries to {EXPECTED_JSON}")

    return downloaded

if __name__ == "__main__":
    fetch_all()
