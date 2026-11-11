"""
scratch/verify_30_dishes.py — Finds and tests 2+ verified URLs for each of the 30 Tier 1 vegetarian dishes.
"""

import urllib.request
import re
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CANDIDATES = [
    {
        "dish": "Kanda Poha",
        "urls": [
            "https://hebbarskitchen.com/poha-recipe-2-ways-kanda-poha-aloo-poha/",
            "https://www.indianhealthyrecipes.com/poha-recipe-kanda-batata-poha/",
            "https://en.wikibooks.org/wiki/Cookbook:Poha"
        ]
    },
    {
        "dish": "Batata Poha",
        "urls": [
            "https://hebbarskitchen.com/poha-recipe-2-ways-kanda-poha-aloo-poha/",
            "https://www.indianhealthyrecipes.com/poha-recipe-kanda-batata-poha/"
        ]
    },
    {
        "dish": "Batata Vada",
        "urls": [
            "https://www.indianhealthyrecipes.com/batata-vada-recipe/",
            "https://en.wikipedia.org/wiki/Batata_vada"
        ]
    },
    {
        "dish": "Vada Pav",
        "urls": [
            "https://www.indianhealthyrecipes.com/vada-pav-recipe/",
            "https://en.wikipedia.org/wiki/Vada_pav"
        ]
    },
    {
        "dish": "Kanda Bhaji / Khekda Bhaji",
        "urls": [
            "https://hebbarskitchen.com/onion-pakoda-recipe-pyaz-ke-pakode/",
            "https://www.indianhealthyrecipes.com/onion-pakoda/"
        ]
    },
    {
        "dish": "Batata Bhaji (Sukhi)",
        "urls": [
            "https://www.indianhealthyrecipes.com/potato-fry-recipe-aloo-fry/",
            "https://hebbarskitchen.com/aloo-sabzi-recipe-aloo-ki-sabji/"
        ]
    },
    {
        "dish": "Pithla",
        "urls": [
            "https://hebbarskitchen.com/pithla-recipe-maharashtrian-pitla-bhaji/",
            "https://en.wikipedia.org/wiki/Pithla_bhakhri"
        ]
    },
    {
        "dish": "Tomato Saar",
        "urls": [
            "https://hebbarskitchen.com/tomato-saar-recipe-maharashtrian-style/",
            "https://en.wikipedia.org/wiki/Saar_(soup)"
        ]
    },
    {
        "dish": "Bharli Vangi",
        "urls": [
            "https://www.indianhealthyrecipes.com/gutthi-vankaya-recipe/",
            "https://en.wikipedia.org/wiki/Bharli_Vangi"
        ]
    },
    {
        "dish": "Vangi Bhaji",
        "urls": [
            "https://www.indianhealthyrecipes.com/brinjal-masala-curry-recipe/",
            "https://en.wikipedia.org/wiki/Baingan_bharta"
        ]
    },
    {
        "dish": "Vangyache Bharit",
        "urls": [
            "https://en.wikipedia.org/wiki/Baingan_bharta",
            "https://www.indianhealthyrecipes.com/baingan-bharta-recipe/"
        ]
    },
    {
        "dish": "Lasun Kharda / Thecha",
        "urls": [
            "https://en.wikipedia.org/wiki/Thecha",
            "https://hebbarskitchen.com/kolhapuri-thecha-recipe-mirchi-thecha/"
        ]
    },
    {
        "dish": "Mirchi Thecha",
        "urls": [
            "https://en.wikipedia.org/wiki/Thecha",
            "https://hebbarskitchen.com/kolhapuri-thecha-recipe-mirchi-thecha/"
        ]
    },
    {
        "dish": "Pav Bhaji",
        "urls": [
            "https://www.indianhealthyrecipes.com/pav-bhaji-recipe-mumbai-style-street-food/",
            "https://en.wikipedia.org/wiki/Pav_bhaji"
        ]
    },
    {
        "dish": "Matki Usal",
        "urls": [
            "https://hebbarskitchen.com/spicy-maharashtrian-misal-pav-recipe/",
            "https://en.wikipedia.org/wiki/Usal"
        ]
    },
    {
        "dish": "Misal Pav",
        "urls": [
            "https://www.indianhealthyrecipes.com/misal-pav-recipe/",
            "https://en.wikipedia.org/wiki/Misal_pav"
        ]
    },
    {
        "dish": "Thalipeeth",
        "urls": [
            "https://hebbarskitchen.com/thalipeeth-recipe-maharashtrian-thalipit/",
            "https://en.wikipedia.org/wiki/Thalipeeth"
        ]
    },
    {
        "dish": "Kothimbir Vadi",
        "urls": [
            "https://hebbarskitchen.com/kothimbir-vadi-recipe-kothimbir-wadi/",
            "https://en.wikipedia.org/wiki/Kothimbir_vadi"
        ]
    },
    {
        "dish": "Phodnicha Bhaat",
        "urls": [
            "https://hebbarskitchen.com/cooked-rice-recipes-leftover-recipes/",
            "https://en.wikibooks.org/wiki/Cookbook:Fried_Rice"
        ]
    },
    {
        "dish": "Masale Bhaat",
        "urls": [
            "https://hebbarskitchen.com/maharashtrian-masale-bhat-recipe/",
            "https://en.wikipedia.org/wiki/Khichdi"
        ]
    },
    {
        "dish": "Puri Bhaji",
        "urls": [
            "https://www.indianhealthyrecipes.com/puri-bhaji-recipe-poori-bhaji/",
            "https://en.wikipedia.org/wiki/Puri_bhaji"
        ]
    },
    {
        "dish": "Shev Bhaji",
        "urls": [
            "https://hebbarskitchen.com/sev-tameta-nu-shaak-recipe/",
            "https://en.wikipedia.org/wiki/Sev_khamani"
        ]
    },
    {
        "dish": "Danyachi Amti",
        "urls": [
            "https://en.wikipedia.org/wiki/Amti",
            "https://hebbarskitchen.com/sambar-recipe-south-indian-sambar/"
        ]
    },
    {
        "dish": "Katachi Amti",
        "urls": [
            "https://en.wikipedia.org/wiki/Amti",
            "https://hebbarskitchen.com/chana-saaru-recipe-chana-kat-saaru/"
        ]
    },
    {
        "dish": "Maharashtrian Kadhi",
        "urls": [
            "https://www.indianhealthyrecipes.com/punjabi-kadhi-pakora-recipe/",
            "https://en.wikipedia.org/wiki/Kadhi"
        ]
    },
    {
        "dish": "Alu Vadi / Patra",
        "urls": [
            "https://en.wikipedia.org/wiki/Patra_(food)",
            "https://hebbarskitchen.com/patra-recipe-alu-vadi-recipe/"
        ]
    },
    {
        "dish": "Sabudana Khichdi",
        "urls": [
            "https://www.indianhealthyrecipes.com/sabudana-khichdi-recipe/",
            "https://en.wikipedia.org/wiki/Sabudana_khichdi"
        ]
    },
    {
        "dish": "Sabudana Vada",
        "urls": [
            "https://www.indianhealthyrecipes.com/sabudana-vada-recipe/",
            "https://en.wikipedia.org/wiki/Sabudana_vada"
        ]
    },
    {
        "dish": "Kanda Batata Rassa",
        "urls": [
            "https://www.indianhealthyrecipes.com/aloo-matar-curry-recipe/",
            "https://en.wikipedia.org/wiki/Aloo_tamatar"
        ]
    },
    {
        "dish": "Methichi Bhaji (Sukhi)",
        "urls": [
            "https://www.indianhealthyrecipes.com/methi-dal-recipe-fenugreek-leaves-dal/",
            "https://en.wikipedia.org/wiki/Fenugreek"
        ]
    }
]

def test_urls():
    results = []
    for item in CANDIDATES:
        dish = item["dish"]
        verified = []
        for u in item["urls"]:
            try:
                req = urllib.request.Request(u, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    code = resp.status
                    if code == 200:
                        raw = resp.read()
                        soup = BeautifulSoup(raw, "html.parser")
                        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                        verified.append((u, 200, title[:60]))
            except Exception as e:
                verified.append((u, getattr(e, "code", 0), str(e)[:40]))
        results.append({"dish": dish, "sources": verified})
        ok_count = sum(1 for s in verified if s[1] == 200)
        print(f"[{'PASS' if ok_count >= 2 else 'CHECK'}] {dish:<28} | Verified URLs: {ok_count}/{len(item['urls'])}")
    return results

if __name__ == "__main__":
    test_urls()

