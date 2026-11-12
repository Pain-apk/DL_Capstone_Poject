import os
import sys
import json
import urllib.request
import urllib.error
import re
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 10 recorded files and their Wikimedia Commons file pages
# The URLs recorded in REFERENCES.md were upload.wikimedia.org URLs.
# The user asks: "for each of the 10 recorded Wikimedia URLs in tests/images, do a real HTTP GET and report the status code and the author and licence text on the file page."
FILES_AND_URLS = [
    {
        "file": "onion_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Onions.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Onions.jpg"
    },
    {
        "file": "tomato_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Tomato_je.jpg"
    },
    {
        "file": "potato_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Patates.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Patates.jpg"
    },
    {
        "file": "garlic_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/2/22/Garlic.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Garlic.jpg"
    },
    {
        "file": "carrot_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/3/32/Carrots_of_many_colors.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Carrots_of_many_colors.jpg"
    },
    {
        "file": "brinjal_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/8/89/Egg_plant.JPG",
        "page_url": "https://commons.wikimedia.org/wiki/File:Egg_plant.JPG"
    },
    {
        "file": "green_chilli_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/3/39/Green_chili_peppers.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Green_chili_peppers.jpg"
    },
    {
        "file": "lemon_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Lemon.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Lemon.jpg"
    },
    {
        "file": "combo_veggies_01.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/3/31/Veggies.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Veggies.jpg"
    },
    {
        "file": "combo_produce_02.jpg",
        "upload_url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Fresh_Vegetable_Produce_%28Unsplash%29.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Fresh_Vegetable_Produce_(Unsplash).jpg"
    }
]

headers = {
    "User-Agent": "DLProjectEvaluationBot/1.0 (academic research test; mailto:student@university.edu)"
}

def check_provenance():
    results = []
    unverified_files = []

    for item in FILES_AND_URLS:
        fname = item["file"]
        p_url = item["page_url"]
        u_url = item["upload_url"]
        print(f"Checking {fname}...")
        
        status_code = None
        author_text = "UNKNOWN"
        license_text = "UNKNOWN"
        verified = False
        
        try:
            req = urllib.request.Request(p_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                status_code = resp.getcode()
                html = resp.read().decode("utf-8", errors="replace")
                
            soup = BeautifulSoup(html, "html.parser")
            
            author_cell = soup.find(id="fileinfotpl_aut")
            if author_cell:
                sibling = author_cell.find_next_sibling("td")
                if sibling:
                    author_text = sibling.get_text(separator=" ", strip=True)
            
            if author_text == "UNKNOWN":
                for td in soup.find_all("td", class_="fileinfo-paramfield"):
                    if "Author" in td.text or "author" in td.text.lower():
                        sib = td.find_next_sibling("td")
                        if sib:
                            author_text = sib.get_text(separator=" ", strip=True)
                            break

            # Fallback to MediaWiki API if author is still unknown
            if author_text == "UNKNOWN":
                title = p_url.split("/wiki/")[-1]
                api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={title}&prop=imageinfo&iiprop=extmetadata|user&format=json"
                req_api = urllib.request.Request(api_url, headers=headers)
                with urllib.request.urlopen(req_api, timeout=10) as r_api:
                    api_data = json.loads(r_api.read().decode("utf-8"))
                    for pg in api_data.get("query", {}).get("pages", {}).values():
                        ii = pg.get("imageinfo", [{}])[0]
                        ext = ii.get("extmetadata", {})
                        art = ext.get("Artist", {}).get("value")
                        if art:
                            # Strip any html tags
                            author_text = re.sub(r'<[^>]+>', '', art).strip()
                        elif ii.get("user"):
                            author_text = ii.get("user")

            # License lookup: usually in licensetpl or license tables
            lic_span = soup.find(class_="licensetpl_short")
            if lic_span:
                license_text = lic_span.get_text(separator=" ", strip=True)
            else:
                lic_cell = soup.find(id="fileinfotpl_perm")
                if lic_cell:
                    sib = lic_cell.find_next_sibling("td")
                    if sib:
                        license_text = sib.get_text(separator=" ", strip=True)
            
            if license_text == "UNKNOWN":
                # Check for standard license divs
                lic_div = soup.find("div", class_="licensetpl")
                if lic_div:
                    license_text = lic_div.get_text(separator=" ", strip=True)

            if status_code == 200:
                verified = True

        except urllib.error.HTTPError as e:
            status_code = e.code
            verified = False
        except Exception as e:
            status_code = f"ERROR: {str(e)}"
            verified = False

        res_entry = {
            "file": fname,
            "page_url": p_url,
            "upload_url": u_url,
            "http_status": status_code,
            "author": author_text[:80] if len(author_text) > 80 else author_text,
            "license": license_text[:80] if len(license_text) > 80 else license_text,
            "verified": verified
        }
        results.append(res_entry)
        if not verified:
            unverified_files.append(fname)
        print(f"  Status: {status_code} | Author: {res_entry['author']} | License: {res_entry['license']} | Verified: {verified}")

    return results, unverified_files

if __name__ == "__main__":
    results, unverified = check_provenance()
    out_path = os.path.join(PROJECT_ROOT, "detector_eval", "wikimedia_provenance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved provenance results to {out_path}")
    print(f"Total unverified files: {len(unverified)}")
