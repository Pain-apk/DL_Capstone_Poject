import urllib.request
import urllib.error
import json
import re
from bs4 import BeautifulSoup

hf_url = "https://huggingface.co/Senu-12/snapstock-fruit-vegetable-detector/raw/main/README.md"
headers = {"User-Agent": "DLProjectEvaluationBot/1.0"}

print("Fetching Hugging Face README.md...")
try:
    req = urllib.request.Request(hf_url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        content = resp.read().decode("utf-8")
        print("HF README Content Preview:")
        print(content[:1000])
        print("\nAll URLs in HF README:")
        urls = re.findall(r'https?://[^\s)\]"]+', content)
        for u in set(urls):
            print("  ", u)
except Exception as e:
    print("Direct raw README failed:", e)
    # Try fetching model page
    try:
        req = urllib.request.Request("https://huggingface.co/Senu-12/snapstock-fruit-vegetable-detector", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")
            print("Model page fetched. Searching github / Heyen links...")
            urls = re.findall(r'https?://github\.com/[^\s"\'<>]+', html)
            for u in set(urls):
                print("  ", u)
    except Exception as e2:
        print("Page fetch failed:", e2)

