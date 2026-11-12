import urllib.request
import json

headers = {"User-Agent": "DLProjectEvaluationBot/1.0"}
api_url = "https://api.github.com/repos/henningheyen/Fruits-And-Vegetables-Detection-Dataset"

print("1. Querying GitHub API for repository metadata:")
try:
    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        lic = data.get("license")
        print("License field:", json.dumps(lic, indent=2))
        print("Default branch:", data.get("default_branch"))
except Exception as e:
    print("API query failed:", e)

branch = "main"
for branch_cand in ["main", "master"]:
    for filename in ["LICENSE", "LICENSE.md", "LICENSE.txt", "README.md"]:
        raw_url = f"https://raw.githubusercontent.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset/{branch_cand}/{filename}"
        try:
            req = urllib.request.Request(raw_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8")
                print(f"\n[+] Found {filename} on branch '{branch_cand}'! Content preview:")
                print(text[:400])
                if "LICENSE" in filename:
                    break
        except Exception as e:
            pass

