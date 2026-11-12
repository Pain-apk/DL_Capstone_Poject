import subprocess
import re
import urllib.request
import json

print("=== 1. CHECK ALL COMMITTED FILES IN HISTORY FOR .env ===")
try:
    out = subprocess.check_output(['git', 'log', '--all', '--full-history', '--', '**/.env*'], text=True)
    print("Git log for .env files:")
    print(out if out.strip() else "NO .env FILES EVER COMMITTED IN GIT HISTORY.")
except Exception as e:
    print("Error running git log for .env:", e)

print("\n=== 2. SEARCH FULL GIT COMMIT DIFFS FOR TOKENS / SECRETS ===")
try:
    p = subprocess.Popen(['git', 'log', '-p', '--all'], stdout=subprocess.PIPE, text=True, errors='ignore')
    patterns = [
        re.compile(r'hf_[a-zA-Z0-9]{30,}'),
        re.compile(r'ghp_[a-zA-Z0-9]{30,}'),
        re.compile(r'sk-[a-zA-Z0-9]{30,}'),
        re.compile(r'(api[_-]?key|secret|token|password)\s*[:=]\s*[\'\"][a-zA-Z0-9_\-]{8,}[\'\"]', re.I)
    ]
    found_secrets = []
    for line in p.stdout:
        for pat in patterns:
            m = pat.search(line)
            if m:
                found_secrets.append(m.group(0))

    print(f"Total potential secret patterns found: {len(found_secrets)}")
    for s in found_secrets[:10]:
        print("  Found:", s)
except Exception as e:
    print("Error checking commit diffs:", e)

print("\n=== 3. CHECK REMOTE REPO VISIBILITY ===")
try:
    req = urllib.request.Request('https://api.github.com/repos/Pain-apk/DL_Capstone_Poject', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode('utf-8')
        info = json.loads(data)
        print("Repo visibility:", "PRIVATE" if info.get("private") else "PUBLIC")
        print("Repo URL:", info.get("html_url"))
        print("Created at:", info.get("created_at"))
        print("Owner:", info.get("owner", {}).get("login"))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.reason)
except Exception as e:
    print("Error checking repo visibility:", e)

