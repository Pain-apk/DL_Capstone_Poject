import urllib.request
import urllib.error
import json

url = "https://api.github.com/repos/Pain-apk/DL_Capstone_Poject"
headers = {"User-Agent": "DLProjectEvaluationBot/1.0"}

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        print("HTTP Status:", resp.getcode())
        data = json.loads(resp.read().decode("utf-8"))
        is_private = data.get("private")
        print("is_private:", is_private)
        print("visibility:", data.get("visibility", "unknown"))
        if is_private is False:
            print("REPOSITORY IS PUBLIC")
        else:
            print("REPOSITORY IS PRIVATE")
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    if e.code == 404:
        print("Repo is either PRIVATE (requires token to see) or does not exist.")
    else:
        print("API error or rate limit.")
except Exception as e:
    print("Error:", e)

