import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import io
from PIL import Image
from fastapi.testclient import TestClient

os.environ["MOCK_DETECTOR"] = "1"
from api.main import app

client = TestClient(app)

# Create a small valid jpeg
buf = io.BytesIO()
Image.new("RGB", (100, 100), (255, 0, 0)).save(buf, format="JPEG")
buf.seek(0)

response = client.post("/detect", files={"image": ("sample.jpg", buf.getvalue(), "image/jpeg")})
print("MOCK_DETECTOR Response (HTTP", response.status_code, "):")
print(json.dumps(response.json(), indent=2))
assert response.status_code == 200
assert "MOCK MODE" in response.json()["warnings"][0]
print("\n[+] MOCK_DETECTOR test passed!")
