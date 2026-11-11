"""
app/main.py — FastAPI backend server for the End-to-End Indian Recipe Assistant.
Python 3.8-compatible. Serves REST API and web UI.
"""

import os
import sys
import io
import time
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.detector import detect
from app.normalize import normalize
from app.scorer import score_dishes
from app.recipes import get_recipe
from app.writer import generate_recipe, is_llm_enabled

app = FastAPI(title="Pantry to Plate — Indian Recipe Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_TOKEN = os.environ.get("API_TOKEN", "").strip()
DEBUG_MODE = os.environ.get("DEBUG", "0").strip() in ("1", "true", "yes")

STATES_LIST = [
    "Maharashtra", "Gujarat", "Goa", "Karnataka",
    "Madhya Pradesh", "Chhattisgarh", "Telangana"
]

PANTRY_OPTIONS = [
    "oil", "salt", "water", "turmeric", "mustard seeds", "cumin seeds",
    "peanuts", "lemon", "coriander leaves", "hing", "sugar",
    "red chilli powder", "ginger", "garlic", "curry leaves"
]


def verify_token(x_api_token: Optional[str] = Header(None)) -> None:
    if API_TOKEN:
        if not x_api_token or x_api_token != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid or missing X-API-Token")


# Pydantic Request Models
class DishesRequest(BaseModel):
    ingredients: List[str] = Field(default_factory=list)
    pantry: List[str] = Field(default_factory=list)
    state: str = "Maharashtra"
    protein_focus: str = "off"
    max_time: int = 0
    servings: int = 2
    vegetarian_only: bool = False
    excluded_allergens: List[str] = Field(default_factory=list)


class RecipeGenerateRequest(BaseModel):
    dish_id: str
    dish: Optional[Dict[str, Any]] = None
    ingredients: List[str] = Field(default_factory=list)
    pantry: List[str] = Field(default_factory=list)
    state: str = "Maharashtra"
    allergens: List[str] = Field(default_factory=list)
    servings: int = 2
    max_time: int = 0
    debug: bool = False


@app.get("/health")
def health_check():
    mock_detector = os.environ.get("MOCK_DETECTOR", "").strip() in ("1", "true", "yes")
    return {
        "status": "ok",
        "llm_enabled": is_llm_enabled(),
        "mock_detector": mock_detector,
        "debug": DEBUG_MODE
    }


@app.get("/config")
def get_config():
    return {
        "states": STATES_LIST,
        "pantry_options": PANTRY_OPTIONS,
        "protein_modes": ["off", "on", "high"]
    }


@app.post("/detect")
async def detect_ingredients(
    image: UploadFile = File(...),
    x_api_token: Optional[str] = Header(None)
):
    verify_token(x_api_token)

    # Validate mime type
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Upload JPEG or PNG.")

    # Validate file size (10 MB limit)
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image size exceeds 10 MB limit.")

    try:
        raw_result = detect(contents)
        # Normalize the detected ingredient names to canonical form
        for item in raw_result.get("ingredients", []):
            item["canonical"] = normalize([item["name"]])[0]
        for item in raw_result.get("low_confidence", []):
            item["canonical"] = normalize([item["name"]])[0]
        return raw_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/dishes")
def get_top_dishes(
    req: DishesRequest,
    x_api_token: Optional[str] = Header(None)
):
    verify_token(x_api_token)

    # Normalize all ingredient and pantry inputs
    raw_combined = req.ingredients + req.pantry
    canonical_pantry = list(dict.fromkeys(normalize(raw_combined)))

    result = score_dishes(
        pantry=canonical_pantry,
        state=req.state,
        protein_focus=req.protein_focus,
        max_time=req.max_time,
        vegetarian_only=req.vegetarian_only,
        excluded_allergens=req.excluded_allergens,
        top_n=5
    )

    # Add explanatory "why this dish" line for each candidate
    for cand in result.get("top", []):
        matched_pct = int(round(cand.get("ingredient_match", 0.0) * 100))
        cand["why"] = (
            f"{matched_pct}% main ingredients matched with your pantry. "
            f"Regional cuisine: {cand.get('state')}."
        )

    return result


@app.post("/recipe/generate")
def generate_recipe_endpoint(
    req: RecipeGenerateRequest,
    x_api_token: Optional[str] = Header(None)
):
    verify_token(x_api_token)

    # Fetch stored recipe as ground-truth facts
    stored = get_recipe(req.dish_id, mode="pilot")
    if not stored:
        raise HTTPException(status_code=404, detail=f"Dish '{req.dish_id}' not found in catalog.")

    dish_meta = req.dish or {
        "dish_name": stored.get("dish_name", req.dish_id),
        "dish_id": req.dish_id,
        "state": stored.get("state", req.state),
        "diet": stored.get("diet", "vegetarian"),
        "rank": 1,
        "ingredient_match": 1.0,
        "missing_mains": [],
        "protein_estimate_g": 0.0,
        "protein_confidence": "medium",
        "verified": stored.get("verified", False)
    }

    # Normalize pantry items
    all_pantry = list(dict.fromkeys(normalize(req.ingredients + req.pantry)))

    enable_debug = req.debug or DEBUG_MODE

    recipe_result = generate_recipe(
        dish=dish_meta,
        stored=stored,
        pantry=all_pantry,
        allergens=req.allergens,
        servings=req.servings,
        max_time=req.max_time,
        prompt_version="B",
        debug=enable_debug
    )

    return recipe_result


@app.get("/recipe/{dish_id}")
def get_stored_recipe_endpoint(
    dish_id: str,
    x_api_token: Optional[str] = Header(None)
):
    verify_token(x_api_token)
    stored = get_recipe(dish_id, mode="pilot")
    if not stored:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found.")
    return stored


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(PROJECT_ROOT, "app", "web", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Pantry to Plate UI loading... Please build app/web/index.html</h1>"

