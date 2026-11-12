"""
api/main.py — Clean FastAPI backend server.
Provides:
  - GET  /health
  - GET  /ingredients
  - POST /detect
  - POST /dishes
  - GET  /dishes/example

Rules:
  - Pure in-memory image processing with EXIF stripped.
  - Max 10 MB payload, JPEG and PNG only.
  - Structured error responses: {"error": {"code": int, "message": str}}
  - Full logging on exceptions.
"""

import os
import sys
import io
import time
import hashlib
import logging
import traceback
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, Form, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api.main")

# Import Scorer and Writer
from src.scorer import DishScorer, canonical_name
from src.writer import RecipeWriter

# Target 20 Ingredients classification
DETECTABLE_CLASSES = [
    "onion", "tomato", "potato", "garlic", "ginger", "green chilli",
    "carrot", "brinjal", "capsicum", "cucumber", "lemon", "green peas", "cauliflower"
]
MANUAL_ONLY_CLASSES = [
    "cabbage", "okra", "spinach", "coriander leaves", "egg", "paneer", "curd"
]

CONFIG_FILE = os.path.join(PROJECT_ROOT, "detector_eval", "final_config.json")
WEIGHTS_FILE = os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt")

# Initialize app
app = FastAPI(title="Recipe Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
model_sha256 = "unknown"
detector_config = {
    "conf": 0.20,
    "imgsz": 640,
    "augment": True
}
dish_scorer: Optional[DishScorer] = None
recipe_writer: Optional[RecipeWriter] = None

def get_file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

@app.on_event("startup")
def startup_event():
    global model, model_sha256, detector_config, dish_scorer, recipe_writer
    logger.info("Initializing Recipe Assistant API...")

    # Initialize Scorer
    try:
        dish_scorer = DishScorer()
        logger.info("DishScorer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DishScorer: {traceback.format_exc()}")
        dish_scorer = None

    # Initialize RecipeWriter
    try:
        recipe_writer = RecipeWriter()
        logger.info("RecipeWriter initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RecipeWriter: {traceback.format_exc()}")
        recipe_writer = None

    # Load detector config
    if os.path.exists(CONFIG_FILE):
        try:
            import json
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                detector_config["conf"] = cfg.get("conf", 0.20)
                detector_config["imgsz"] = cfg.get("imgsz", 640)
                # TTA (augment) is a config flag, default True per spec
                detector_config["augment"] = cfg.get("tta", cfg.get("augment", True))
        except Exception as e:
            logger.warning(f"Error loading {CONFIG_FILE}, using defaults: {e}")
    # Enforce default True for TTA per specification unless TTA=0 is set in env
    if os.getenv("DETECTOR_TTA") == "0":
        detector_config["augment"] = False
    else:
        detector_config["augment"] = True

    # Check MOCK_DETECTOR
    if os.getenv("MOCK_DETECTOR") == "1":
        logger.warning("MOCK_DETECTOR=1 is set. Model loading bypassed.")
        model = "MOCK"
        model_sha256 = "mock-sha256"
        return

    # Load YOLO detector
    if os.path.exists(WEIGHTS_FILE):
        try:
            import torch
            from ultralytics import YOLO
            model_sha256 = get_file_sha256(WEIGHTS_FILE)
            model = YOLO(WEIGHTS_FILE)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # Warm up
            dummy = Image.new("RGB", (640, 640), (128, 128, 128))
            _ = model.predict(source=dummy, conf=0.20, imgsz=640, device=device, verbose=False)
            logger.info(f"YOLO detector loaded from {WEIGHTS_FILE} on {device}. SHA256: {model_sha256[:12]}...")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {WEIGHTS_FILE}: {traceback.format_exc()}")
            model = None
    else:
        logger.error(f"Detector weights file not found: {WEIGHTS_FILE}")
        model = None

# Custom error response helper
def error_response(code: int, message: str):
    return JSONResponse(
        status_code=code,
        content={"error": {"code": code, "message": message}}
    )

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "detector_loaded": model is not None,
        "scorer_loaded": dish_scorer is not None,
        "writer_loaded": recipe_writer is not None,
        "llm_mode": os.getenv("LLM_MODE", "off"),
        "mock_mode": os.getenv("MOCK_DETECTOR") == "1"
    }

@app.get("/ingredients")
def get_ingredients():
    return {
        "detectable": DETECTABLE_CLASSES,
        "manual_only": MANUAL_ONLY_CLASSES
    }

@app.post("/detect")
async def detect_ingredients(
    request: Request,
    image: Optional[UploadFile] = File(None)
):
    """
    Accepts multipart field 'image', JPEG/PNG only, max 10 MB.
    EXIF stripped, processed in memory only.
    """
    t0 = time.time()
    try:
        # Check if image field was provided
        if image is None or not hasattr(image, "read"):
            # Check if multipart form was sent with different field name or missing
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" not in content_type:
                return error_response(400, "Content-Type must be multipart/form-data")
            return error_response(400, "Missing 'image' field in multipart form-data")

        # Read into memory
        file_bytes = await image.read()
        file_size = len(file_bytes)

        # Check 10 MB size limit
        MAX_SIZE = 10 * 1024 * 1024
        if file_size > MAX_SIZE:
            return error_response(413, "Payload Too Large: image size exceeds 10 MB limit")

        # Validate MIME type and magic bytes
        mime = (image.content_type or "").lower()
        valid_mimes = ["image/jpeg", "image/png", "image/jpg"]
        is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
        is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")

        if not (is_jpeg or is_png) and mime not in valid_mimes:
            return error_response(415, "Unsupported Media Type: only JPEG and PNG are allowed")
        if mime and mime not in valid_mimes and not (is_jpeg or is_png):
            return error_response(415, "Unsupported Media Type: only JPEG and PNG are allowed")
        if not (is_jpeg or is_png) and (mime.startswith("text/") or mime.startswith("application/")):
            return error_response(415, "Unsupported Media Type: only JPEG and PNG are allowed")

        # Open and strip EXIF in memory
        try:
            with Image.open(io.BytesIO(file_bytes)) as pil_img:
                width, height = pil_img.size
                # Creating a clean RGB image strips all EXIF metadata
                clean_img = Image.new("RGB", pil_img.size)
                clean_img.paste(pil_img)
        except Exception as e:
            logger.error(f"Image open error: {traceback.format_exc()}")
            return error_response(400, f"Cannot parse image: file is corrupted or not a valid image")

        # Check mock mode
        if os.getenv("MOCK_DETECTOR") == "1" or model == "MOCK":
            return {
                "ingredients": [
                    {"name": "onion", "confidence": 0.88},
                    {"name": "tomato", "confidence": 0.75}
                ],
                "low_confidence": [
                    {"name": "potato", "confidence": 0.15}
                ],
                "image": {"width": width, "height": height},
                "inference_ms": round((time.time() - t0) * 1000.0, 2),
                "model": {
                    "file": "mock",
                    "sha256": "mock",
                    "conf": detector_config["conf"],
                    "imgsz": detector_config["imgsz"],
                    "tta": detector_config["augment"]
                },
                "warnings": ["MOCK MODE: Model inference bypassed via MOCK_DETECTOR=1"]
            }

        if model is None:
            logger.error("Detector model is not loaded.")
            return error_response(500, "Internal Server Error: Object detection model is not loaded")

        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cutoff = detector_config["conf"]
        imgsz = detector_config["imgsz"]
        tta = detector_config.get("augment", True)

        # Run inference at conf=0.10 to collect low-confidence detections
        t_inf_start = time.time()
        res = model.predict(
            source=clean_img,
            conf=0.10,
            imgsz=imgsz,
            augment=tta,
            device=device,
            verbose=False
        )[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf_ms = (time.time() - t_inf_start) * 1000.0

        # Parse detections and dedup by canonical name (keeping max confidence)
        high_conf_dict = {}
        low_conf_dict = {}

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                raw_name = model.names[cls_id]
                score = round(float(b.conf.item()), 4)
                c_name = canonical_name(raw_name)

                # Only include target ingredients or mapped produce
                if score >= cutoff:
                    if c_name not in high_conf_dict or score > high_conf_dict[c_name]:
                        high_conf_dict[c_name] = score
                elif score >= 0.10:
                    if c_name not in low_conf_dict or score > low_conf_dict[c_name]:
                        low_conf_dict[c_name] = score

        # Remove items from low_conf if already in high_conf
        for k in high_conf_dict:
            if k in low_conf_dict:
                del low_conf_dict[k]

        ingredients = [{"name": k, "confidence": v} for k, v in sorted(high_conf_dict.items(), key=lambda x: x[1], reverse=True)]
        low_confidence = [{"name": k, "confidence": v} for k, v in sorted(low_conf_dict.items(), key=lambda x: x[1], reverse=True)]

        return {
            "ingredients": ingredients,
            "low_confidence": low_confidence,
            "image": {"width": width, "height": height},
            "inference_ms": round(inf_ms, 2),
            "model": {
                "file": "data/cnn/best.pt",
                "sha256": model_sha256,
                "conf": cutoff,
                "imgsz": imgsz,
                "tta": tta
            },
            "warnings": []
        }

    except Exception as e:
        logger.error(f"Exception during /detect: {traceback.format_exc()}")
        return error_response(500, "Internal Server Error during detection")

# Pydantic model for /dishes request
class DishesRequest(BaseModel):
    ingredients: List[str] = Field(default_factory=list, description="List of detected or input ingredient names")
    pantry: Optional[List[str]] = Field(default_factory=list, description="Optional available pantry staples")
    state: Optional[str] = Field(default="Maharashtra", description="Target regional cuisine state")
    vegetarian_only: Optional[bool] = Field(default=False, description="Strictly exclude meat, fish, and eggs")
    allergens: Optional[List[str]] = Field(default_factory=list, description="Allergens to exclude entirely")
    protein_w: Optional[float] = Field(default=None, description="Optional protein boost weight")

@app.post("/dishes")
def get_dishes(req: DishesRequest):
    """
    Score recipe candidates from catalog based on inputs and constraints.
    """
    if dish_scorer is None:
        logger.error("DishScorer is not loaded.")
        return error_response(500, "Internal Server Error: Recipe scorer not initialized")

    try:
        result = dish_scorer.score_dishes(
            ingredients=req.ingredients,
            pantry=req.pantry,
            state=req.state or "Maharashtra",
            vegetarian_only=bool(req.vegetarian_only),
            allergens=req.allergens or [],
            protein_w=req.protein_w
        )
        return result
    except Exception as e:
        logger.error(f"Exception during /dishes: {traceback.format_exc()}")
        return error_response(500, "Internal Server Error during dish scoring")

@app.get("/dishes/example")
def get_dishes_example():
    """
    Returns a ready-to-use sample request body for POST /dishes.
    """
    return {
        "ingredients": ["onion", "potato", "green chilli"],
        "pantry": ["oil", "salt", "turmeric"],
        "state": "Maharashtra",
        "vegetarian_only": True,
        "allergens": []
    }

# Pydantic model for /recipe/generate request
class RecipeGenerateRequest(BaseModel):
    dish_id: str = Field(..., description="Target dish ID or dish name from catalog")
    ingredients: Optional[List[str]] = Field(default_factory=list, description="User available ingredients")
    pantry: Optional[List[str]] = Field(default_factory=list, description="User pantry staples")
    servings: Optional[float] = Field(default=2.0, description="Target number of servings")
    time_limit: Optional[int] = Field(default=None, description="Max cooking time in minutes")
    allergens: Optional[List[str]] = Field(default_factory=list, description="Allergens to exclude entirely")
    vegetarian_only: Optional[bool] = Field(default=False, description="Strictly vegetarian filter")
    llm_mode: Optional[str] = Field(default=None, description="Optional override: off | base | adapter")

@app.post("/recipe/generate")
def generate_recipe(req: RecipeGenerateRequest):
    """
    Generate or retrieve recipe for the specified dish_id.
    Flow:
      LLM (if LLM_MODE is not off) -> validators -> on failure retry once
      -> still failing, model missing, CUDA error, or LLM_MODE=off: return
      stored catalog recipe with source 'stored'.
      Guaranteed: NEVER returns 500 because of the model.
    """
    if dish_scorer is None or recipe_writer is None:
        logger.error("DishScorer or RecipeWriter not initialized.")
        return error_response(500, "Internal Server Error: Backend services not initialized")

    # Match dish in catalog
    matched_dish = None
    req_id = req.dish_id.strip().lower()
    for d in dish_scorer.dishes:
        if d["id"] == req_id or d["name"].lower() == req_id:
            matched_dish = d
            break

    if not matched_dish:
        # Check partial or substring match
        for d in dish_scorer.dishes:
            if req_id in d["id"] or req_id in d["name"].lower() or d["id"] in req_id:
                matched_dish = d
                break

    if not matched_dish:
        return error_response(404, f"Dish '{req.dish_id}' not found in catalog")

    try:
        result = recipe_writer.generate(
            dish=matched_dish,
            user_ingredients=req.ingredients or [],
            pantry=req.pantry or [],
            servings=req.servings or 2.0,
            time_limit=req.time_limit,
            allergens=req.allergens or [],
            vegetarian_only=bool(req.vegetarian_only),
            mode_override=req.llm_mode
        )
        return result
    except Exception as e:
        logger.error(f"Unexpected exception in /recipe/generate: {traceback.format_exc()}")
        stored = recipe_writer.get_stored_recipe(
            dish=matched_dish,
            servings=req.servings or 2.0,
            user_ingredients=req.ingredients or [],
            pantry=req.pantry or []
        )
        demo_ready = str(matched_dish.get("raw_row", {}).get("demo_ready", "")).strip().lower()
        return {
            "recipe": stored,
            "source": "stored",
            "recipe_status": "reviewed" if demo_ready == "yes" else "unreviewed",
            "validation": {"passed": False, "failures": [f"Generation exception: {str(e)}"]},
            "warnings": ["Unexpected generation error; returned verified stored catalog recipe."]
        }

