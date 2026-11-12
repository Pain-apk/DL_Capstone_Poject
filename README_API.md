# Pantry to Plate — FastAPI Backend API Reference

This document provides complete reference documentation for the REST API endpoints provided by `api.main:app`.

---

## 1. Overview & Architecture

The API exposes a 3-layer architecture for edge/laptop execution:
- **Layer 1: Produce Detection (`POST /detect`)** — YOLOv8m produce detector processing images purely in memory with EXIF stripping.
- **Layer 2: Recipe Candidate Scorer (`POST /dishes`)** — Deterministic regional Indian cuisine ranking algorithm with allergen/vegetarian filtering.
- **Layer 3: Recipe Generator & Catalog Fallback (`POST /recipe/generate`)** — Code-enforced decision header generator with 8 structural validators and zero-failure fallback to verified stored recipes.

---

## 2. Endpoints

### 2.1 System & Metadata

#### `GET /health`
Returns the operational health and model status of the backend.
- **Response `200 OK`**:
  ```json
  {
    "status": "ok",
    "detector_loaded": true,
    "scorer_loaded": true,
    "writer_loaded": true,
    "llm_mode": "off",
    "mock_mode": false
  }
  ```

#### `GET /ingredients`
Returns the vocabulary of detectable produce items and common pantry / manual-only ingredients.
- **Response `200 OK`**:
  ```json
  {
    "detectable": ["onion", "tomato", "potato", "carrot", "garlic", "lemon", "brinjal", "cucumber", "..."],
    "manual_only": ["rice", "poha", "rava", "besan", "wheat flour", "sugar", "ghee", "..."]
  }
  ```

---

### 2.2 Produce Detection

#### `POST /detect`
Detects produce and vegetables from an uploaded image in memory.
- **Request**: Multipart form data with field `image` (JPEG or PNG, max 10 MB).
- **Processing**: EXIF metadata stripped in memory; 0 disk writes; YOLOv8m inference at 640px.
- **Response `200 OK`**:
  ```json
  {
    "ingredients": [
      {"name": "onion", "confidence": 0.89}
    ],
    "low_confidence": [
      {"name": "potato", "confidence": 0.14}
    ],
    "image": {
      "width": 640,
      "height": 480
    },
    "inference_ms": 38.4,
    "model": {
      "file": "data/cnn/best.pt",
      "sha256": "...",
      "conf": 0.20,
      "imgsz": 640,
      "tta": true
    },
    "warnings": []
  }
  ```
- **Error Responses**:
  - `400 Bad Request`: Missing `image` form field or corrupt image data.
  - `413 Payload Too Large`: Uploaded file exceeds 10 MB limit.
  - `415 Unsupported Media Type`: File format is not `image/jpeg` or `image/png`.
  - Format: `{"error": {"code": 415, "message": "Unsupported Media Type: only JPEG and PNG are allowed"}}`.

---

### 2.3 Recipe Candidate Scoring

#### `GET /dishes/example`
Returns a ready-to-use sample request payload for `/dishes`.

#### `POST /dishes`
Ranks candidate recipes from the catalog matching available ingredients, regional cuisine preference, and dietary filters.
- **Request `application/json`**:
  ```json
  {
    "ingredients": ["onion", "potato"],
    "pantry": ["oil", "salt", "mustard seeds", "turmeric"],
    "state": "Maharashtra",
    "vegetarian_only": true,
    "allergens": []
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "dishes": [
      {
        "id": "batata_poha",
        "name": "Batata Poha",
        "local_name": "Batata Poha",
        "score": 0.825,
        "ingredient_match": 1.0,
        "matched": ["potato", "onion"],
        "missing": ["poha", "raw peanuts", "green chillies"],
        "region_score": 1.0,
        "protein_g_estimate": 4.5,
        "recipe_status": "unreviewed"
      }
    ]
  }
  ```
- **Empty Result**: If no dishes satisfy the 50% non-staple match threshold and filters:
  ```json
  {
    "dishes": [],
    "message": "No dishes found"
  }
  ```

---

### 2.4 Recipe Generation & Catalog Retrieval

#### `POST /recipe/generate`
Generates an authentic scaled recipe or retrieves the catalog recipe with strict ingredient scaling and allergen safety.
- **Request `application/json`**:
  ```json
  {
    "dish_id": "kanda_poha",
    "ingredients": ["poha", "onion"],
    "pantry": ["oil", "salt"],
    "servings": 4.0,
    "time_limit": 30,
    "allergens": ["dairy"],
    "vegetarian_only": true,
    "llm_mode": "off"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "recipe": {
      "dish_name": "Kanda Poha",
      "local_name": "Kande Pohe",
      "ingredients": [
        {"item": "poha", "quantity": "3 cup"},
        {"item": "onion", "quantity": "2 large"},
        {"item": "raw peanuts", "quantity": "6 tbsp"},
        {"item": "green chillies", "quantity": "4"},
        {"item": "peanut oil", "quantity": "4 tbsp"},
        {"item": "mustard seeds", "quantity": "1 tsp"},
        {"item": "turmeric powder", "quantity": "1/2 tsp"},
        {"item": "fresh curry leaves", "quantity": "16"},
        {"item": "fresh lemon juice", "quantity": "2 tbsp"},
        {"item": "fresh coriander leaves", "quantity": "4 tbsp"},
        {"item": "salt", "quantity": "to taste"}
      ],
      "steps": [
        "Place flattened rice in a colander and rinse under gentle running water...",
        "Heat peanut oil in a kadai over medium flame...",
        "Add mustard seeds to the hot oil and let them crackle...",
        "Add finely diced onions and saute slowly until translucent...",
        "Sprinkle turmeric powder and salt then gently fold in...",
        "Cover with a tight lid and let steam on the lowest flame...",
        "Turn off the flame and finish with freshly squeezed lemon juice..."
      ],
      "time_minutes": 20,
      "missing_ingredients": []
    },
    "source": "stored",
    "recipe_status": "reviewed",
    "validation": {
      "passed": true,
      "failures": []
    },
    "warnings": [
      "LLM_MODE is off: returning stored catalog recipe (unreviewed)."
    ]
  }
  ```

---

## Known limits
- missing_ingredients uses simple name matching and can be wrong (e.g. "peanuts" vs "raw peanuts"); do not use it for a shopping list yet.
- validation.passed describes the LLM attempt only. When source is "stored", the recipe is the stored one regardless of validation.

