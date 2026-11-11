"""
app/detector.py

detect(image_bytes: bytes) -> dict

Returns:
  {
    "ingredients": [{"name": str, "confidence": float}],  # conf >= 0.4, one per name (highest)
    "low_confidence": [{"name": str, "confidence": float}] # 0.25 <= conf < 0.4
  }

Environment:
  MOCK_DETECTOR=1  -> return MOCK_RESULT without loading model
  DETECTOR_MODEL=path  -> path to .pt weights (default: data/cnn/best.pt)
  DETECTOR_CLASSES=comma,separated -> class names in index order (overrides model names)
"""

import os
import sys
import io
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MOCK_RESULT = {
    "ingredients": [
        {"name": "onion", "confidence": 0.91},
        {"name": "tomato", "confidence": 0.87},
        {"name": "potato", "confidence": 0.76},
    ],
    "low_confidence": [
        {"name": "garlic", "confidence": 0.31},
    ],
}

CONF_HIGH = 0.4
CONF_LOW = 0.25

# Module-level singleton — loaded lazily on first detect() call
_model = None  # type: Any
_class_names = None  # type: Optional[List[str]]
_model_loaded = False


def _get_model():
    """Lazy-load the YOLO model once, respecting env vars."""
    global _model, _class_names, _model_loaded

    if _model_loaded:
        return _model, _class_names

    _model_loaded = True  # mark attempted so we don't retry on failure

    model_path = os.environ.get(
        "DETECTOR_MODEL",
        os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt"),
    )

    if not os.path.exists(model_path):
        raise ImportError(
            f"Detector model weights not found at '{model_path}'. "
            "Set DETECTOR_MODEL env var to the correct path, or set "
            "MOCK_DETECTOR=1 to use the mock result instead."
        )

    try:
        from ultralytics import YOLO  # type: ignore
        _model = YOLO(model_path)
    except Exception as exc:
        raise ImportError(
            f"Failed to load detector model from '{model_path}': {exc}"
        ) from exc

    # Class names: env override takes priority over model names
    env_classes = os.environ.get("DETECTOR_CLASSES", "").strip()
    if env_classes:
        _class_names = [c.strip() for c in env_classes.split(",")]
    else:
        # ultralytics exposes names as a dict {int: str}
        try:
            names_dict = _model.names
            _class_names = [names_dict[i] for i in sorted(names_dict.keys())]
        except Exception:
            _class_names = []

    return _model, _class_names


def _strip_exif(image_bytes: bytes) -> bytes:
    """Return image bytes with EXIF stripped (never saves to disk)."""
    try:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB to drop EXIF / metadata
        img = img.convert("RGB")

        # Resize if either dimension > 1280px (maintain aspect ratio)
        max_dim = 1280
        w, h = img.size
        if w > max_dim or h > max_dim:
            ratio = min(max_dim / w, max_dim / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        # Pillow unavailable or corrupt image — return original
        return image_bytes


def detect(image_bytes: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run ingredient detection on raw image bytes.

    Returns a dict with two keys:
      - "ingredients": detections with confidence >= 0.4 (highest per name)
      - "low_confidence": detections with 0.25 <= confidence < 0.4
    """
    if os.environ.get("MOCK_DETECTOR", "").strip() in ("1", "true", "yes"):
        return MOCK_RESULT

    model, class_names = _get_model()

    # Pre-process: strip EXIF, resize
    clean_bytes = _strip_exif(image_bytes)

    import io as _io
    from PIL import Image as _Image  # type: ignore

    pil_img = _Image.open(_io.BytesIO(clean_bytes)).convert("RGB")

    # Run inference
    results = model(pil_img, verbose=False)

    # Gather raw detections: {name -> highest_conf}
    high_map = {}  # type: Dict[str, float]
    low_map = {}   # type: Dict[str, float]

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if class_names and cls_idx < len(class_names):
                name = class_names[cls_idx]
            else:
                name = str(cls_idx)

            if conf >= CONF_HIGH:
                if name not in high_map or conf > high_map[name]:
                    high_map[name] = conf
            elif conf >= CONF_LOW:
                if name not in low_map or conf > low_map[name]:
                    low_map[name] = conf

    # Names in high_map take priority; remove from low_map
    for name in high_map:
        low_map.pop(name, None)

    ingredients = [
        {"name": name, "confidence": round(conf, 4)}
        for name, conf in sorted(high_map.items(), key=lambda x: -x[1])
    ]
    low_confidence = [
        {"name": name, "confidence": round(conf, 4)}
        for name, conf in sorted(low_map.items(), key=lambda x: -x[1])
    ]

    return {"ingredients": ingredients, "low_confidence": low_confidence}
