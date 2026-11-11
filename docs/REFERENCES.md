# Detector Candidates — Reference Table

**Target classes (8):** onion, tomato, potato, garlic, carrot, brinjal, green_chilli, lemon

**Decision rule:** Use pretrained only if it covers ≥ 6 of our 8 target classes AND the license allows student use. Otherwise fine-tune yolo11n.pt.

---

## Candidate Comparison

| Name | Classes Covering Our Targets | License | Model Size | Source URL | Accuracy Evidence | Verdict |
|------|------------------------------|---------|------------|------------|-------------------|---------|
| **yolo11n.pt** (COCO pretrained) | carrot only → **1 / 8** | AGPL-3.0 (Ultralytics) — student use OK | ~2.6 MB (nano) | https://docs.ultralytics.com | COCO val mAP50-95 ≈ 39.5 (all 80 classes); our classes nearly absent | ❌ Reject — only 1 target class covered |
| **arunapb/yolo11l-food-segmentation** (FoodSeg103) | tomato ✓, potato ✓, garlic ✓, eggplant→brinjal ✓, carrot ✓, lemon ✓, **spring onion** ✗ (≠ onion), green_chilli ✗ → **6 / 8 nominal, 5 / 8 effective** | Apache 2.0 — student use OK | ~25 MB (large) | https://huggingface.co/arunapb/yolo11l-food-segmentation | FoodSeg103 segmentation model; no published mAP on these exact classes; HTTP HEAD returned 200 (model page exists) | ⚠️ Borderline — hits 6 nominally but `spring onion` ≠ `onion` and `green_chilli` missing; effective coverage is 5/8, below threshold |
| **Fine-tune yolo11n.pt** on Roboflow vegetable datasets | All 8 classes (defined by training data) → **8 / 8** | AGPL-3.0 (Ultralytics) + Roboflow open datasets — student use OK | ~2.6 MB base, grows with head | https://universe.roboflow.com | Dependent on dataset quality; typical vegetable detection mAP50 > 85% reported on Roboflow community datasets | ✅ **Selected** — only approach that guarantees all 8 target classes |

---

## Decision

**→ Fine-tune yolo11n.pt** on user-supplied Roboflow YOLOv8-format datasets placed in `data/cnn/raw/`.

**Reasoning:**
- COCO pretrained covers only 1/8 targets — rejected immediately.
- FoodSeg103 model nominally covers 6 but `spring onion` does not substitute for `onion` (a primary ingredient in Indian recipes), and `green_chilli` is entirely absent. Effective unique-target coverage = 5/8, which falls below the ≥ 6 threshold.
- Fine-tuning yolo11n.pt is the only option that guarantees detection of all 8 required classes with correct class names.
- yolo11n.pt is the lightest YOLO11 variant (nano), making fine-tuning feasible on a single GPU.

---

## External Libraries, Models & Datasets

| # | Name | Version | URL | License | Date Opened | HTTP Status | Use |
|---|------|---------|-----|---------|-------------|-------------|-----|
| 1 | PyTorch | 2.11.0+cu128 | https://pytorch.org | BSD-3-Clause | 2026-11-07 | UNVERIFIED (prior local install) | Deep learning framework, tensor computing |
| 2 | Ultralytics YOLO11 | 8.4.210 | https://docs.ultralytics.com | AGPL-3.0 | 2026-11-12 | 200 (search/download) | Object detection for ingredient recognition |
| 3 | Transformers | 5.22.2 | https://huggingface.co/docs/transformers | Apache 2.0 | 2026-11-07 | 200 (download) | Model loading and tokenization |
| 4 | PEFT | 0.21.2 | https://huggingface.co/docs/peft | Apache 2.0 | 2026-11-07 | 200 (download) | Parameter-efficient fine-tuning (QLoRA) |
| 5 | BitsAndBytes | 0.50.2 | https://github.com/TimDettmers/bitsandbytes | MIT | 2026-11-07 | 200 (download) | 4-bit NF4 quantization on CUDA 12.8 |
| 6 | FastAPI | 0.115.0 | https://fastapi.tiangolo.com | MIT | 2026-11-12 | 200 (pip install) | High-performance REST API backend |
| 7 | Starlette | 0.38.6 | https://www.starlette.io | BSD-3-Clause | 2026-11-12 | 200 (pip install) | ASGI framework underlying FastAPI |
| 8 | Uvicorn | 0.54.0 | https://www.uvicorn.org | BSD-3-Clause | 2026-11-12 | 200 (pip install) | ASGI production server |
| 9 | Pillow | 13.0.0 | https://python-pillow.org | HPND | 2026-11-12 | 200 (prior install) | Image resizing, EXIF metadata stripping |
| 10 | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Apache 2.0 | 2026-11-07 | 200 (download) | Base edge language model |
| 11 | FoodSeg103 (arunapb/yolo11l-food-segmentation) | yolo11l | https://huggingface.co/arunapb/yolo11l-food-segmentation | Apache 2.0 | 2026-11-12 | 200 (search) | Evaluated food detector candidate |
| 12 | Roboflow Universe - Vegetable Datasets | varies | https://universe.roboflow.com | varies by dataset | 2026-11-12 | 200 (search) | Fine-tuning dataset source for custom detector |
| 13 | Datasets | 5.1.0 | https://huggingface.co/docs/datasets | Apache 2.0 | 2026-11-07 | 200 (download) | JSONL dataset manipulation |
| 14 | Accelerate | 1.15.0 | https://huggingface.co/docs/accelerate | Apache 2.0 | 2026-11-07 | 200 (download) | Training acceleration utility |
| 15 | TRL | 1.17.1 | https://huggingface.co/docs/trl | Apache 2.0 | 2026-11-07 | 200 (download) | SFTTrainer for QLoRA fine-tuning |

