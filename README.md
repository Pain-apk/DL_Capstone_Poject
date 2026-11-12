# Pantry to Plate — Indian Regional Recipe Assistant (2-Model Edge Architecture)

A streamlined 2-model pipeline designed for edge deployment (e.g. NVIDIA Jetson) and laptop GPUs (NVIDIA RTX 5050, 8 GB VRAM):
1. **Model 1 (Produce/Ingredient Detector):** Ultralytics YOLOv8m Produce Detector covering 63 food and vegetable classes (onion, tomato, potato, carrot, garlic, green chilli, lemon, brinjal, capsicum, ginger, peas, etc.), processing photos directly in memory (<103 ms/image, EXIF stripped, 0 disk writes).
2. **Deterministic Scorer & Decision Engine:** Ranks candidate regional dishes based on ingredient availability, regional proximity graph, diet restrictions (vegetarian hard filter), allergen exclusion, and cooking time. Generates the top-1 `[DECISION]` header.
3. **Model 2 (Recipe Generator with Prompt Header):** 4-bit NF4 quantized `Qwen/Qwen2.5-1.5B-Instruct` + QLoRA adapter generating structured, authentic recipes in strict JSON schema directly from the code decision header.

---

## Clean Directory Structure
```
DL_Capstone_Poject/
├── app/
│   ├── detector.py          # Model 1: Real-time YOLO ingredient detection (<103ms)
│   ├── normalize.py         # Canonical ingredient name normalization
│   ├── scorer.py            # Deterministic dish scorer & decision engine
│   ├── recipes.py           # Catalog recipe loader & scaling
│   ├── writer.py            # Model 2: Recipe generator with [DECISION] header
│   ├── main.py              # FastAPI backend (5 endpoints + static mount)
│   └── web/
│       └── index.html       # Responsive single-page web application
├── data/
│   ├── cnn/                 # Model 1 weights (best.pt) and class definitions
│   ├── processed/           # Nutritional protein lookup table (IFCT 2017)
│   └── raw/                 # Dish catalog & regional neighbour graphs
├── docs/
│   ├── JETSON_NOTES.md      # Edge deployment guide (TensorRT & llama.cpp)
│   └── REFERENCES.md        # Official library, dataset & model citations
├── outputs/
│   └── pilot_unverified/
│       └── adapter/         # Model 2 QLoRA adapter weights & tokenizer
├── src/
│   ├── infer.py             # CLI inference script for Model 2
│   ├── score.py             # Scoring formula module
│   ├── protein.py           # Verified protein calculation engine
│   ├── nutrition.py         # ICMR IFCT 2017 & USDA reference database
│   ├── train.py             # QLoRA fine-tuning script
│   └── train_yolo.py        # YOLO fine-tuning script
├── tests/
│   ├── images/              # 10 test produce images (CC/Public Domain)
│   ├── expected.json        # Ground-truth annotations
│   ├── benchmark_detector.py# Model 1 benchmark (latency, precision, recall)
│   ├── test_e2e.py          # End-to-end API pipeline simulation
│   └── test_integration.py  # 9 automated unit/integration tests
├── config.json              # Configurable scorer weights
├── requirements.txt         # Pinned production dependencies
└── README.md
```

---

## How to Run

### 1. Working Directory Note
Always execute commands from the `DL_Capstone_Poject` directory:
```powershell
cd C:\Users\ACER\Desktop\Dev\DL_Project\DL_Capstone_Poject
```

### 2. Run Verification Tests
```powershell
# Run the 9 automated unit & integration tests
.\venv\Scripts\python.exe -m unittest tests/test_integration.py

# Run the end-to-end 5-endpoint pipeline test
.\venv\Scripts\python.exe tests/test_e2e.py

# Benchmark Model 1 (Produce Detector)
.\venv\Scripts\python.exe tests/benchmark_detector.py
```

### 3. Launch Web Server
```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open your browser at:
```
http://127.0.0.1:8000
```

---

## Verified Benchmarks & Resource Metrics

| Metric | Measured Value | Requirement / Limit | Status |
| :--- | :--- | :--- | :--- |
| **Model 1 Latency** | 102.94 ms / image (9.7 FPS) | < 2,000 ms | **PASSED** |
| **Model 1 Coverage** | 63 produce classes (including all 8 targets) | $\ge 5$ classes | **PASSED** |
| **Model 2 Generation VRAM** | 1.212 GB (4-bit NF4) | < 7.0 GB | **PASSED** |
| **Total Combined VRAM** | ~1.35 GB | < 7.0 GB | **PASSED** |
| **JSON Schema Adherence** | 100.0% valid JSON | Strict JSON | **PASSED** |
| **Recipe Generation Source** | `source="llm"` (AI generated from prompt header) | Authentic AI output | **PASSED** |
| **Allergen Safety Enforcement**| 0.0% allergen leaks across 20 randomized trials | Zero leaks | **PASSED** |

---

## Layer 1 — Detector Evaluation Benchmarks

Evaluation performed on `omorros/kitchen-fruit-vegetable-detection` (120 images, 585 annotations, 40 simple / 40 medium / 40 hard tiers) and out-of-distribution test image `tests/images/test1.jpg`.

### Commands Executed:
```powershell
# 1. Benchmark dataset intake & verification
.\venv\Scripts\python.exe -c "import os, glob; ..."

# 2. Candidate evaluation scripts
.\venv\Scripts\python.exe -u detector_eval/run_eval.py --candidate best --weights data/cnn/best.pt --license "other"
.\venv\Scripts\python.exe -u detector_eval/run_eval.py --candidate yoloworld --weights detector_eval/test_setup/yolov8s-worldv2.pt --license "AGPL-3.0"
.\venv\Scripts\python.exe -u detector_eval/run_eval.py --candidate yoloe --weights detector_eval/models/yoloe-11s-seg.pt --license "AGPL-3.0"

# 3. Master summary aggregation
.\venv\Scripts\python.exe -c "import pandas as pd; pd.concat([pd.read_csv(f'detector_eval/results/summary_{c}.csv') for c in ['best','yoloworld','yoloe']]).to_csv('detector_eval/results/summary_all_candidates.csv', index=False)"
```

### Phase C — Safety & Archive Commands
```powershell
New-Item -ItemType Directory -Force -Path 'C:\Users\ACER\Desktop\project_archive_big'
Move-Item -Path 'DL_Capstone_Poject\mobileclip_blt.ts' -Destination 'C:\Users\ACER\Desktop\project_archive_big\'
Move-Item -Path 'DL_Capstone_Poject\weights\clip\ViT-B-32.pt' -Destination 'C:\Users\ACER\Desktop\project_archive_big\'

# Archiving legacy components into archive/
git mv app archive/app
git mv src/score.py archive/src/score.py
git mv src/test_score.py archive/src/test_score.py
git mv src/infer.py archive/src/infer.py
git mv src/train.py archive/src/train.py
git mv src/train_yolo.py archive/src/train_yolo.py
git mv src/intake_yolo_dataset.py archive/src/intake_yolo_dataset.py
git mv CHECKLIST.md archive/CHECKLIST.md
git mv tests/test_e2e.py archive/tests/test_e2e.py
git mv tests/test_integration.py archive/tests/test_integration.py
git mv tests/benchmark_detector.py archive/tests/benchmark_detector.py
git mv tests/fixtures/sample_dishes.json archive/tests/fixtures/sample_dishes.json
Move-Item yolo11n.pt archive/yolo11n.pt
```

---

## Layer 2 — FastAPI Detector & Recipe Scorer Endpoints

Clean backend implementation featuring in-memory EXIF-stripped YOLO produce detection, deterministic recipe candidate scoring with allergen/vegetarian filtering, and regional cuisine weighting.

### Commands Executed:
```powershell
# 1. Scorer unit test verification
.\venv\Scripts\python.exe -m unittest -v tests/test_scorer.py

# 2. Server startup
.\venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

# 3. Detector API tests
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8000/ingredients
curl.exe -s -X POST http://127.0.0.1:8000/detect -F 'image=@tests\images\onion_01.jpg;type=image/jpeg'
curl.exe -s -X POST http://127.0.0.1:8000/detect -F 'image=@tests\images\tomato_01.jpg;type=image/jpeg'
curl.exe -s -X POST http://127.0.0.1:8000/detect -F 'image=@tests\images\potato_01.jpg;type=image/jpeg'
curl.exe -s -X POST http://127.0.0.1:8000/detect -F 'image=@tests\images\carrot_01.jpg;type=image/jpeg'
curl.exe -s -X POST http://127.0.0.1:8000/detect -F 'image=@tests\images\lemon_01.jpg;type=image/jpeg'

# 4. Error handling tests (415, 400, 413)
curl.exe -s -i -X POST http://127.0.0.1:8000/detect -F 'image=@README.md;type=text/markdown'
curl.exe -s -i -X POST http://127.0.0.1:8000/detect -F 'not_image=@tests\images\onion_01.jpg;type=image/jpeg'
curl.exe -s -i -X POST http://127.0.0.1:8000/detect -F 'image=@tests\large_test_file.jpg;type=image/jpeg'

# 5. Scorer API tests
curl.exe -s http://127.0.0.1:8000/dishes/example
curl.exe -s -X POST http://127.0.0.1:8000/dishes -H 'Content-Type: application/json' --data-binary @tests/sample_requests/req1.json
curl.exe -s -X POST http://127.0.0.1:8000/dishes -H 'Content-Type: application/json' --data-binary @tests/sample_requests/req2.json
curl.exe -s -X POST http://127.0.0.1:8000/dishes -H 'Content-Type: application/json' --data-binary @tests/sample_requests/req3.json
curl.exe -s -X POST http://127.0.0.1:8000/dishes -H 'Content-Type: application/json' --data-binary @tests/sample_requests/req4.json
curl.exe -s -X POST http://127.0.0.1:8000/dishes -H 'Content-Type: application/json' --data-binary @tests/sample_requests/req5.json
```




