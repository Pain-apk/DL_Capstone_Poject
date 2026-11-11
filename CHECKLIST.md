# End-to-End Demo Checklist

> Rules: tick only with evidence (command output, file path, test result). Never tick from memory.

---

## Phase 0 — Audit

- [x] **0a** Template rate measured: 2464 total step sentences, 69 unique (97.2% duplicated). Non-veg cook-through sentence `"Pour in 2 cups water cover pot…"` appears 56×. `scratch/phase0_audit.py` output `2026-11-12`.
- [x] **0b** Ingredient split bugs found: 49 entries across 49 dishes (e.g. `8 curry leaves` → qty=`8 curry`, item=`leaves`). Parser fix written to `src/build_dataset.py`. No retrain.
- [x] **0c** `.gitignore` checked: `venv/` ✓, `data/cache/` ✓, `outputs/` ✓, `*.safetensors` ✓, `*.gguf` ✓, `.env` **was missing → added**. 30 `data/cache/*.txt` files and `outputs/.gitkeep` were committed; `data/cache/` files removed from git index with `git rm --cached`.

---

## Phase 1 — Pick the Detector

- [ ] **1a** Detector candidates researched (table with name, classes, license, size, URL, accuracy).
- [ ] **1b** Decision made: pretrained OR fine-tune `yolo11n.pt`.
- [ ] **1c** `ultralytics` installed (`--no-deps`), `torch.cuda.is_available()` verified still `True`.
- [ ] **1d** If fine-tuning: training started with 80/10/10 split, imgsz=640, 40 epochs, patience=10.
- [ ] **1e** `data/cnn/` directory structure created and `data.yaml` written.

---

## Phase 2 — Test Data

- [ ] **2a** `tests/images/` directory created with placeholder.
- [ ] **2b** `tests/expected.json` schema documented (waiting for user photos).
- [ ] **2c** `tests/images/.gitkeep` committed.

---

## Phase 3 — `app/detector.py`

- [ ] **3a** `detect(image_bytes) -> dict` written (cutoff 0.4, low-confidence 0.25–0.4, dedup by name).
- [ ] **3b** `MOCK_DETECTOR=1` returns fixed list.
- [ ] **3c** Image resizing, EXIF strip, no file save.
- [ ] **3d** Precision/recall reported per photo and overall against `tests/expected.json`.
- [ ] **3e** Time per image measured.

---

## Phase 4 — `app/normalize.py` and `app/scorer.py`

- [ ] **4a** `normalize.py` maps detector/typed names via synonym file to canonical names.
- [ ] **4b** `scorer.py` matches against each dish's ingredient list (not title); returns top 5 with score, ingredient_match, missing mains, state, protein estimate, protein_confidence, verified flag.
- [ ] **4c** Staple list defined (salt, oil, water, turmeric, mustard seeds, cumin seeds, hing, sugar).
- [ ] **4d** Tests (10 cases): Maharashtra vs Gujarat different rankings ✓; protein_focus=high changes rank only when ingredient_match ≥ 0.6 ✓; vegetarian-only box removes meat dishes ✓; no meat in input → no non-veg in top 5 ✓.

---

## Phase 5 — `app/recipes.py`

- [ ] **5a** `get_recipe(dish_id)` returns stored recipe dict with `verified` flag.
- [ ] **5b** Pilot mode reads all drafted rows; final mode reads only `verified=yes` rows.
- [ ] **5c** Stored recipe is reference and fallback; never the default LLM output without generation.

---

## Phase 6 — `app/writer.py`

- [ ] **6a** Base model + adapter loaded once at server start (4-bit).
- [ ] **6b** VRAM measured with detector loaded; total < 7 GB confirmed.
- [ ] **6c** `LLM_ENABLED` env flag works (default true).
- [ ] **6d** Prompt header format implemented exactly per spec.
- [ ] **6e** All 9 code checks implemented.
- [ ] **6f** Fallback to stored recipe on any check failure, with `source="fallback"` and failed check list.
- [ ] **6g** A vs B prompt comparison run on 15 test cases; table reported.
- [ ] **6h** Decision: use format B only if not worse than A.

---

## Phase 7 — `app/main.py` (FastAPI)

- [ ] **7a** `GET /health` returns 200.
- [ ] **7b** `GET /config` returns states, pantry options, protein modes.
- [ ] **7c** `POST /detect` accepts multipart image (10 MB, JPEG/PNG only).
- [ ] **7d** `POST /dishes` returns top 5 + decision.
- [ ] **7e** `POST /recipe/generate` returns `{recipe, source, failed_checks, protein_estimate, protein_confidence, verified}`.
- [ ] **7f** `GET /recipe/{dish_id}` returns stored reference.
- [ ] **7g** `DEBUG=1` returns exact header sent to model.
- [ ] **7h** CORS enabled.
- [ ] **7i** Static serve of `app/web/index.html` at `/`.
- [ ] **7j** Server runs: `uvicorn app.main:app --host 127.0.0.1 --port 8000`.

---

## Phase 8 — `app/web/index.html`

- [ ] **8a** Upload + camera capture with preview.
- [ ] **8b** Detected ingredients as removable chips + "add ingredient" box + pantry tick-list.
- [ ] **8c** State dropdown, protein focus, max time, servings, vegetarian-only checkbox.
- [ ] **8d** "Find dishes" shows top 5 with score, match%, missing items, protein badge, verified badge, "why this dish" line.
- [ ] **8e** Top dish auto-selected; clicking another regenerates recipe.
- [ ] **8f** Recipe card labels: "Written by AI…" or "Showing stored recipe (AI version failed checks)".
- [ ] **8g** "Show reference recipe" toggle.
- [ ] **8h** "Show what the model was told" expander (DEBUG=1 only).
- [ ] **8i** Works on phone browser; friendly error messages.
- [ ] **8j** No external CDN; single file.

---

## Phase 9 — Integration Tests and Jetson Notes

- [ ] **9a** `tests/test_integration.py` written and run against `tests/images/`.
- [ ] **9b** Detector precision/recall, end-to-end seconds/photo reported with raw values.
- [ ] **9c** Ranking tests pass.
- [ ] **9d** Allergy test: peanut allergy never returns peanuts across 20 inputs.
- [ ] **9e** `LLM_ENABLED=false` returns full stored recipe.
- [ ] **9f** No image file left on disk after request.
- [ ] **9g** `docs/JETSON_NOTES.md` written with TensorRT detector, llama.cpp/GGUF, Docker, env vars, Python version.

---

## Final Report

- [ ] CHECKLIST.md fully ticked with evidence.
- [ ] `docs/REFERENCES.md` table complete (every library, model, dataset).
- [ ] Detector precision/recall table.
- [ ] A vs B prompt comparison table.
- [ ] End-to-end time per photo measured and reported.
- [ ] Known problems section written.
