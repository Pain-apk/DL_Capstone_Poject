# Recipe-Writing LLM (QLoRA & Jetson llama.cpp Deployment)

A specialized small language model fine-tuned to generate structured recipe instructions from detected ingredients, Indian regional context (default: Maharashtra), and dietary constraints, designed to run on resource-constrained edge devices (NVIDIA Jetson) using `llama.cpp`.

---

## Hardware Environment
- **Device:** Windows 11 Laptop
- **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, `sm_120` Blackwell Architecture)
- **CUDA Capability:** `(12, 0)`
- **Target Edge Device:** NVIDIA Jetson with `llama.cpp`

---

## Project Structure
```
DL_Capstone_Poject/
├── data/
│   ├── raw/            # Source CSVs & raw files
│   └── processed/      # Filtered, normalized JSONL dataset
├── src/
│   ├── build_dataset.py  # Data cleaning, normalization & chat formatting
│   ├── train.py          # QLoRA fine-tuning script
│   └── ingredient_synonyms.json # Standardized ingredient mappings
├── eval/
│   ├── cases.json        # 30 hand-written test scenarios
│   └── run_eval.py       # Metrics evaluation (JSON validity, allergen violations, etc.)
├── outputs/              # Checkpoints, adapters, and GGUF exports
├── requirements.txt      # Pinned dependency versions
└── README.md             # Documentation and command logs
```

---

## Command Log & Execution History

### Step 0: Hardware Check
- Selected Base Model: `Qwen/Qwen2.5-1.5B-Instruct` (GPU: 8 GB VRAM).
- Detected Architecture: NVIDIA RTX 5050 (`sm_120`).

### Step 1: Project Setup & Environment Configuration
```powershell
# Create project directories
New-Item -ItemType Directory -Force -Path "data/raw", "data/processed", "src", "outputs", "eval"

# Setup virtual environment
python -m venv venv

# Install PyTorch with CUDA 12.8 support for Blackwell (sm_120)
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install core fine-tuning dependencies
.\venv\Scripts\python.exe -m pip install transformers==5.22.2 peft==0.21.2 trl==1.17.1 bitsandbytes==0.50.2 datasets==5.1.0 accelerate==1.15.0

# Validation of CUDA capability and 4-bit quantization on RTX 5050
.\venv\Scripts\python.exe -c "import torch, bitsandbytes as bnb; print('Device capability:', torch.cuda.get_device_capability(0)); linear = bnb.nn.Linear4bit(64, 64, bias=False).to('cuda'); print('Linear4bit works on CUDA!')"
```

### Step 2-P: Protein Engine & Scoring Verification
```powershell
# Run besan sanity test (verifies IFCT 2017 lookup)
.\venv\Scripts\python.exe src\protein.py

# Run comprehensive test suite for score.py (10 unit tests + protein off vs high comparison)
.\venv\Scripts\python.exe src\test_score.py

# Generate Phase 1 dish sources catalog (Tier 1 Maharashtra + Tier 2 Neighbours)
.\venv\Scripts\python.exe src\generate_sources.py

# Relocate test fixtures and generate verified 10-dish pilot drafts
.\venv\Scripts\python.exe src\update_pilot_and_sources.py

# Strict source verification: real HTTP requests, HTML title and ingredient extraction, caching, and draft blocking
.\venv\Scripts\python.exe src\rigorous_source_verifier.py

# Draft recipes for all 395 dishes with automated validation & human review sheet export
.\venv\Scripts\python.exe src\draft_all_recipes.py

# Apply human reviews script (ok, fix, reject)
.\venv\Scripts\python.exe src\apply_reviews.py

# Build dataset in pilot mode (90/5/5 split by dish title, prompt augmentation)
.\venv\Scripts\python.exe src\build_dataset.py --mode pilot

# Run automated smoke test and conditional full pilot training pipeline
.\venv\Scripts\python.exe src\run_pilot_pipeline.py

# Run pilot inference test to verify JSON output adherence
.\venv\Scripts\python.exe src\test_pilot_inference.py
```

---

## Pilot Model Training Summary & Metrics

### Smoke Test (50 Steps)
- **Initial Loss:** 0.5858
- **Final Loss (Step 50):** 0.0652
- **Peak VRAM:** 6,206.6 MiB (6.06 GB) vs 7.2 GB limit (**PASSED**)
- **Zero Errors:** Passed all 3 continuation gates.

### Full 2-Epoch Training Run (444 Steps)
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Fine-tuning Technique:** QLoRA 4-bit (NF4, double quant, bfloat16 computation, LoRA $r=16, \alpha=32$)
- **Dataset Size:** 1,775 train examples, 100 validation examples (grouped by dish title, 0 data leakage)
- **Training Duration:** 3,186.9 seconds (~53 minutes)
- **Final Training Loss:** 0.04746 (batch loss dropped to ~0.002)
- **Final Validation Loss:** 0.005819
- **Peak VRAM Recorded:** 6,206.6 MiB (6.06 GB)
- **Saved Adapter Location:** `outputs/pilot_unverified/adapter` (73.9 MB safetensors)

### Inference Verification
- Verified generation on unseen prompt with candidate dish selection (`Kanda Poha`).
- Output parsed with strict JSON schema compliance (`dish_name`, `local_name`, `ingredients`, `steps`, `time_minutes`, `missing_ingredients`).
- Candidate constraint satisfied: Generated target dish from provided candidate list.

