Before drafting any recipes:

1. Source check. For 10 random dishes from maharashtra_sources.csv, show me each source URL, the date you opened it, and one line of what the page said. Tell me honestly if any URL was written from memory without being opened. Remove those, and add a column url_checked (yes/no). Only count sources you actually opened toward the "3 sources" rule.

2. Where did the demo dishes (Kanda Poha, Pithla, Batata Vada, Khaman Dhokla, Paneer Bhurji) and their quantities and protein numbers come from? Show the file path. If they are test data, move them to tests/fixtures/, label every row source="test_fixture", and make build_dataset.py refuse to read that folder. They must never reach data/raw or training.

3. Romanized local_name is required for every dish. Native-script names are optional: only copy them from a source page, never write them from memory, and mark them in a separate column native_name_checked (yes/no).

4. Add columns meat_type (chicken/mutton/fish/pork/beef/none) and contains_egg (yes/no) to the dish files. Fill them only from the sources.

5. Pilot: draft 10 Tier 1 dishes (a mix: 3 veg breakfast, 3 dal/veg curries, 2 non-veg, 2 breads or rice) into data/raw/maharashtra_drafts.csv with verified=no. Show me the rows. Do not draft any others until I approve the pilot.# Recipe-Writing LLM (QLoRA & Jetson llama.cpp Deployment)

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
```





