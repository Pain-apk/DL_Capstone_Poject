# NVIDIA Jetson Edge Deployment Guide

This document details the exact architectural adaptations, software dependencies, and runtime configurations required to deploy the **Pantry to Plate** system on an **NVIDIA Jetson** edge device (e.g., Jetson Orin Nano 8GB, Jetson Orin NX 16GB, or Jetson AGX Orin).

---

## 1. Architecture Overview (Laptop vs. Jetson)

| Component | Development Laptop (Current) | NVIDIA Jetson Edge Target | Rationale |
| :--- | :--- | :--- | :--- |
| **GPU / Compute** | RTX 5050 Laptop (8GB, `sm_120`) | NVIDIA Orin Ampere GPU (`sm_87`) | Shared unified memory (CPU + GPU share RAM) |
| **Operating System** | Windows 11 (PowerShell) | JetPack 6.x (Ubuntu 22.04 LTS aarch64) | Standard Jetson Linux BSP |
| **Python Version** | Python 3.11.0 (Windows) | Python 3.10 / 3.8 (Linux aarch64) | JetPack system Python standard |
| **Ingredient Detector** | Ultralytics PyTorch (`.pt` weights) | **TensorRT Engine (`.engine`)** | 3–5× faster inference, zero PyTorch runtime overhead |
| **Recipe LLM Writer** | HuggingFace 4-bit bitsandbytes | **`llama.cpp` GGUF (`Q4_K_M`)** | CPU/GPU unified memory offloading, FlashAttention, fast edge prompt evaluation |
| **Server Framework** | FastAPI + Uvicorn | FastAPI + Uvicorn (Inside Docker) | Identical API surface and contract |

---

## 2. Component 1: TensorRT Ingredient Detector

On NVIDIA Jetson, running PyTorch models directly consumes excessive memory. We export the trained YOLO weights (`best.pt`) directly to a TensorRT engine.

### Export Command (Run on Jetson):
```bash
# Export trained YOLO model to TensorRT engine FP16
yolo export model=data/cnn/best.pt format=engine half=True device=0
```
This generates `data/cnn/best.engine`.

### TensorRT Detector Wrapper (`app/detector_jetson.py`):
In `app/detector.py`, replace or wrap the inference call:
```python
import cv2
import numpy as np
from ultralytics import YOLO

# Ultralytics natively executes TensorRT engines on Jetson
model = YOLO("data/cnn/best.engine")
results = model(pil_image)
```

---

## 3. Component 2: Edge Recipe LLM via `llama.cpp` (GGUF Q4_K_M)

On an 8 GB Jetson Orin Nano, running 1.5B parameters via `llama.cpp` uses only ~1.2 GB unified memory.

### Step-by-Step GGUF Conversion:
1. **Merge LoRA Adapter on Host/Laptop:**
   ```bash
   python -c "
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel

   base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.float16)
   model = PeftModel.from_pretrained(base, 'outputs/pilot_unverified/adapter')
   merged = model.merge_and_unload()
   merged.save_pretrained('outputs/pilot_unverified/merged_pilot')
   tokenizer = AutoTokenizer.from_pretrained('outputs/pilot_unverified/adapter')
   tokenizer.save_pretrained('outputs/pilot_unverified/merged_pilot')
   "
   ```

2. **Convert to FP16 GGUF using `llama.cpp`:**
   ```bash
   python llama.cpp/convert_hf_to_gguf.py outputs/pilot_unverified/merged_pilot/ --outtype f16 --outfile outputs/pilot_unverified/pilot-f16.gguf
   ```

3. **Quantize to Q4_K_M:**
   ```bash
   ./llama.cpp/llama-quantize outputs/pilot_unverified/pilot-f16.gguf outputs/pilot_unverified/pilot-q4_k_m.gguf Q4_K_M
   ```

4. **Benchmarking with Jetson Optimized Runtime Flags:**
   ```bash
   ./llama.cpp/llama-cli \
     -m outputs/pilot_unverified/pilot-q4_k_m.gguf \
     -n 400 \
     -ngl 99 \
     -fa \
     -ctk q8_0 -ctv q8_0 \
     --prompt "<prompt>"
   ```
   *Flags:*
   - `-ngl 99`: Offload all transformer layers to Jetson GPU.
   - `-fa`: Enable FlashAttention on Orin tensor cores.
   - `-ctk q8_0 -ctv q8_0`: 8-bit quantized KV-cache, reducing context memory by 50%.

---

## 4. Jetson Environment Variables

Configure these environment variables in `/etc/environment` or `.env`:

```bash
# Hardware & Backend Configuration
LLM_BACKEND=llamacpp
LLAMACPP_MODEL_PATH=/app/outputs/pilot-q4_k_m.gguf
LLAMACPP_THREADS=4
DETECTOR_BACKEND=tensorrt
DETECTOR_MODEL=/app/data/cnn/best.engine

# Application settings
LLM_ENABLED=true
MOCK_DETECTOR=0
API_TOKEN=your-secure-edge-token
DEBUG=0
PORT=8000
HOST=0.0.0.0
```

---

## 5. Dockerfile for Jetson (JetPack 6.x)

Deploying via NVIDIA's official JetPack base container ensures CUDA, cuDNN, and TensorRT are natively available without driver mismatches:

```dockerfile
# Base image with JetPack 6.0 (CUDA 12.2, TensorRT 8.6, Ubuntu 22.04 aarch64)
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install llama-cpp-python with CUDA/cuBLAS support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN python3 -m pip install llama-cpp-python --verbose

# Install project dependencies
COPY requirements-jetson.txt .
RUN python3 -m pip install -r requirements-jetson.txt

# Copy application code and model artifacts
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Jetson Memory Budget & Verification

On an 8 GB Jetson Orin Nano, memory is shared dynamically:

| Component | Resident Memory | Peak Usage |
| :--- | :--- | :--- |
| Ubuntu System + JetPack OS | ~1.1 GB | ~1.3 GB |
| TensorRT Detector (YOLO11n FP16) | ~0.3 GB | ~0.5 GB |
| `llama.cpp` LLM Engine (Q4_K_M + 8-bit KV) | ~1.2 GB | ~1.6 GB |
| FastAPI Server + Buffers | ~0.2 GB | ~0.4 GB |
| **Total System RAM / VRAM** | **~2.8 GB** | **~3.8 GB (< 8.0 GB Limit)** |

The system comfortably fits within the 8 GB edge budget with over 4 GB of headroom for system services and camera stream frame buffers.

