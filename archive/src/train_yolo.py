"""
src/train_yolo.py — Fine-tune YOLO11n on data/cnn/ for the 8 target kitchen ingredient classes.
Hyperparameters: imgsz=640, 40 epochs, patience=10, workers=2, batch=16 (fits comfortably in 7.2 GB VRAM).
Guarded: Never trains on tests/images/.
"""

import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = PROJECT_ROOT / "data" / "cnn" / "data.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "detector"
TESTS_IMAGES_DIR = PROJECT_ROOT / "tests" / "images"

def check_training_guardrails():
    # Strict isolation check
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        content = f.read().lower()
        if "tests/images" in content or "tests\\images" in content:
            raise ValueError("[FATAL] data.yaml references tests/images! Never train on test evaluation images.")
    print("[*] Guardrail passed: tests/images/ is strictly isolated from training data.")

def train():
    check_training_guardrails()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Never train on CPU!")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[*] Training on GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.2f} GB VRAM)")

    from ultralytics import YOLO

    # Load pretrained YOLO11n base
    model = YOLO("yolo11n.pt")

    # Hyperparameters per spec:
    # 40 epochs, patience 10, imgsz 640, workers 2, batch 16 (fits ~2.8 GB VRAM on RTX 5050)
    print("[*] Starting YOLO11n fine-tuning...")
    results = model.train(
        data=str(YAML_PATH),
        epochs=40,
        patience=10,
        imgsz=640,
        batch=16,
        workers=2,
        project=str(OUTPUTS_DIR),
        name="ingredient_detector",
        exist_ok=True,
        save=True,
        plots=True,
        seed=42
    )

    best_weights = OUTPUTS_DIR / "ingredient_detector" / "weights" / "best.pt"
    print(f"[+] Training complete! Best weights saved to: {best_weights}")
    return results

if __name__ == "__main__":
    train()

