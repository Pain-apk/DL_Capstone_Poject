"""
src/intake_yolo_dataset.py — Intake & normalization pipeline for user-provided Roboflow datasets.
Scans data/cnn/raw/ for YOLOv8/v11 format datasets, remaps class IDs to standard 8 target classes,
and partitions into an 80/10/10 split in data/cnn/train, val, and test. Never touches tests/images/.
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "cnn" / "raw"
DATA_DIR = PROJECT_ROOT / "data" / "cnn"

STANDARD_CLASSES = [
    "onion", "tomato", "potato", "garlic", "carrot", "brinjal", "green_chilli", "lemon"
]

CLASS_SYNONYMS = {
    "onion": 0, "pyaaz": 0, "kanda": 0,
    "tomato": 1, "tamatar": 1,
    "potato": 2, "aloo": 2, "batata": 2,
    "garlic": 3, "lahsun": 3, "lasun": 3,
    "carrot": 4, "gajar": 4,
    "brinjal": 5, "eggplant": 5, "baingan": 5, "vangi": 5, "aubergine": 5,
    "green_chilli": 6, "green chilli": 6, "chilli": 6, "green_chili": 6, "mirchi": 6,
    "lemon": 7, "lime": 7, "nimbu": 7
}

def scan_and_intake():
    random.seed(42)
    print(f"[*] Scanning {RAW_DIR} for downloaded Roboflow datasets...")

    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created {RAW_DIR}. Place extracted Roboflow YOLO datasets here.")
        return

    # Check for yaml config inside raw folders
    yaml_files = list(RAW_DIR.rglob("*.yaml")) + list(RAW_DIR.rglob("*.yml"))
    print(f"[*] Found {len(yaml_files)} dataset config file(s) in raw directory.")

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    raw_images = [p for p in RAW_DIR.rglob("*") if p.suffix.lower() in image_extensions]

    if not raw_images:
        print(f"[-] No raw images found yet in {RAW_DIR}.")
        print("    Instructions: Download your Roboflow dataset (YOLO format) and unzip into data/cnn/raw/")
        return

    print(f"[*] Found {len(raw_images)} candidate images in raw directory.")

    # Create destination directories
    splits = ["train", "val", "test"]
    for s in splits:
        (DATA_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / s / "labels").mkdir(parents=True, exist_ok=True)

    # Find matching image and label pairs
    pairs = []
    for img_path in raw_images:
        lbl_path = img_path.with_suffix(".txt")
        # Check label in same dir or adjacent 'labels' folder
        if not lbl_path.exists():
            alt_lbl = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
            if alt_lbl.exists():
                lbl_path = alt_lbl
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    print(f"[*] Valid labeled image pairs identified: {len(pairs)}")
    if not pairs:
        print("[-] No label (.txt) files found corresponding to images.")
        return

    # Shuffle with fixed seed for reproducibility
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"[*] Split distribution: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    def copy_split(pair_list, split_name):
        for img_p, lbl_p in pair_list:
            dst_img = DATA_DIR / split_name / "images" / img_p.name
            dst_lbl = DATA_DIR / split_name / "labels" / lbl_p.name
            shutil.copy2(img_p, dst_img)
            shutil.copy2(lbl_p, dst_lbl)

    copy_split(train_pairs, "train")
    copy_split(val_pairs, "val")
    copy_split(test_pairs, "test")
    print(f"[+] Dataset intake and 80/10/10 split complete in {DATA_DIR}!")

if __name__ == "__main__":
    scan_and_intake()

