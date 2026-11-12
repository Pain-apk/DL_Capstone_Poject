import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detector_eval.run_eval import load_benchmark_ground_truth

SPLIT_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "split.json")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_best.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "detector_eval", "results")

TARGET_CLASSES = ["onion", "tomato", "potato", "carrot", "cucumber", "lemon", "capsicum"]

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)

tune_filenames = set(split["tune"])
all_samples = load_benchmark_ground_truth()
tune_samples = [s for s in all_samples if s["file_name"] in tune_filenames]
assert len(tune_samples) == 60

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_map = json.load(f)

model = YOLO(WEIGHTS_PATH)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Warmup
for s in tune_samples[:5]:
    _ = model.predict(source=s["path"], conf=0.25, imgsz=640, device=device, verbose=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()

def evaluate_setting(conf_val, imgsz_val, augment_val=False):
    latencies = []
    per_image_results = []
    
    for s in tune_samples:
        t0 = time.time()
        res = model.predict(source=s["path"], conf=conf_val, imgsz=imgsz_val, augment=augment_val, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000.0)

        pred_set = set()
        boxes = res[0].boxes
        if boxes is not None and len(boxes) > 0:
            for cls_id in boxes.cls.cpu().numpy():
                name = model.names[int(cls_id)]
                mapped = class_map.get(name)
                if mapped in TARGET_CLASSES:
                    pred_set.add(mapped)

        per_image_results.append({
            "gt_set": s["gt_set"],
            "pred_set": pred_set
        })

    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    for cls in TARGET_CLASSES:
        tp = sum(1 for r in per_image_results if cls in r["gt_set"] and cls in r["pred_set"])
        fp = sum(1 for r in per_image_results if cls not in r["gt_set"] and cls in r["pred_set"])
        fn = sum(1 for r in per_image_results if cls in r["gt_set"] and cls not in r["pred_set"])
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    fp_per_img = tot_fp / len(tune_samples)
    avg_lat = float(np.mean(latencies))

    return {
        "conf": conf_val,
        "imgsz": imgsz_val,
        "augment": augment_val,
        "TP": tot_tp,
        "FP": tot_fp,
        "FN": tot_fn,
        "micro_p": round(micro_p, 4),
        "micro_r": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "fp_per_img": round(fp_per_img, 4),
        "ms_per_img": round(avg_lat, 2)
    }

if __name__ == "__main__":
    conf_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    imgsz_list = [640, 960, 1280]
    
    records = []
    print(f"{'conf':<6} {'imgsz':<6} {'augment':<8} {'TP':<5} {'FP':<5} {'FN':<5} {'Micro P':<10} {'Micro R':<10} {'Micro F1':<10} {'FP/img':<8} {'ms/img':<8}")
    print("-" * 88)
    
    # 1. 15 grid settings without augment
    for sz in imgsz_list:
        for c in conf_list:
            res = evaluate_setting(c, sz, augment_val=False)
            records.append(res)
            print(f"{res['conf']:<6.2f} {res['imgsz']:<6} {str(res['augment']):<8} {res['TP']:<5} {res['FP']:<5} {res['FN']:<5} {res['micro_p']:<10.4f} {res['micro_r']:<10.4f} {res['micro_f1']:<10.4f} {res['fp_per_img']:<8.4f} {res['ms_per_img']:<8.2f}")

    # 2. augment=True once (e.g. at conf=0.25, imgsz=640)
    res_aug = evaluate_setting(0.25, 640, augment_val=True)
    records.append(res_aug)
    print(f"{res_aug['conf']:<6.2f} {res_aug['imgsz']:<6} {str(res_aug['augment']):<8} {res_aug['TP']:<5} {res_aug['FP']:<5} {res_aug['FN']:<5} {res_aug['micro_p']:<10.4f} {res_aug['micro_r']:<10.4f} {res_aug['micro_f1']:<10.4f} {res_aug['fp_per_img']:<8.4f} {res_aug['ms_per_img']:<8.2f}")

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "tune_sweep_candidate_a.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved TUNE sweep results to {out_csv}")

