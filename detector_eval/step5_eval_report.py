import os
import sys
import json
import time
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detector_eval.run_eval import load_benchmark_ground_truth

SPLIT_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "split.json")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_best.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "final_config.json")

TARGET_CLASSES = ["onion", "tomato", "potato", "carrot", "cucumber", "lemon", "capsicum"]

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)

report_filenames = set(split["report"])
all_samples = load_benchmark_ground_truth()
report_samples = [s for s in all_samples if s["file_name"] in report_filenames]
assert len(report_samples) == 60

# Adopted map
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    final_map = json.load(f)
final_map["zucchini/courgette"] = "cucumber"

model = YOLO(WEIGHTS_PATH)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Warmup
for s in report_samples[:5]:
    _ = model.predict(source=s["path"], conf=0.20, imgsz=640, device=device, verbose=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()

def evaluate_on_report(conf_val=0.20, imgsz_val=640, aug_val=False):
    latencies = []
    per_image_results = []
    
    for s in report_samples:
        t0 = time.time()
        res = model.predict(source=s["path"], conf=conf_val, imgsz=imgsz_val, augment=aug_val, device=device, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000.0)

        pred_set = set()
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                name = model.names[int(b.cls.item())]
                mapped = final_map.get(name)
                if mapped in TARGET_CLASSES:
                    pred_set.add(mapped)

        per_image_results.append({
            "file_name": s["file_name"],
            "tier": s["tier"],
            "gt_set": s["gt_set"],
            "pred_set": pred_set
        })

    tot_tp, tot_fp, tot_fn = 0, 0, 0
    by_cls = {}
    for cls in TARGET_CLASSES:
        tp = sum(1 for r in per_image_results if cls in r["gt_set"] and cls in r["pred_set"])
        fp = sum(1 for r in per_image_results if cls not in r["gt_set"] and cls in r["pred_set"])
        fn = sum(1 for r in per_image_results if cls in r["gt_set"] and cls not in r["pred_set"])
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else "n/a"
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec != "n/a" and (prec + rec) > 0) else 0.0
        by_cls[cls] = {"TP": tp, "FP": fp, "FN": fn, "P": prec, "R": rec, "F1": f1}

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    fp_per_img = tot_fp / len(report_samples)
    avg_lat = float(np.mean(latencies))

    return by_cls, tot_tp, tot_fp, tot_fn, micro_p, micro_r, micro_f1, fp_per_img, avg_lat

if __name__ == "__main__":
    print("=== FINAL CANDIDATE A EVALUATION ON UNSEEN REPORT HALF (60 IMAGES) ===\n")
    
    # 1. Standard final setting: conf=0.20, imgsz=640, augment=False
    by_cls, tp, fp, fn, up, ur, uf1, fp_img, lat = evaluate_on_report(conf_val=0.20, imgsz_val=640, aug_val=False)
    print(f"Final Setting (conf=0.20, imgsz=640, augment=False):")
    print(f"Micro Metrics: TP={tp}, FP={fp}, FN={fn} | Micro Precision={up:.4f}, Micro Recall={ur:.4f}, Micro F1={uf1:.4f} | FP/img={fp_img:.4f} | Latency={lat:.2f} ms")
    hdr = f"{'Class':<12} {'TP':<5} {'FP':<5} {'FN':<5} {'Precision':<12} {'Recall':<10} {'F1':<10}"
    print(hdr)
    print("-" * len(hdr))
    for cls in TARGET_CLASSES:
        m = by_cls[cls]
        p_str = f"{m['P']:.4f}" if m['P'] != "n/a" else "n/a"
        print(f"{cls:<12} {m['TP']:<5} {m['FP']:<5} {m['FN']:<5} {p_str:<12} {m['R']:<10.4f} {m['F1']:<10.4f}")

    # 2. Save final_config.json
    final_config = {
        "candidate": "Candidate A (best.pt)",
        "weights": "data/cnn/best.pt",
        "conf": 0.20,
        "imgsz": 640,
        "augment": False,
        "device": "cuda:0",
        "target_classes": TARGET_CLASSES,
        "class_map_additions": {
            "zucchini/courgette": "cucumber"
        },
        "report_performance": {
            "total_images": 60,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "micro_precision": round(up, 4),
            "micro_recall": round(ur, 4),
            "micro_f1": round(uf1, 4),
            "fp_per_image": round(fp_img, 4),
            "latency_ms": round(lat, 2)
        }
    }
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
    print(f"\n[+] Saved final configuration to {CONFIG_PATH}")

