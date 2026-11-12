import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO, YOLOWorld

try:
    from ultralytics import YOLOE
except ImportError:
    YOLOE = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_METADATA = os.path.join(PROJECT_ROOT, "detector_eval", "data", "data", "test", "metadata.jsonl")
BENCHMARK_IMG_DIR = os.path.join(PROJECT_ROOT, "detector_eval", "data", "data", "test")
BENCHMARK_MAP_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_benchmark.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "detector_eval", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_CLASSES = ["onion", "tomato", "potato", "carrot", "cucumber", "lemon", "capsicum"]
PROMPTS_20 = [
    "onion", "tomato", "potato", "garlic", "ginger", "green chilli", "carrot",
    "brinjal", "capsicum", "cucumber", "lemon", "green peas", "cauliflower",
    "cabbage", "okra", "spinach", "coriander leaves", "egg", "paneer", "curd"
]

def load_benchmark_ground_truth():
    with open(BENCHMARK_MAP_PATH, "r", encoding="utf-8") as f:
        bench_map = json.load(f)

    samples = []
    with open(BENCHMARK_METADATA, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            fn = item["file_name"]
            img_path = os.path.join(BENCHMARK_IMG_DIR, fn)
            tier = item.get("tier", "unknown")
            confusing_pairs = item.get("confusing_pairs", [])
            
            # Canonical ground truth set for the 7 shared classes
            gt_set = set()
            raw_cats = item["objects"].get("category_names", [])
            for c in raw_cats:
                mapped = bench_map.get(c)
                if mapped in TARGET_CLASSES:
                    gt_set.add(mapped)
                    
            samples.append({
                "file_name": fn,
                "path": img_path,
                "tier": tier,
                "confusing_pairs": confusing_pairs,
                "raw_categories": raw_cats,
                "gt_set": gt_set
            })
    return samples

def init_model(candidate_name, weights_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing candidate '{candidate_name}' on {device} from {weights_path}...")
    
    if candidate_name == "best":
        model = YOLO(weights_path)
        class_map_file = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_best.json")
    elif candidate_name == "yoloworld":
        model = YOLOWorld(weights_path)
        model.set_classes(PROMPTS_20)
        class_map_file = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_yoloworld.json")
    elif candidate_name == "yoloe":
        if YOLOE is None:
            raise RuntimeError("YOLOE could not be imported from ultralytics.")
        model = YOLOE(weights_path)
        text_pe = model.get_text_pe(PROMPTS_20)
        model.set_classes(PROMPTS_20, text_pe)
        class_map_file = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_yoloe.json")
    else:
        raise ValueError(f"Unknown candidate: {candidate_name}")

    with open(class_map_file, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    return model, class_map, file_size_mb

def evaluate_candidate_at_conf(model, class_map, samples, conf_thresh, device="cuda:0"):
    # Warmup
    for i in range(min(5, len(samples))):
        _ = model.predict(source=samples[i]["path"], conf=conf_thresh, device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Evaluation loop
    latencies = []
    per_image_results = []
    
    for sample in samples:
        t0 = time.time()
        res = model.predict(source=sample["path"], conf=conf_thresh, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000.0) # ms

        # Detected set of target classes
        pred_set = set()
        det_boxes = res[0].boxes
        if det_boxes is not None and len(det_boxes) > 0:
            for cls_id in det_boxes.cls.cpu().numpy():
                cls_name = model.names[int(cls_id)]
                mapped = class_map.get(cls_name)
                if mapped in TARGET_CLASSES:
                    pred_set.add(mapped)

        per_image_results.append({
            "file_name": sample["file_name"],
            "tier": sample["tier"],
            "confusing_pairs": sample["confusing_pairs"],
            "raw_categories": sample["raw_categories"],
            "gt_set": sample["gt_set"],
            "pred_set": pred_set
        })

    avg_latency = float(np.mean(latencies))
    return per_image_results, avg_latency

def compute_metrics(per_image_results, target_classes=TARGET_CLASSES):
    # Overall and per-class stats
    metrics_by_class = {}
    
    for cls in target_classes:
        tp = sum(1 for r in per_image_results if cls in r["gt_set"] and cls in r["pred_set"])
        fp = sum(1 for r in per_image_results if cls not in r["gt_set"] and cls in r["pred_set"])
        fn = sum(1 for r in per_image_results if cls in r["gt_set"] and cls not in r["pred_set"])
        tn = sum(1 for r in per_image_results if cls not in r["gt_set"] and cls not in r["pred_set"])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        metrics_by_class[cls] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        }

    # Macro averages
    macro_prec = float(np.mean([m["Precision"] for m in metrics_by_class.values()]))
    macro_rec = float(np.mean([m["Recall"] for m in metrics_by_class.values()]))
    macro_f1 = float(np.mean([m["F1"] for m in metrics_by_class.values()]))

    # Total False Positives across all target classes
    total_fp = sum(m["FP"] for m in metrics_by_class.values())
    total_tp = sum(m["TP"] for m in metrics_by_class.values())
    total_fn = sum(m["FN"] for m in metrics_by_class.values())
    fp_per_image = total_fp / len(per_image_results) if per_image_results else 0.0

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "total_fp": total_fp,
        "fp_per_image": fp_per_image,
        "per_class": metrics_by_class
    }

def analyze_confusion_and_errors(per_image_results):
    confusion_records = []
    for r in per_image_results:
        # Check false positives
        fps = r["pred_set"] - r["gt_set"]
        fns = r["gt_set"] - r["pred_set"]
        if fps:
            for fp_cls in fps:
                confusion_records.append({
                    "file_name": r["file_name"],
                    "tier": r["tier"],
                    "fp_class": fp_cls,
                    "gt_classes": list(r["gt_set"]),
                    "raw_all_categories": r["raw_categories"],
                    "confusing_pairs_annotated": r["confusing_pairs"]
                })
    return confusion_records

def run_evaluation_for_candidate(candidate_name, weights_path, license_str):
    samples = load_benchmark_ground_truth()
    model, class_map, file_size_mb = init_model(candidate_name, weights_path)
    
    thresholds = [0.25, 0.40, 0.50, 0.60]
    candidate_summary = []
    all_class_rows = []
    all_confusions = []
    
    for conf in thresholds:
        print(f"\n--- Running {candidate_name} @ conf={conf} ---")
        img_results, latency = evaluate_candidate_at_conf(model, class_map, samples, conf)
        
        # Overall metrics
        overall = compute_metrics(img_results)
        
        # Tier metrics
        tier_metrics = {}
        for tier in ["simple", "medium", "hard"]:
            tier_subset = [r for r in img_results if r["tier"] == tier]
            tm = compute_metrics(tier_subset)
            tier_metrics[tier] = tm

        summary_row = {
            "candidate": candidate_name,
            "threshold": conf,
            "macro_precision": round(overall["macro_precision"], 4),
            "macro_recall": round(overall["macro_recall"], 4),
            "macro_f1": round(overall["macro_f1"], 4),
            "fp_per_image": round(overall["fp_per_image"], 4),
            "latency_ms": round(latency, 2),
            "model_size_mb": round(file_size_mb, 2),
            "license": license_str,
            "simple_f1": round(tier_metrics["simple"]["macro_f1"], 4),
            "medium_f1": round(tier_metrics["medium"]["macro_f1"], 4),
            "hard_f1": round(tier_metrics["hard"]["macro_f1"], 4)
        }
        candidate_summary.append(summary_row)
        
        print(f"Results @ conf {conf}: Macro P={summary_row['macro_precision']:.3f}, R={summary_row['macro_recall']:.3f}, F1={summary_row['macro_f1']:.3f}, FP/img={summary_row['fp_per_image']:.2f}, Latency={summary_row['latency_ms']}ms, Size={summary_row['model_size_mb']}MB")
        print(f"  Tiers F1: simple={summary_row['simple_f1']:.3f}, medium={summary_row['medium_f1']:.3f}, hard={summary_row['hard_f1']:.3f}")
        
        # Save per-class rows
        for cls, cm in overall["per_class"].items():
            all_class_rows.append({
                "candidate": candidate_name,
                "threshold": conf,
                "tier": "all",
                "class": cls,
                "TP": cm["TP"],
                "FP": cm["FP"],
                "FN": cm["FN"],
                "Precision": round(cm["Precision"], 4),
                "Recall": round(cm["Recall"], 4),
                "F1": round(cm["F1"], 4)
            })

        confusions = analyze_confusion_and_errors(img_results)
        for c in confusions:
            c["candidate"] = candidate_name
            c["threshold"] = conf
            all_confusions.append(c)

    return candidate_summary, all_class_rows, all_confusions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=["best", "yoloworld", "yoloe"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()

    summary, class_rows, confusions = run_evaluation_for_candidate(args.candidate, args.weights, args.license)

    # Save candidate specific CSVs
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(RESULTS_DIR, f"summary_{args.candidate}.csv"), index=False)

    df_cls = pd.DataFrame(class_rows)
    df_cls.to_csv(os.path.join(RESULTS_DIR, f"per_class_{args.candidate}.csv"), index=False)

    df_conf = pd.DataFrame(confusions)
    df_conf.to_csv(os.path.join(RESULTS_DIR, f"confusion_{args.candidate}.csv"), index=False)
    print(f"\n[+] Saved evaluation results for {args.candidate} to {RESULTS_DIR}")

