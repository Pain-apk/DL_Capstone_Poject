import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detector_eval.run_eval import load_benchmark_ground_truth, init_model, evaluate_candidate_at_conf

samples = load_benchmark_ground_truth()
target_classes = ["onion", "tomato", "potato", "carrot", "cucumber", "lemon", "capsicum"]

def score_results(img_results):
    metrics = {}
    for cls in target_classes:
        tp = sum(1 for r in img_results if cls in r["gt_set"] and cls in r["pred_set"])
        fp = sum(1 for r in img_results if cls not in r["gt_set"] and cls in r["pred_set"])
        fn = sum(1 for r in img_results if cls in r["gt_set"] and cls not in r["pred_set"])
        
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else "n/a"
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if prec != "n/a" and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        metrics[cls] = {"TP": tp, "FP": fp, "FN": fn, "P": prec, "R": rec, "F1": f1}
        
    tot_tp = sum(m["TP"] for m in metrics.values())
    tot_fp = sum(m["FP"] for m in metrics.values())
    tot_fn = sum(m["FN"] for m in metrics.values())
    
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    return metrics, tot_tp, tot_fp, tot_fn, micro_p, micro_r, micro_f1

candidates = [
    ("Candidate A (best.pt)", "best", os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt")),
    ("Candidate B (yolov8s-worldv2)", "yoloworld", os.path.join(PROJECT_ROOT, "detector_eval", "test_setup", "yolov8s-worldv2.pt")),
    ("Candidate C (yoloe-11s-seg)", "yoloe", os.path.join(PROJECT_ROOT, "detector_eval", "models", "yoloe-11s-seg.pt"))
]

if __name__ == "__main__":
    for label, cand, weights in candidates:
        print(f"\n{'='*30} {label} {'='*30}")
        model, class_map, _ = init_model(cand, weights)
        for conf in [0.25, 0.40, 0.50, 0.60]:
            img_results, latency = evaluate_candidate_at_conf(model, class_map, samples, conf)
            cls_m, t_tp, t_fp, t_fn, u_p, u_r, u_f1 = score_results(img_results)
            print(f"\n--- Threshold: {conf} | Latency: {latency:.2f} ms ---")
            print(f"Micro Metrics: TP={t_tp}, FP={t_fp}, FN={t_fn} | Micro Precision={u_p:.4f}, Micro Recall={u_r:.4f}, Micro F1={u_f1:.4f} | FP/img={t_fp/len(samples):.4f}")
            hdr = f"{'Class':<12} {'TP':<5} {'FP':<5} {'FN':<5} {'Precision':<12} {'Recall':<10} {'F1':<10}"
            print(hdr)
            print("-" * len(hdr))
            for cls in target_classes:
                m = cls_m[cls]
                p_str = f"{m['P']:.4f}" if m['P'] != "n/a" else "n/a"
                print(f"{cls:<12} {m['TP']:<5} {m['FP']:<5} {m['FN']:<5} {p_str:<12} {m['R']:<10.4f} {m['F1']:<10.4f}")

