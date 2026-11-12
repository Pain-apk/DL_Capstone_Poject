"""
scratch/benchmark_detector.py — Benchmarks app/detector.py against tests/images/ and tests/expected.json.
Reports per-photo and overall precision, recall, and inference time per image.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detector import detect

EXPECTED_JSON = os.path.join(PROJECT_ROOT, "tests", "expected.json")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "tests", "images")

def run_benchmark():
    os.environ.pop("MOCK_DETECTOR", None)
    
    with open(EXPECTED_JSON, "r", encoding="utf-8") as f:
        expected = json.load(f)

    print("=" * 80)
    print("DETECTOR BENCHMARK ON REAL TEST IMAGES")
    print("=" * 80)
    print(f"{'Image':<22} | {'Time (ms)':<9} | {'Expected':<18} | {'Detected (High conf)':<22} | {'P':<4} | {'R':<4}")
    print("-" * 80)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    times = []

    # Warm-up run to exclude initial PyTorch JIT overhead
    dummy_bytes = open(os.path.join(IMAGES_DIR, list(expected.keys())[0]), "rb").read()
    _ = detect(dummy_bytes)

    for fn, exp_classes in sorted(expected.items()):
        img_path = os.path.join(IMAGES_DIR, fn)
        if not os.path.exists(img_path):
            print(f"[!] Missing {img_path}")
            continue

        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        t0 = time.time()
        res = detect(raw_bytes)
        dt = (time.time() - t0) * 1000
        times.append(dt)

        det_names = [i["name"] for i in res.get("ingredients", [])]
        
        # Calculate Precision and Recall for this photo
        tp = sum(1 for d in det_names if d in exp_classes)
        fp = len(det_names) - tp
        fn_count = sum(1 for e in exp_classes if e not in det_names)

        total_tp += tp
        total_fp += fp
        total_fn += fn_count

        prec = tp / max(tp + fp, 1)
        rec = tp / max(len(exp_classes), 1)

        det_str = ", ".join(f"{i['name']}({i['confidence']:.2f})" for i in res.get("ingredients", [])[:3])
        exp_str = ", ".join(exp_classes)
        print(f"{fn:<22} | {dt:7.1f}ms | {exp_str:<18} | {det_str:<22} | {prec:.2f} | {rec:.2f}")

    overall_prec = total_tp / max(total_tp + total_fp, 1)
    overall_rec = total_tp / max(total_tp + total_fn, 1)
    avg_time = sum(times) / max(len(times), 1)

    print("=" * 80)
    print(f"Overall Precision: {overall_prec:.4f} ({overall_prec*100:.1f}%)")
    print(f"Overall Recall:    {overall_rec:.4f} ({overall_rec*100:.1f}%)")
    print(f"Average Latency:   {avg_time:.2f} ms / image (FPS: {1000/avg_time:.1f})")
    print(f"Total Test Images: {len(times)}")
    print("=" * 80)

    return {
        "precision": overall_prec,
        "recall": overall_rec,
        "avg_latency_ms": avg_time,
        "total_images": len(times)
    }

if __name__ == "__main__":
    run_benchmark()
