import os
import sys
import json
import torch
import numpy as np
from ultralytics import YOLO, YOLOWorld
try:
    from ultralytics.models.yolo.yoloe import YOLOE
except ImportError:
    try:
        from ultralytics import YOLOE
    except ImportError:
        YOLOE = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detector_eval.run_eval import load_benchmark_ground_truth

PROMPTS_7 = ["egg", "paneer", "curd", "okra", "cabbage", "spinach", "coriander leaves"]
THRESHOLDS = [0.25, 0.35, 0.45, 0.55]

BENCHMARK_SAMPLES = load_benchmark_ground_truth()
assert len(BENCHMARK_SAMPLES) == 120

TEST1_PATH = os.path.join(PROJECT_ROOT, "tests", "images", "test1.jpg")
MY_PHOTOS_DIR = os.path.join(PROJECT_ROOT, "detector_eval", "my_photos")

CANDIDATES = [
    ("Candidate B (yolov8s-worldv2)", "yoloworld", os.path.join(PROJECT_ROOT, "detector_eval", "test_setup", "yolov8s-worldv2.pt")),
    ("Candidate C (yoloe-11s-seg)", "yoloe", os.path.join(PROJECT_ROOT, "detector_eval", "models", "yoloe-11s-seg.pt"))
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def run_open_vocab_eval():
    results_summary = {}

    for cand_label, cand_type, weights_path in CANDIDATES:
        print(f"\n{'='*30}\n{cand_label}\n{'='*30}")
        if cand_type == "yoloworld":
            model = YOLOWorld(weights_path)
            model.set_classes(PROMPTS_7)
        elif cand_type == "yoloe":
            if YOLOE is None:
                raise RuntimeError("YOLOE could not be imported.")
            model = YOLOE(weights_path)
            text_pe = model.get_text_pe(PROMPTS_7)
            model.set_classes(PROMPTS_7, text_pe)

        cand_results = {}

        # 1. Benchmark 120 images (all detections are FP)
        print(f"\n--- Benchmark 120 Images (False Positive Counts across 4 thresholds) ---")
        # Format table header
        hdr = f"{'Threshold':<12} " + " ".join([f"{p:<14}" for p in PROMPTS_7]) + f"{'Total FP':<10}"
        print(hdr)
        print("-" * len(hdr))

        for conf in THRESHOLDS:
            fp_by_class = {p: 0 for p in PROMPTS_7}
            total_fp_boxes = 0

            for sample in BENCHMARK_SAMPLES:
                res = model.predict(source=sample["path"], conf=conf, device=device, verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    for b in res.boxes:
                        cls_name = model.names[int(b.cls.item())]
                        if cls_name in fp_by_class:
                            fp_by_class[cls_name] += 1
                            total_fp_boxes += 1

            row_str = f"{conf:<12.2f} " + " ".join([f"{fp_by_class[p]:<14}" for p in PROMPTS_7]) + f"{total_fp_boxes:<10}"
            print(row_str)
            cand_results[conf] = {"fp_by_class": fp_by_class, "total_fp": total_fp_boxes}

        # 2. Test on tests/images/test1.jpg
        print(f"\n--- Detections on tests/images/test1.jpg ---")
        test1_dets = {}
        for conf in THRESHOLDS:
            res = model.predict(source=TEST1_PATH, conf=conf, device=device, verbose=False)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_name = model.names[int(b.cls.item())]
                    score = float(b.conf.item())
                    dets.append((cls_name, round(score, 4)))
            test1_dets[conf] = dets
            print(f"  conf={conf:.2f}: {dets}")

        # 3. Check detector_eval/my_photos/
        my_photos_dets = {}
        if os.path.exists(MY_PHOTOS_DIR):
            photo_files = [os.path.join(MY_PHOTOS_DIR, f) for f in os.listdir(MY_PHOTOS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"\n--- Detections on detector_eval/my_photos/ ({len(photo_files)} photos) ---")
            for pf in photo_files:
                base = os.path.basename(pf)
                print(f"  Photo: {base}")
                for conf in THRESHOLDS:
                    res = model.predict(source=pf, conf=conf, device=device, verbose=False)[0]
                    dets = [(model.names[int(b.cls.item())], round(float(b.conf.item()), 4)) for b in res.boxes] if res.boxes is not None else []
                    print(f"    conf={conf:.2f}: {dets}")
        else:
            print("\n--- detector_eval/my_photos/ does NOT exist (skipped) ---")

        results_summary[cand_label] = cand_results

    return results_summary

if __name__ == "__main__":
    run_open_vocab_eval()

