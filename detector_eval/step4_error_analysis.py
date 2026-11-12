import os
import sys
import json
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detector_eval.run_eval import load_benchmark_ground_truth

SPLIT_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "split.json")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "detector_eval", "class_map_best.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "data", "cnn", "best.pt")

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)

tune_filenames = set(split["tune"])
all_samples = load_benchmark_ground_truth()
tune_samples = [s for s in all_samples if s["file_name"] in tune_filenames]

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_map = json.load(f)

model = YOLO(WEIGHTS_PATH)
target_investigate = ["onion", "potato", "cucumber"]

print("=== INSPECTING MISSED ONION, POTATO, CUCUMBER ON TUNE (conf=0.25, imgsz=640) ===\n")

missed_records = {k: [] for k in target_investigate}

for s in tune_samples:
    res = model.predict(source=s["path"], conf=0.25, imgsz=640, device="cuda:0", verbose=False)[0]
    
    raw_preds = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            c_name = model.names[int(b.cls.item())]
            c_conf = float(b.conf.item())
            raw_preds.append((c_name, round(c_conf, 3)))
            
    mapped_preds = set(class_map.get(name) for name, _ in raw_preds if class_map.get(name))
    
    for target in target_investigate:
        if target in s["gt_set"] and target not in mapped_preds:
            missed_records[target].append({
                "file_name": s["file_name"],
                "all_gt": list(s["gt_set"]),
                "raw_predictions": raw_preds
            })

for target in target_investigate:
    print(f"*** Target: {target.upper()} | Total Missed: {len(missed_records[target])} ***")
    for m in missed_records[target]:
        if m["raw_predictions"]:
            preds_str = ", ".join([f"{name} ({conf})" for name, conf in m["raw_predictions"]])
        else:
            preds_str = "nothing"
        print(f"  {m['file_name']} (GT: {m['all_gt']}) -> predicted: {preds_str}")
    print()

# Now investigate potential additions
print("=== POTENTIAL CLASS MAP ADDITIONS AND THEIR FREQUENCY ON TUNE ===")
# Count raw predictions of unmapped classes across all 60 TUNE images
unmapped_counts = {}
for s in tune_samples:
    res = model.predict(source=s["path"], conf=0.25, imgsz=640, device="cuda:0", verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            name = model.names[int(b.cls.item())]
            if class_map.get(name) is None:
                unmapped_counts[name] = unmapped_counts.get(name, 0) + 1

print("Unmapped predicted classes on TUNE:", unmapped_counts)
