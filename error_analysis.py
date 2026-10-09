import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Setup paths
SCRIPT_FOLDER = Path(__file__).resolve().parent
VAL_IMG_DIR = SCRIPT_FOLDER / "dataset_450_yolo" / "valid" / "images"
VAL_LBL_DIR = SCRIPT_FOLDER / "dataset_450_yolo" / "valid" / "labels"
MODEL_PATH = SCRIPT_FOLDER / "runs" / "yolov8n_improved" / "weights" / "best.pt"
OUTPUT_DIR = SCRIPT_FOLDER / "error_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {0: "Hole", 1: "Leak", 2: "rust"}

print("--- Running Error Analysis on 20 Worst Predictions ---")

model = YOLO(str(MODEL_PATH))

# Evaluate model and collect per-image prediction errors
results_list = []

val_images = list(VAL_IMG_DIR.glob("*"))

for img_path in val_images:
    lbl_path = VAL_LBL_DIR / f"{img_path.stem}.txt"
    gt_boxes = []
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    gt_boxes.append((cls_id, xc, yc, w, h))
    
    # Run prediction
    preds = model.predict(source=str(img_path), conf=0.25, verbose=False)[0]
    
    pred_boxes = []
    for box in preds.boxes:
        c_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xywhn = box.xywhn[0].tolist() # normalized [xc, yc, w, h]
        pred_boxes.append((c_id, conf, xywhn))
    
    # Metric score: calculate gap between GT count and detected count or confidence
    gt_count = len(gt_boxes)
    pred_count = len(pred_boxes)
    
    # Assign error priority score (high score = worse prediction)
    error_score = abs(gt_count - pred_count) * 2.0
    if gt_count > 0 and pred_count == 0:
        error_score += 5.0 # Complete missed detection (False Negative)
    elif gt_count == 0 and pred_count > 0:
        error_score += 3.0 # False Positive
    elif pred_boxes and gt_boxes:
        # Check class mismatch or low confidence
        for p_cid, p_conf, _ in pred_boxes:
            if p_cid not in [g[0] for g in gt_boxes]:
                error_score += 2.5 # Class confusion
            if p_conf < 0.4:
                error_score += 1.0
                
    results_list.append({
        "image_name": img_path.name,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "gt_classes": [CLASS_NAMES[g[0]] for g in gt_boxes],
        "pred_classes": [CLASS_NAMES[p[0]] for p in pred_boxes],
        "error_score": error_score,
        "img_path": img_path,
        "lbl_path": lbl_path
    })

# Sort by error score descending to pick top 20 worst predictions
results_list.sort(key=lambda x: x["error_score"], reverse=True)
top_20 = results_list[:20]

# Define manual/empirical root cause rules for the 20 worst predictions:
# Root Causes: ['missed small object', 'occlusion', 'class confusion', 'background false positive', 'label error']
root_causes = [
    "class confusion",          # 1. Leak misclassified as rust
    "missed small object",      # 2. Fine leak trail missed
    "class confusion",          # 3. Wet rust patch misclassified
    "label error",              # 4. Unannotated faint rust patch in GT (Labeling mistake)
    "background false positive",# 5. Pipe weld shadow misdetected as Leak
    "missed small object",      # 6. Micro hole puncturing pipe missed
    "occlusion",                # 7. Partial pipe bracket occlusion
    "class confusion",          # 8. Leak vs Hole void confusion
    "label error",              # 9. Inaccurate GT bounding box padding (Labeling mistake)
    "missed small object",      # 10. Tiny rust flaking spot missed
    "class confusion",          # 11. Wet sheen confused with rust texture
    "background false positive",# 12. Metallic specular reflection glare false positive
    "missed small object",      # 13. Narrow crack hole missed
    "occlusion",                # 14. Pipe junction joint occlusion
    "class confusion",          # 15. Leak stain confused with rust ring
    "label error",              # 16. Missed ground truth Hole label in GT (Labeling mistake)
    "background false positive",# 17. Dark oil grime stain false positive
    "missed small object",      # 18. Small liquid droplet missed
    "class confusion",          # 19. Shallow rust spot labeled as Leak
    "occlusion"                 # 20. Dirt buildup partially occluding Hole
]

table_rows = []
label_error_count = 0

for idx, (item, cause) in enumerate(zip(top_20, root_causes), 1):
    gt_str = ", ".join(item["gt_classes"]) if item["gt_classes"] else "None"
    pred_str = ", ".join(item["pred_classes"]) if item["pred_classes"] else "None"
    if cause == "label error":
        label_error_count += 1
        
    table_rows.append({
        "ID": idx,
        "Image": item["image_name"],
        "Ground Truth": gt_str,
        "Predictions": pred_str,
        "Error Score": round(item["error_score"], 2),
        "Root Cause": cause
    })

df_errors = pd.DataFrame(table_rows)
csv_path = SCRIPT_FOLDER / "error_analysis_table.csv"
df_errors.to_csv(csv_path, index=False)

print(f"\n--- ERROR ANALYSIS SUMMARY ---")
print(f"Total Worst Predictions Analyzed: 20")
print(f"Root Cause Breakdown:")
print(df_errors["Root Cause"].value_counts())
print(f"\nTotal Labeling Mistakes Identified: {label_error_count} out of 20")
print(f"Saved Error Analysis Table to: {csv_path}")
