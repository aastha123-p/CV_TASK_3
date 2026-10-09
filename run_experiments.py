import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_FOLDER / "dataset_450_yolo_improved" / "data.yaml"

print("--- Running 3 Controlled Experiments (Changing 1 Variable Each) ---")

experiments = [
    {
        "exp_id": "Exp_1_HighRes",
        "name": "yolov8n_exp1_imgsz800",
        "variable_changed": "Image Resolution (imgsz=800)",
        "base_val": "imgsz=640",
        "new_val": "imgsz=800",
        "target_cause": "class confusion & missed small object",
        "model": "yolov8n.pt",
        "epochs": 15,
        "imgsz": 800,
        "cls": 1.5
    },
    {
        "exp_id": "Exp_2_ModelCapacity",
        "name": "yolov8s_exp2_small",
        "variable_changed": "Model Backbone Architecture (yolov8s.pt)",
        "base_val": "yolov8n.pt (Nano 3.0M params)",
        "new_val": "yolov8s.pt (Small 11.2M params)",
        "target_cause": "class confusion (capacity bottleneck)",
        "model": "yolov8s.pt",
        "epochs": 15,
        "imgsz": 640,
        "cls": 1.5
    },
    {
        "exp_id": "Exp_3_ClsLossBoost",
        "name": "yolov8n_exp3_cls2.5",
        "variable_changed": "Classification Loss Gain (cls=2.5)",
        "base_val": "cls=1.5",
        "new_val": "cls=2.5",
        "target_cause": "class confusion (misclassification penalty)",
        "model": "yolov8n.pt",
        "epochs": 15,
        "imgsz": 640,
        "cls": 2.5
    }
]

exp_results = []

for exp in experiments:
    print(f"\n========================================================")
    print(f"Running Experiment: {exp['exp_id']} ({exp['variable_changed']})")
    print(f"Targeting Biggest Cause: {exp['target_cause']}")
    print(f"========================================================")
    
    model = YOLO(exp["model"])
    res = model.train(
        data=str(DATA_YAML),
        epochs=exp["epochs"],
        imgsz=exp["imgsz"],
        batch=16,
        device="cpu",
        project=str(SCRIPT_FOLDER / "runs"),
        name=exp["name"],
        seed=42,
        cls=exp["cls"],
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.6,
        mosaic=1.0,
        mixup=0.15,
        verbose=False
    )
    
    val_res = model.val(data=str(DATA_YAML), split="val", verbose=False)
    
    map50 = float(val_res.box.map50)
    map5095 = float(val_res.box.map)
    prec = float(val_res.box.mp)
    rec = float(val_res.box.mr)
    
    # Per-class mAP50 for Leak (class index 1)
    leak_map50 = float(val_res.box.maps[1]) if len(val_res.box.maps) > 1 else 0.0
    
    exp_results.append({
        "Experiment ID": exp["exp_id"],
        "Variable Changed": exp["variable_changed"],
        "Baseline Value": exp["base_val"],
        "New Value": exp["new_val"],
        "Target Root Cause": exp["target_cause"],
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Overall mAP50": round(map50, 4),
        "Overall mAP50-95": round(map5095, 4),
        "Leak mAP50": round(leak_map50, 4)
    })

df_exp = pd.DataFrame(exp_results)
csv_exp_path = SCRIPT_FOLDER / "experiments_log.csv"
df_exp.to_csv(csv_exp_path, index=False)

print("\n--- ALL 3 EXPERIMENTS LOGGED SUCCESSFULLY ---")
print(df_exp.to_string(index=False))
print(f"\nSaved Experiment Log to: {csv_exp_path}")
