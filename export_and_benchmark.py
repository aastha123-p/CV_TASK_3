import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

SCRIPT_FOLDER = Path(__file__).resolve().parent
MODEL_PT_PATH = SCRIPT_FOLDER / "runs" / "yolov8n_improved" / "weights" / "best.pt"
MODEL_ONNX_PATH = SCRIPT_FOLDER / "runs" / "yolov8n_improved" / "weights" / "best.onnx"

print("--- Step 1: Exporting Best Model to ONNX Format ---")
if not MODEL_ONNX_PATH.exists():
    model = YOLO(str(MODEL_PT_PATH))
    onnx_file_path = model.export(format="onnx", imgsz=640, dynamic=False, opset=12)
    print(f"✅ Successfully exported ONNX model to: {onnx_file_path}")
else:
    print(f"✅ ONNX model already exists at: {MODEL_ONNX_PATH}")

print("\n--- Step 2: Benchmarking Inference Latency (PyTorch vs ONNX) ---")

# Dummy image: H=640, W=640, C=3 uint8
dummy_img_hwc = np.zeros((640, 640, 3), dtype=np.uint8)
# Preprocessed float32 tensor for raw ONNX session [1, 3, 640, 640]
dummy_img_chw = np.transpose(dummy_img_hwc, (2, 0, 1)).astype(np.float32) / 255.0
dummy_img_bchw = np.expand_dims(dummy_img_chw, axis=0)

num_warmup = 10
num_runs = 100

# 1. Benchmark PyTorch (.pt) Model via Ultralytics Predict
pt_model = YOLO(str(MODEL_PT_PATH))
for _ in range(num_warmup):
    _ = pt_model.predict(source=dummy_img_hwc, verbose=False)

pt_start = time.perf_counter()
for _ in range(num_runs):
    _ = pt_model.predict(source=dummy_img_hwc, verbose=False)
pt_end = time.perf_counter()

pt_avg_ms = ((pt_end - pt_start) / num_runs) * 1000.0

# 2. Benchmark ONNX Runtime (.onnx) Session
session = ort.InferenceSession(str(MODEL_ONNX_PATH), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

for _ in range(num_warmup):
    _ = session.run(None, {input_name: dummy_img_bchw})

onnx_start = time.perf_counter()
for _ in range(num_runs):
    _ = session.run(None, {input_name: dummy_img_bchw})
onnx_end = time.perf_counter()

onnx_avg_ms = ((onnx_end - onnx_start) / num_runs) * 1000.0
speedup_factor = pt_avg_ms / onnx_avg_ms

benchmark_results = [
    {
        "Framework / Engine": "PyTorch (.pt)",
        "Device": "CPU (Intel Core i5)",
        "Input Resolution": "640x640",
        "Average Latency (ms)": round(pt_avg_ms, 2),
        "FPS (Frames/sec)": round(1000.0 / pt_avg_ms, 1),
        "Speedup": "1.0x (Baseline)"
    },
    {
        "Framework / Engine": "ONNX Runtime (.onnx)",
        "Device": "CPU (Intel Core i5)",
        "Input Resolution": "640x640",
        "Average Latency (ms)": round(onnx_avg_ms, 2),
        "FPS (Frames/sec)": round(1000.0 / onnx_avg_ms, 1),
        "Speedup": f"{round(speedup_factor, 2)}x Faster"
    }
]

df_benchmark = pd.DataFrame(benchmark_results)
benchmark_csv = SCRIPT_FOLDER / "onnx_latency_benchmark.csv"
df_benchmark.to_csv(benchmark_csv, index=False)

print("\n================ LATENCY COMPARISON RESULTS ================")
print(df_benchmark.to_string(index=False))
print(f"\nSaved Latency Benchmark Table to: {benchmark_csv}")
