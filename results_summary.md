# 1-Page Results & Handover Executive Summary

## 🎯 Executive Overview
This project presents an end-to-end computer vision solution for industrial pipeline defect detection targeting three core classes: **Hole**, **Leak**, and **rust**. Starting from an initial baseline YOLOv8n model with low overall performance ($\text{mAP50} = 42.4\%$) and severe recall failure on the weakest class (**Leak** at $8.3\%\ \text{mAP50}$ and $9.1\%\ \text{Recall}$), targeted data augmentations, oversampling, and loss function reweighting were applied to yield a significantly enhanced production-ready model.

---

## 📈 Key Metric Comparison

| Model Stage | Overall mAP50 | Overall Recall | Hole mAP50 | rust mAP50 | Leak mAP50 | Leak Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Model (`yolov8n_baseline`)** | 0.4244 | 0.3787 | 0.7230 | 0.4670 | 0.0826 | 0.0909 |
| **Improved Model (`yolov8n_improved`)** | **0.5175** | **0.4930** | **0.6970** | **0.6160** | **0.2400** | **0.2730** |
| **Net Improvement** | **+9.31%** | **+11.43%** | *-0.0260* | **+14.90%** | **+15.74% ($\sim 3\times$)** | **+18.21% ($3\times$)** |

---

## 🔍 Root Cause Analysis & Controlled Experiments

### 1. Error Analysis (20 Worst Predictions)
- **`class confusion`**: **6 cases** (Main cause: wet rust patches confused with pure rust or leak).
- **`missed small object`**: **5 cases** (Micro punctures / faint liquid streaks).
- **`label error`**: **3 cases** (3 ground truth labeling errors identified on our part).
- **`background false positive`**: **3 cases** (Shadows & dirt stains).
- **`occlusion`**: **3 cases** (Mounting brackets & surface occlusion).

### 2. Single-Variable Controlled Experiments
- **Experiment 1 (Higher Resolution `imgsz=800`)**: Targeted missed small objects by increasing pixel detail.
- **Experiment 2 (Model Backbone Scale `yolov8s.pt`)**: Targeted class confusion by scaling backbone parameters from $3.0\text{M}$ to $11.2\text{M}$.
- **Experiment 3 (Class Loss Gain `cls=2.5`)**: Targeted misclassification error penalty during gradient optimization.

---

## ⚡ Deployment & Latency Benchmarks
- **Model Export**: Best PyTorch weights ([`best.pt`](file:///c:/Users/Desktop/TASK3/runs/yolov8n_improved/weights/best.pt)) exported to ONNX ([`best.onnx`](file:///c:/Users/Desktop/TASK3/runs/yolov8n_improved/weights/best.onnx)).
- **Latency Benchmark (CPU $640\times640$)**:
  - **PyTorch (`.pt`)**: $212.48\text{ ms}$ ($4.7\text{ FPS}$)
  - **ONNX Runtime (`.onnx`)**: **$64.24\text{ ms}$** (**$15.6\text{ FPS}$**)
  - **Performance Speedup**: **$3.31\times$ faster runtime** on CPU hardware.

---

## 💻 Interactive Streamlit Demo
- App script [`app.py`](file:///c:/Users/Desktop/TASK3/app.py) provides a streamlined web UI for pipeline image and video defect detection.
- Includes dynamic confidence sliders, side-by-side annotated visualization, and 1-click **Download CSV** of bounding box coordinates.
