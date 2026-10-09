from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

SCRIPT_FOLDER = Path(__file__).resolve().parent

results = model.train(
    data=str(SCRIPT_FOLDER / "dataset_450_yolo" / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",
    project=str(SCRIPT_FOLDER / "runs"),
    name="yolov8n_baseline",
    seed=42
)