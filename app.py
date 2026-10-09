import os
import cv2
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# Page setup
st.set_page_config(
    page_title="Pipeline Defect Detection Demo",
    page_icon="🔍",
    layout="wide"
)

SCRIPT_FOLDER = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_FOLDER / "runs" / "yolov8n_improved" / "weights" / "best.pt"
ONNX_MODEL = SCRIPT_FOLDER / "runs" / "yolov8n_improved" / "weights" / "best.onnx"

CLASS_NAMES = {0: "Hole", 1: "Leak", 2: "rust"}
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 165, 255)} # Red for Hole, Blue for Leak, Orange for Rust

# App Header
st.title("🔍 Pipeline Defect Detection System")
st.markdown("Upload an image or video of industrial pipelines to detect **Hole**, **Leak**, and **rust** defects in real-time.")

# Sidebar Controls
st.sidebar.header("⚙️ Model & Detection Settings")

model_choice = st.sidebar.selectbox(
    "Select Model Format",
    ["PyTorch (.pt)", "ONNX Runtime (.onnx)"]
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU NMS Threshold", 0.10, 1.00, 0.45, 0.05)

# Load YOLO Model
@st.cache_resource
def load_yolo_model(model_path_str):
    return YOLO(model_path_str)

model_path = ONNX_MODEL if (model_choice == "ONNX Runtime (.onnx)" and ONNX_MODEL.exists()) else DEFAULT_MODEL

if not model_path.exists():
    st.error(f"Model file not found at {model_path}. Please ensure training or ONNX export is complete.")
    st.stop()

model = load_yolo_model(str(model_path))

# Media File Uploader
uploaded_file = st.file_uploader(
    "Upload Pipeline Image or Video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    is_video = file_type in ["mp4", "avi", "mov"]

    if not is_video:
        # IMAGE PROCESSING
        st.subheader("📷 Image Defect Detection Results")
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        # Run Prediction
        results = model.predict(source=img_np, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        
        annotated_img = results.plot() # BGR numpy array
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Image", use_container_width=True)
        with col2:
            st.image(annotated_rgb, caption="Annotated Output Image", use_container_width=True)

        # Collect Detections Table
        detections = []
        for box in results.boxes:
            c_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
            
            detections.append({
                "Filename": uploaded_file.name,
                "Defect Class": CLASS_NAMES.get(c_id, f"Class {c_id}"),
                "Confidence Score": round(conf, 4),
                "Xmin": round(xyxy[0], 1),
                "Ymin": round(xyxy[1], 1),
                "Xmax": round(xyxy[2], 1),
                "Ymax": round(xyxy[3], 1)
            })

        df_detections = pd.DataFrame(detections)

        st.subheader("📋 Detected Defects Data")
        if not df_detections.empty:
            st.dataframe(df_detections, use_container_width=True)
            
            csv_data = df_detections.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detections CSV",
                data=csv_data,
                file_name=f"defect_detections_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
        else:
            st.info("No defects detected at current confidence threshold.")

    else:
        # VIDEO PROCESSING
        st.subheader("📹 Video Defect Detection Results")
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

        out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_tfile.name, fourcc, fps, (width, height))

        video_detections = []
        frame_idx = 0
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            annotated_frame = results.plot()
            out_writer.write(annotated_frame)

            for box in results.boxes:
                c_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                video_detections.append({
                    "Video Name": uploaded_file.name,
                    "Frame": frame_idx,
                    "Defect Class": CLASS_NAMES.get(c_id, f"Class {c_id}"),
                    "Confidence Score": round(conf, 4),
                    "Xmin": round(xyxy[0], 1),
                    "Ymin": round(xyxy[1], 1),
                    "Xmax": round(xyxy[2], 1),
                    "Ymax": round(xyxy[3], 1)
                })
            
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out_writer.release()

        st.success("Video processing complete!")
        st.video(out_tfile.name)

        df_vid_detections = pd.DataFrame(video_detections)
        if not df_vid_detections.empty:
            st.dataframe(df_vid_detections, use_container_width=True)
            csv_vid_data = df_vid_detections.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Video Detections CSV",
                data=csv_vid_data,
                file_name=f"video_detections_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
else:
    st.info("👆 Please upload an image or video file above to start detection.")
