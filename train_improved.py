import shutil
from pathlib import Path
from ultralytics import YOLO

SCRIPT_FOLDER = Path(__file__).resolve().parent
SRC_DATA_DIR = SCRIPT_FOLDER / "dataset_450_yolo"
IMPROVED_DATA_DIR = SCRIPT_FOLDER / "dataset_450_yolo_improved"

print("--- Step 1: Creating Oversampled Dataset Structure ---")

# Setup improved dataset directory
if IMPROVED_DATA_DIR.exists():
    shutil.rmtree(IMPROVED_DATA_DIR)

# Copy validation and test sets as-is
shutil.copytree(SRC_DATA_DIR / "valid", IMPROVED_DATA_DIR / "valid")
shutil.copytree(SRC_DATA_DIR / "test", IMPROVED_DATA_DIR / "test")

# Create train directories
(IMPROVED_DATA_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
(IMPROVED_DATA_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)

# Copy all original train files
train_img_dir = SRC_DATA_DIR / "train" / "images"
train_lbl_dir = SRC_DATA_DIR / "train" / "labels"

leak_count = 0
total_count = 0

for img_file in train_img_dir.glob("*"):
    lbl_file = train_lbl_dir / f"{img_file.stem}.txt"
    total_count += 1
    
    # Copy original
    shutil.copy(img_file, IMPROVED_DATA_DIR / "train" / "images" / img_file.name)
    if lbl_file.exists():
        shutil.copy(lbl_file, IMPROVED_DATA_DIR / "train" / "labels" / lbl_file.name)
    
    # Check if image contains class 1 (Leak)
    has_leak = False
    if lbl_file.exists():
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == 1:
                    has_leak = True
                    break
    
    # Oversample (duplicate) images containing Leak
    if has_leak:
        leak_count += 1
        dup_img_name = f"dup_leak_{img_file.name}"
        dup_lbl_name = f"dup_leak_{lbl_file.name}"
        shutil.copy(img_file, IMPROVED_DATA_DIR / "train" / "images" / dup_img_name)
        if lbl_file.exists():
            shutil.copy(lbl_file, IMPROVED_DATA_DIR / "train" / "labels" / dup_lbl_name)

print(f"Original train images: {total_count}")
print(f"Leak images oversampled: {leak_count}")
print(f"Total oversampled train images: {total_count + leak_count}")

# Create data_improved.yaml
yaml_content = f"""train: {IMPROVED_DATA_DIR.as_posix()}/train/images
val: {IMPROVED_DATA_DIR.as_posix()}/valid/images
test: {IMPROVED_DATA_DIR.as_posix()}/test/images

nc: 3
names: ['Hole', 'Leak', 'rust']
"""

yaml_path = IMPROVED_DATA_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Saved dataset configuration to {yaml_path}")

print("\n--- Step 2: Training Improved YOLOv8 Baseline Model ---")
print("Applying Targeted Augmentations, Loss Function Weighting (cls=1.5), and Oversampling...")

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",
    project=str(SCRIPT_FOLDER / "runs"),
    name="yolov8n_improved",
    seed=42,
    # Loss Function Weighting
    cls=1.5,           # Increased classification loss weight to prioritize class accuracy (targeting Leak recall drop)
    box=7.5,           # Standard box loss weight
    dfl=1.5,           # Standard DFL loss weight
    # Targeted Augmentations (Color, Sheen, Glare & Spatial)
    hsv_h=0.02,        # Fractional HSV-Hue gain
    hsv_s=0.8,         # Fractional HSV-Saturation gain (decouples liquid sheen/stains from background)
    hsv_v=0.6,         # Fractional HSV-Value gain (handles specular glare & liquid reflection)
    degrees=15.0,      # Image rotation (-15 to +15 deg)
    translate=0.1,     # Image translation
    scale=0.5,         # Image scale jitter
    shear=2.0,         # Image shear
    perspective=0.0005,# Perspective jitter
    flipud=0.5,        # Vertical flip
    fliplr=0.5,        # Horizontal flip
    mosaic=1.0,        # Mosaic augmentation
    mixup=0.15,        # Mixup augmentation (blends diffuse liquid edges with background)
    copy_paste=0.3     # Copy-paste augmentation
)

print("Training script execution complete.")
