import os
import sys
import glob
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION & COLOR PALETTE
# ============================================================
DATASET_ROOT = Path("dataset_450_yolo")
DATA_YAML_PATH = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = Path("annotated_output")

# Palette for classes (RGB)
CLASS_COLORS = {
    0: {"name": "Hole", "color": (255, 69, 58),   "bg": (255, 69, 58, 60)},    # Vibrant Coral / Red
    1: {"name": "Leak", "color": (10, 132, 255),  "bg": (10, 132, 255, 60)},   # Vivid Azure / Blue
    2: {"name": "rust", "color": (255, 214, 10),  "bg": (255, 214, 10, 60)},   # Bright Yellow / Gold
}

DEFAULT_COLOR = (48, 209, 88)  # Bright Green

def load_data_yaml():
    """Load class names from data.yaml if available."""
    if DATA_YAML_PATH.exists():
        with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            names = data.get("names", [])
            if isinstance(names, dict):
                names = list(names.values())
            for idx, name in enumerate(names):
                if idx in CLASS_COLORS:
                    CLASS_COLORS[idx]["name"] = name
                else:
                    CLASS_COLORS[idx] = {
                        "name": name,
                        "color": (175, 82, 222),
                        "bg": (175, 82, 222, 60)
                    }

def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO label txt file into list of bounding box dicts."""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue

            # Convert normalized YOLO center format to pixel bounds
            xmin = max(0, int((x_center - bw / 2.0) * img_w))
            ymin = max(0, int((y_center - bh / 2.0) * img_h))
            xmax = min(img_w, int((x_center + bw / 2.0) * img_w))
            ymax = min(img_h, int((y_center + bh / 2.0) * img_h))

            box_w = xmax - xmin
            box_h = ymax - ymin

            boxes.append({
                "class_id": cls_id,
                "bbox": [xmin, ymin, xmax, ymax],
                "width": box_w,
                "height": box_h,
                "norm": [x_center, y_center, bw, bh]
            })
    return boxes

def draw_annotations(img, boxes):
    """Draw bounding boxes with transparent fills and high-legibility badges."""
    # Convert image to RGBA for semi-transparent overlay
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw_over = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base)

    img_w, img_h = base.size

    # Try loading default system font or scaling font size according to image dimensions
    font_size = max(14, int(min(img_w, img_h) * 0.035))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for box in boxes:
        cls_id = box["class_id"]
        xmin, ymin, xmax, ymax = box["bbox"]
        
        info = CLASS_COLORS.get(cls_id, {
            "name": f"Class {cls_id}",
            "color": DEFAULT_COLOR,
            "bg": (48, 209, 88, 60)
        })
        
        stroke_color = info["color"] + (255,)
        fill_color = info.get("bg", stroke_color[:3] + (50,))
        cls_name = info["name"]

        # Draw semi-transparent rectangle fill
        draw_over.rectangle([xmin, ymin, xmax, ymax], fill=fill_color, outline=stroke_color, width=3)

        # Label badge setup
        label_text = f" {cls_name} ({box['width']}x{box['height']}px) "
        
        # Calculate text bounding box
        try:
            bbox_text = font.getbbox(label_text)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_w, text_h = font.getsize(label_text)

        # Position badge above or inside bounding box
        badge_ymin = ymin - text_h - 6 if ymin - text_h - 6 > 0 else ymin
        badge_ymax = badge_ymin + text_h + 6
        badge_xmax = min(img_w, xmin + text_w + 4)

        # Draw text background badge
        draw_over.rectangle([xmin, badge_ymin, badge_xmax, badge_ymax], fill=stroke_color)
        
        # Draw white text
        draw_over.text((xmin + 2, badge_ymin + 2), label_text, fill=(255, 255, 255, 255), font=font)

    # Composite overlay onto base image
    annotated = Image.alpha_composite(base, overlay)
    return annotated.convert("RGB")

def create_html_dashboard(metadata):
    """Generate an interactive HTML gallery dashboard to inspect annotated images."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annotated Bounding Boxes Inspector</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #38bdf8;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: #334155;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 24px;
            min-height: 100vh;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        h1 {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            background: #0284c7;
            color: white;
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
        }}
        .controls {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            background: var(--card-bg);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        label {{
            font-size: 0.85rem;
            color: var(--text-sub);
            font-weight: 600;
        }}
        select, input {{
            background: #0f172a;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
        }}
        select:focus, input:focus {{
            border-color: var(--accent-color);
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            border-color: var(--accent-color);
        }}
        .card-img-wrap {{
            position: relative;
            aspect-ratio: 4 / 3;
            background: #020617;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .card-img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: zoom-in;
        }}
        .card-info {{
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .card-title {{
            font-size: 0.9rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--text-main);
        }}
        .card-tags {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        .tag {{
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
        }}
        .tag-Hole {{ background: rgba(255, 69, 58, 0.2); color: #ff453a; border: 1px solid #ff453a; }}
        .tag-Leak {{ background: rgba(10, 132, 255, 0.2); color: #0a84ff; border: 1px solid #0a84ff; }}
        .tag-rust {{ background: rgba(255, 214, 10, 0.2); color: #ffd60a; border: 1px solid #ffd60a; }}
        .tag-split {{ background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid var(--border-color); }}

        /* Modal lightbox */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.9);
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .modal-content {{
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 8px;
            border: 2px solid var(--border-color);
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🎯 Annotated Bounding Boxes Inspector <span class="badge">{len(metadata)} Images</span></h1>
    </header>

    <div class="controls">
        <div class="control-group">
            <label for="splitFilter">Dataset Split</label>
            <select id="splitFilter" onchange="filterGallery()">
                <option value="all">All Splits</option>
                <option value="train">Train</option>
                <option value="valid">Valid</option>
                <option value="test">Test</option>
            </select>
        </div>

        <div class="control-group">
            <label for="classFilter">Defect Class</label>
            <select id="classFilter" onchange="filterGallery()">
                <option value="all">All Classes</option>
                <option value="Hole">Hole</option>
                <option value="Leak">Leak</option>
                <option value="rust">rust</option>
            </select>
        </div>

        <div class="control-group">
            <label for="searchInput">Search Filename</label>
            <input type="text" id="searchInput" placeholder="Type filename..." onkeyup="filterGallery()">
        </div>
    </div>

    <div class="gallery" id="galleryGrid">
"""
    for item in metadata:
        classes_str = " ".join([f"class-{cls}" for cls in item["classes"]])
        tags_html = "".join([f'<span class="tag tag-{cls}">{cls}</span>' for cls in item["classes"]])
        if not item["classes"]:
            tags_html = '<span class="tag tag-split">No Annotations</span>'

        rel_path = item["rel_path"].replace("\\", "/")
        html_content += f"""
        <div class="card" data-split="{item['split']}" data-classes="{classes_str}" data-name="{item['filename'].lower()}">
            <div class="card-img-wrap">
                <img src="{rel_path}" alt="{item['filename']}" onclick="openModal(this.src)" loading="lazy">
            </div>
            <div class="card-info">
                <div class="card-title" title="{item['filename']}">{item['filename']}</div>
                <div class="card-tags">
                    <span class="tag tag-split">{item['split'].upper()}</span>
                    {tags_html}
                </div>
                <div style="font-size: 0.75rem; color: var(--text-sub);">Boxes: {item['box_count']} | Size: {item['width']}x{item['height']}px</div>
            </div>
        </div>
"""

    html_content += """
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>

    <script>
        function filterGallery() {
            const splitVal = document.getElementById('splitFilter').value;
            const classVal = document.getElementById('classFilter').value;
            const searchVal = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.card');

            cards.forEach(card => {
                const matchSplit = (splitVal === 'all' || card.getAttribute('data-split') === splitVal);
                const matchClass = (classVal === 'all' || card.getAttribute('data-classes').includes('class-' + classVal));
                const matchSearch = card.getAttribute('data-name').includes(searchVal);

                if (matchSplit && matchClass && matchSearch) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImg');
            modal.style.display = 'flex';
            modalImg.src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
    </script>
</body>
</html>
"""
    with open(OUTPUT_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    load_data_yaml()
    print("Class mapping:", {k: v["name"] for k, v in CLASS_COLORS.items()})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": DATASET_ROOT / "train",
        "valid": DATASET_ROOT / "valid",
        "test": DATASET_ROOT / "test",
    }

    metadata = []

    total_images_processed = 0
    total_boxes_drawn = 0

    for split_name, split_path in splits.items():
        img_dir = split_path / "images"
        lbl_dir = split_path / "labels"

        out_split_dir = OUTPUT_DIR / split_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"Skipping missing directory: {img_dir}")
            continue

        image_files = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        print(f"Processing split '{split_name}': found {len(image_files)} images...")

        for img_path in image_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    boxes = parse_yolo_label(lbl_path, img_w, img_h)
                    annotated_img = draw_annotations(img, boxes)

                    out_path = out_split_dir / img_path.name
                    annotated_img.save(out_path, quality=95)

                    classes_in_image = sorted(list(set(CLASS_COLORS.get(b["class_id"], {}).get("name", str(b["class_id"])) for b in boxes)))

                    metadata.append({
                        "filename": img_path.name,
                        "split": split_name,
                        "rel_path": f"{split_name}/{img_path.name}",
                        "width": img_w,
                        "height": img_h,
                        "box_count": len(boxes),
                        "classes": classes_in_image
                    })

                    total_images_processed += 1
                    total_boxes_drawn += len(boxes)

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

    create_html_dashboard(metadata)

    print("\n" + "="*60)
    print("SUCCESS: COMPLETED VISUALIZATION!")
    print(f"Total images annotated: {total_images_processed}")
    print(f"Total bounding boxes drawn: {total_boxes_drawn}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"HTML Inspector Gallery: {OUTPUT_DIR.resolve() / 'index.html'}")
    print("="*60)

if __name__ == "__main__":
    main()
