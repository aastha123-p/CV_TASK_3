# Pipeline Defect Detection: Bounding Box Annotation Guidelines

## 1. Overview & Purpose
This document defines the standard operating procedures (SOP) and annotation standards for labeling pipeline defect datasets used in computer vision object detection models (e.g., YOLOv8, YOLOv11, Faster R-CNN). 

Accurate, consistent annotations are critical for ensuring reliable defect detection in industrial pipeline inspection pipelines (ultrasonic, visual RGB, endoscopic, or crawler camera systems).

---

## 2. Dataset Class Schema

| Class ID | Class Name | Description | Key Visual Characteristics |
| :---: | :---: | :--- | :--- |
| **`0`** | **`Hole`** | Physical punctures, perforations, deep pitting, or structural openings through/on the pipe wall. | Dark/shadowed circular or irregular cavity, distinct void edges, loss of metal thickness. |
| **`1`** | **`Leak`** | Fluid or gas emission, wet liquid trails, moisture stains, droplet accumulation, active spray/seepage. | Shiny/reflective wet surface, liquid stain perimeter, residue rings, fluid pooling or flow streaks. |
| **`2`** | **`rust`** | Surface oxidation, iron oxide scale, reddish-brown corrosion patches, metal flaking. | Discolored reddish/brownish/orange rough texture, paint peeling due to oxidation underneath. |

---

## 3. Bounding Box Labeling Principles

### 3.1. General Rules
1. **Tight Fit Requirement**: Bounding boxes must enclose the outermost pixels of the defect tightly. No unnecessary background padding (>5%) should be included inside the box.
2. **Orthogonal Rectangles**: All bounding boxes must be axis-aligned rectangles `(x_min, y_min, x_max, y_max)`.
3. **Minimum Resolution**: Do not label micro-imperfections smaller than **10x10 pixels** unless specifically instructed for micro-defect inspection modes.

---

### 3.2. Class-Specific Rules

#### A. Class `0`: Hole
- **Scope**: Label the entire void area of the puncture or structural cavity.
- **Surrounding Corrosion**: If a hole is inside a larger rust patch, label the **`Hole`** tightly around the opening, and label the surrounding rust area separately as **`rust`** (see Co-occurrence Section 4).
- **Cracks/Slits**: Narrow fracture openings leading to structural void are labeled as `Hole`.

#### B. Class `1`: Leak
- **Scope**: Enclose the full visible extent of the fluid/moisture output, including wet streaks, liquid droplets, or active fluid spray.
- **Dry Stains**: If a dried-up stain clearly indicates past leakage with visible residue rings, annotate as `Leak` if liquid origin is evident.
- **Reflections/Glare**: Do NOT confuse clean water/oil reflections on undamaged metallic surfaces with a `Leak`. Ensure texture disruption or fluid output point is visible.

#### C. Class `2`: rust
- **Contiguous Patches**: If multiple small rust spots are connected or separated by less than 15 pixels, group them into a single larger bounding box for `rust`.
- **Isolated Rust Spots**: Annotate distinct, separate rust patches (>20px apart) with individual bounding boxes.
- **Paint Peeling vs Corrosion**: Only label area where metal oxidation (red/brown/yellow scales) is visible.

---

## 4. Edge Cases & Co-occurrence Matrix

```
┌────────────────────────────────────────────────────────┐
│                   PIPELINE DEFECT                      │
│                                                        │
│   ┌──────────────────────────────────────────────┐     │
│   │ rust (Class 2)                               │     │
│   │   ┌────────────────────────┐                 │     │
│   │   │ Hole (Class 0)         │                 │     │
│   │   └────────────────────────┘                 │     │
│   └──────────────────────────────────────────────┘     │
│                                                        │
│   ┌──────────────────────────────────────────────┐     │
│   │ Leak (Class 1)                               │     │
│   └──────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────┘
```

| Edge Case Scenario | Correct Labeling Strategy |
| :--- | :--- |
| **Hole inside a Rust Patch** | Draw a tight bounding box around the **`Hole`** (Class 0). Draw a separate larger bounding box enclosing the surrounding **`rust`** area (Class 2). Both boxes will overlap. |
| **Leak originating from a Hole** | Label the physical puncture cavity as **`Hole`** (Class 0). Label the liquid stream/wet patch extending from it as **`Leak`** (Class 1). |
| **Wet Rust (Rust + Leak)** | Label the oxidation zone as **`rust`** (Class 2). Label the fluid flow/wetness boundaries as **`Leak`** (Class 1). |
| **Truncated Defect at Frame Edge** | Annotate the visible portion of the defect if at least **20% of the defect** is visible in the frame. |
| **Partially Occluded Defect** | Extend the bounding box logically across light occlusion (e.g. wire, small pipe bracket) if the defect is clearly continuous. |

---

## 5. Label Format & File Structure Specs

### 5.1. YOLO Annotation Format (`.txt`)
Each image must have a corresponding `.txt` file with identical filename root in normalized coordinates:

$$\text{Format: } \langle\text{class\_id}\rangle\ \langle\text{x\_center}\rangle\ \langle\text{y\_center}\rangle\ \langle\text{width}\rangle\ \langle\text{height}\rangle$$

- All values are normalized between `0.0` and `1.0` relative to image width and height.

#### Example Annotation File (`pipe_sample_001.txt`):
```text
0 0.4852 0.3120 0.1240 0.0980
2 0.4910 0.3540 0.3100 0.2850
1 0.7120 0.6210 0.2150 0.4100
```

---

## 6. Quality Control (Do's and Don'ts)

| Do's ✅ | Don'ts ❌ |
| :--- | :--- |
| ✅ Keep bounding boxes tight to defect boundaries. | ❌ Do not leave large padding of undamaged pipe surface inside the box. |
| ✅ Check all 3 classes systematically for every image. | ❌ Do not miss small holes inside large rust zones. |
| ✅ Ensure correct class index assignment (`0=Hole`, `1=Leak`, `2=rust`). | ❌ Do not confuse pipe joints, welds, or shadows with defects. |
| ✅ Maintain consistent bounding rules across annotators. | ❌ Do not label reflections, dust, or background machinery as defects. |

---

## 7. QA Checklist for Reviewers
- [ ] Are class IDs mapped correctly (`0: Hole`, `1: Leak`, `2: rust`)?
- [ ] Are bounding boxes tightly bounding without excessive background?
- [ ] Are co-occurring defects (e.g., Hole inside Rust) annotated with overlapping boxes?
- [ ] Have non-defect artifacts (pipe welds, shadows, reflections) been excluded?
- [ ] Are normalized YOLO coordinates verified to be in range `[0.0, 1.0]`?
