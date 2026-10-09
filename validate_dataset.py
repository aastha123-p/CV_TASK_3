from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"dataset_450_yolo")

SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test"
}

CLASS_NAMES = {
    0: "Hole",
    1: "Leak",
    2: "rust"
}

NUM_CLASSES = 3

# Maximum allowed ratio between largest and smallest class
IMBALANCE_RATIO = 3.0


# ============================================================
# GET IMAGES
# ============================================================

def get_images(image_dir):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    return [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    ]


# ============================================================
# VALIDATE ONE LABEL FILE
# ============================================================

def validate_label_file(label_path, split):

    errors = 0
    class_counter = Counter()

    try:

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

    except Exception as e:

        print(
            f"❌ Cannot read label: "
            f"{label_path} | {e}"
        )

        return class_counter, 1

    # Empty label
    if not lines:

        print(
            f"⚠️ Empty label: "
            f"{split}/{label_path.name}"
        )

        return class_counter, 1

    # Check every annotation
    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        # YOLO format = 5 values
        if len(parts) != 5:

            print(
                f"❌ Invalid format: "
                f"{split}/{label_path.name} "
                f"line {line_number}"
            )

            errors += 1
            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            print(
                f"❌ Non-numeric value: "
                f"{split}/{label_path.name} "
                f"line {line_number}"
            )

            errors += 1
            continue

        # ----------------------------------------------------
        # CLASS ID CHECK
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            print(
                f"❌ Invalid class ID {class_id}: "
                f"{split}/{label_path.name} "
                f"line {line_number}"
            )

            errors += 1

        else:

            class_counter[class_id] += 1

        # ----------------------------------------------------
        # COORDINATE CHECK
        # ----------------------------------------------------

        coordinates = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        }

        for name, value in coordinates.items():

            if value < 0 or value > 1:

                print(
                    f"❌ Invalid {name}={value}: "
                    f"{split}/{label_path.name} "
                    f"line {line_number}"
                )

                errors += 1

        # Width / Height cannot be zero
        if width <= 0:

            print(
                f"❌ Invalid width={width}: "
                f"{split}/{label_path.name}"
            )

            errors += 1

        if height <= 0:

            print(
                f"❌ Invalid height={height}: "
                f"{split}/{label_path.name}"
            )

            errors += 1

    return class_counter, errors


# ============================================================
# VALIDATE SPLIT
# ============================================================

def validate_split(split_name, folder_name):

    image_dir = DATASET_ROOT / folder_name / "images"
    label_dir = DATASET_ROOT / folder_name / "labels"

    print("\n" + "=" * 60)
    print(f"CHECKING {split_name.upper()}")
    print("=" * 60)

    # Check directories
    if not image_dir.exists():

        print(f"❌ Image folder missing:")
        print(image_dir)

        return Counter(), 1, 0

    if not label_dir.exists():

        print(f"❌ Label folder missing:")
        print(label_dir)

        return Counter(), 1, 0

    images = get_images(image_dir)

    labels = list(label_dir.glob("*.txt"))

    print(f"Images found : {len(images)}")
    print(f"Labels found : {len(labels)}")

    total_errors = 0
    class_counter = Counter()

    image_stems = {
        image.stem
        for image in images
    }

    label_stems = {
        label.stem
        for label in labels
    }

    # --------------------------------------------------------
    # MISSING LABELS
    # --------------------------------------------------------

    missing_labels = image_stems - label_stems

    if missing_labels:

        print("\n❌ MISSING LABELS:")

        for name in sorted(missing_labels):

            print(f"   {name}")

        total_errors += len(missing_labels)

    else:

        print("✓ No missing labels")

    # --------------------------------------------------------
    # ORPHAN LABELS
    # --------------------------------------------------------

    orphan_labels = label_stems - image_stems

    if orphan_labels:

        print("\n⚠️ ORPHAN LABELS:")

        for name in sorted(orphan_labels):

            print(f"   {name}.txt")

        total_errors += len(orphan_labels)

    else:

        print("✓ No orphan labels")

    # --------------------------------------------------------
    # VALIDATE LABEL CONTENT
    # --------------------------------------------------------

    for label_path in labels:

        counts, errors = validate_label_file(
            label_path,
            split_name
        )

        class_counter.update(counts)
        total_errors += errors

    return class_counter, total_errors, len(images)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def check_class_distribution(counter):

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    total_objects = sum(counter.values())

    print(
        f"\nTotal annotated objects: "
        f"{total_objects}"
    )

    if total_objects == 0:

        print("❌ No annotations found.")

        return False

    for class_id, class_name in CLASS_NAMES.items():

        count = counter[class_id]

        percentage = (
            count / total_objects * 100
        )

        print(
            f"{class_id} - {class_name:<8} : "
            f"{count:>5} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # IMBALANCE CHECK
    # --------------------------------------------------------

    counts = [
        counter[class_id]
        for class_id in CLASS_NAMES
        if counter[class_id] > 0
    ]

    if len(counts) < NUM_CLASSES:

        print(
            "\n❌ At least one class "
            "has ZERO annotations."
        )

        return False

    minimum = min(counts)
    maximum = max(counts)

    ratio = maximum / minimum

    print(
        f"\nLargest / smallest class ratio: "
        f"{ratio:.2f}"
    )

    if ratio > IMBALANCE_RATIO:

        print(
            "⚠️ WARNING: Significant class imbalance."
        )

        return False

    print("✓ Class distribution looks reasonable.")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("        YOLO DATASET VALIDATOR")
    print("=" * 60)

    print(
        f"\nDataset:\n{DATASET_ROOT}"
    )

    if not DATASET_ROOT.exists():

        print("\n❌ DATASET ROOT NOT FOUND")

        print(
            "\nCheck this path:"
        )

        print(DATASET_ROOT)

        return

    total_counter = Counter()

    total_errors = 0
    total_images = 0

    split_counts = {}

    # --------------------------------------------------------
    # TRAIN / VALID / TEST
    # --------------------------------------------------------

    for split_name, folder_name in SPLITS.items():

        counter, errors, image_count = validate_split(
            split_name,
            folder_name
        )

        total_counter.update(counter)

        total_errors += errors

        total_images += image_count

        split_counts[split_name] = image_count

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"\nTrain images : "
        f"{split_counts['train']}"
    )

    print(
        f"Validation images : "
        f"{split_counts['val']}"
    )

    print(
        f"Test images : "
        f"{split_counts['test']}"
    )

    print(
        f"Total images : "
        f"{total_images}"
    )

    # --------------------------------------------------------
    # CLASS CHECK
    # --------------------------------------------------------

    class_balance_ok = check_class_distribution(
        total_counter
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if total_errors == 0 and class_balance_ok:

        print("\n✅ DATASET VALIDATION PASSED")

        print(
            "Your dataset is ready for YOLO training."
        )

    else:

        print("\n❌ DATASET VALIDATION FAILED")

        if total_errors > 0:

            print(
                f"\nErrors found: "
                f"{total_errors}"
            )

        if not class_balance_ok:

            print(
                "Class distribution needs attention."
            )

        print(
            "\nFix the issues and run "
            "the validator again."
        )

    print("\n" + "=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()