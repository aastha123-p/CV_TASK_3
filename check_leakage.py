from pathlib import Path
from PIL import Image
import imagehash

ROOT = Path(r"dataset_450_yolo")

splits = {
    "train": ROOT / "train/images",
    "val": ROOT / "valid/images",
    "test": ROOT / "test/images"
}

hashes = {}

for split, folder in splits.items():
    for img_path in folder.glob("*"):
        try:
            h = imagehash.phash(Image.open(img_path))
            hashes.setdefault(str(h), []).append((split, img_path.name))
        except:
            pass

leaks = []

for h, imgs in hashes.items():
    split_names = {x[0] for x in imgs}

    if len(split_names) > 1:
        leaks.append(imgs)

print("\n=== DATASET LEAKAGE CHECK ===")

if not leaks:
    print("✅ No exact/near-duplicate images found across splits.")
else:
    print(f"❌ Possible leakage groups: {len(leaks)}")

    for group in leaks:
        print("\nPossible duplicate:")
        for split, name in group:
            print(f"  {split}: {name}")