from pathlib import Path
import random
import shutil

SOURCE = Path("datasets/audio_devices/yolo_merged")
OUTPUT = Path("datasets/audio_devices/yolo_dataset")

CLASSES = {
    0: "earbud",
    1: "wired_earphone",
    2: "headphone",
    3: "headset",
    4: "cell_phone",
}

random.seed(42)

# Create output folders
for split in ["train", "val", "test"]:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# Group images by their class ID
class_files = {class_id: [] for class_id in CLASSES}

for label_path in (SOURCE / "labels").glob("*.txt"):
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if not lines:
        continue

    ids = {int(line.split()[0]) for line in lines}

    if len(ids) != 1:
        print(f"Skipping mixed-class file: {label_path.name}")
        continue

    class_id = ids.pop()

    if class_id not in CLASSES:
        print(f"Unknown class ID: {label_path.name}")
        continue

    image_path = None

    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = SOURCE / "images" / (label_path.stem + ext)
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"Missing image: {label_path.name}")
        continue

    class_files[class_id].append((image_path, label_path))

print("\n========== DATASET SPLIT ==========")

totals = {"train": 0, "val": 0, "test": 0}

for class_id, class_name in CLASSES.items():
    files = class_files[class_id]
    random.shuffle(files)

    n = len(files)
    train_n = int(n * 0.70)
    val_n = int(n * 0.20)
    test_n = n - train_n - val_n

    splits = {
        "train": files[:train_n],
        "val": files[train_n:train_n + val_n],
        "test": files[train_n + val_n:],
    }

    print(f"\n{class_name}: {n} images")

    for split, items in splits.items():
        for image_path, label_path in items:
            shutil.copy2(image_path, OUTPUT / "images" / split / image_path.name)
            shutil.copy2(label_path, OUTPUT / "labels" / split / label_path.name)

        totals[split] += len(items)
        print(f"  {split}: {len(items)}")

# Write YOLO configuration
yaml_text = f"""path: {OUTPUT.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: earbud
  1: wired_earphone
  2: headphone
  3: headset
  4: cell_phone
"""

(OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

print("\n========== FINAL TOTALS ==========")
print(f"Train: {totals['train']}")
print(f"Val:   {totals['val']}")
print(f"Test:  {totals['test']}")
print(f"Total: {sum(totals.values())}")
print(f"\nDataset created at: {OUTPUT.resolve()}")
print("data.yaml created successfully.")
