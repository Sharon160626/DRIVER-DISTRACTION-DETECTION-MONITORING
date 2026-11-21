from pathlib import Path
import shutil

BASE = Path("annotation_exports")
OUT = Path("datasets/audio_devices/yolo_merged")

CLASS_MAP = {
    "earbud": 0,
    "wired_earphone": 1,
    "headphone": 2,
    "headset": 3,
    "cell_phone": 4,
}

image_exts = {".jpg", ".jpeg", ".png"}
OUT_IMAGES = OUT / "images"
OUT_LABELS = OUT / "labels"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

total_images = 0
total_boxes = 0
errors = []

print("\n========== YOLO MERGE START ==========\n")

for folder, new_class_id in CLASS_MAP.items():
    source = BASE / folder
    print(f"\nProcessing: {folder} -> class ID {new_class_id}")

    images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    class_images = 0
    class_boxes = 0

    for image_path in images:
        label_path = image_path.with_suffix(".txt")

        if not label_path.exists():
            errors.append(f"MISSING LABEL: {image_path.name}")
            continue

        new_name = f"{folder}_{image_path.name}"
        new_image = OUT_IMAGES / new_name
        new_label = OUT_LABELS / f"{Path(new_name).stem}.txt"

        corrected_lines = []

        for line_num, line in enumerate(label_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                errors.append(f"INVALID FORMAT: {label_path.name}, line {line_num}")
                continue

            try:
                old_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"NON-NUMERIC LABEL: {label_path.name}, line {line_num}")
                continue

            if old_id != 0:
                errors.append(f"UNEXPECTED CLASS ID: {label_path.name}, line {line_num}: {old_id}")
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                errors.append(f"OUT-OF-RANGE BOX: {label_path.name}, line {line_num}")
                continue

            if x - w/2 < -0.001 or x + w/2 > 1.001 or y - h/2 < -0.001 or y + h/2 > 1.001:
                errors.append(f"BOX OUTSIDE IMAGE: {label_path.name}, line {line_num}")
                continue

            corrected_lines.append(f"{new_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            class_boxes += 1

        shutil.copy2(image_path, new_image)
        new_label.write_text("\n".join(corrected_lines) + ("\n" if corrected_lines else ""))

        class_images += 1
        total_images += 1

    total_boxes += class_boxes
    print(f"Images copied: {class_images}")
    print(f"Boxes remapped: {class_boxes}")

print("\n========== MERGE SUMMARY ==========")
print(f"Total images copied: {total_images}")
print(f"Total boxes: {total_boxes}")
print(f"Output folder: {OUT.resolve()}")

if errors:
    print(f"\nWARNING: {len(errors)} problems found:")
    for error in errors:
        print(" -", error)
else:
    print("\nNo annotation errors detected.")

print("\nMerge finished.")
