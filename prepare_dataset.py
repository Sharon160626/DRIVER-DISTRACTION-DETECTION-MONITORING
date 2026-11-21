
# ==========================================
# DRIVER BEHAVIOR DATASET PREPARATION
# YOLO annotations -> Image Classification
# ==========================================

import re
import shutil
from pathlib import Path

import pandas as pd

from config import (
    DRIVER_RISK_DIR,
    PROCESSED_DIR,
    CLASS_NAMES,
    YOLO_CLASS_MAPPING,
)

# ---------- Recording-wise split ----------
# All images from one recording remain in the same split.
RECORDING_SPLIT = {
    "ce8291e3-2691-44cd-ac9f-e0558d819eba": "train",
    "d37c44c3-53f6-4f22-b4db-0ceb0c55d1a8": "train",
    "fcb311b4-a8ec-4cfa-98f0-5b89d7426ba9": "train",
    "f78505a2-5c98-4a51-a449-b632ff620b0a": "val",
    "d9d40c4d-9689-4683-a284-7fc28afa342a": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------- Output directory ----------
OUTPUT_DIR = PROCESSED_DIR / "driver_risk_behavior"


def get_recording_id(filename):
    """
    Extract recording UUID from a filename.

    Example:
    ce8291e3-2691-44cd-ac9f-e0558d819eba_105.jpg
    """
    match = re.match(r"^(.+)_\d+$", filename.stem)

    if match:
        return match.group(1)

    return None


def parse_annotation(annotation_path):
    """
    Read YOLO annotation file.

    YOLO format:
    class_id x_center y_center width height

    Returns:
        class_ids: set of unique valid class IDs
        invalid: whether invalid annotation lines were found
    """
    class_ids = set()
    invalid = False

    try:
        lines = annotation_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except Exception:
        return set(), True

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # Each YOLO annotation must contain 5 values.
        if len(parts) != 5:
            invalid = True
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            invalid = True
            continue

        # Verify class ID
        if class_id not in YOLO_CLASS_MAPPING:
            invalid = True
            continue

        # Verify bounding box values are normalized.
        if any(value < 0 or value > 1 for value in values):
            invalid = True
            continue

        # Width and height must be positive.
        if values[2] <= 0 or values[3] <= 0:
            invalid = True
            continue

        class_ids.add(class_id)

    # Return even if class_ids is empty.
    # Empty files will be counted separately.
    return class_ids, invalid


def main():

    print("=" * 65)
    print("DRIVER BEHAVIOR DATASET PREPARATION")
    print("=" * 65)

    # ---------- Verify dataset path ----------
    if not DRIVER_RISK_DIR.exists():
        print("\nDataset directory not found:")
        print(DRIVER_RISK_DIR)
        return

    # ---------- Verify classes.txt ----------
    classes_file = DRIVER_RISK_DIR / "classes.txt"

    if not classes_file.exists():
        print("\nclasses.txt not found.")
        return

    dataset_classes = [
        line.strip()
        for line in classes_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if dataset_classes != CLASS_NAMES:
        raise ValueError(
            "\nDataset classes do not match config.py!\n"
            f"Dataset classes: {dataset_classes}\n"
            f"Config classes:  {CLASS_NAMES}"
        )

    print("\nClasses verified:", dataset_classes)

    # ---------- Collect image files ----------
    images = sorted(
        path
        for path in DRIVER_RISK_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("Total images found:", len(images))

    # ---------- Annotation audit ----------
    records = []
    pure_records = []

    mixed_count = 0
    empty_count = 0
    invalid_count = 0
    missing_annotation_count = 0
    unknown_group_count = 0

    for image_path in images:

        annotation_path = image_path.with_suffix(".txt")
        recording_id = get_recording_id(image_path)

        if recording_id is None:
            unknown_group_count += 1

        # Missing annotation file
        if not annotation_path.exists():

            missing_annotation_count += 1

            records.append({
                "image": image_path.name,
                "recording_id": recording_id,
                "classes": "",
                "class_name": "",
                "status": "missing_annotation",
                "split": "",
                "source_path": str(image_path),
            })

            continue

        # Read annotation
        class_ids, has_invalid = parse_annotation(
            annotation_path
        )

        # Invalid annotation lines
        if has_invalid:

            invalid_count += 1

            records.append({
                "image": image_path.name,
                "recording_id": recording_id,
                "classes": ",".join(
                    str(class_id)
                    for class_id in sorted(class_ids)
                ),
                "class_name": "",
                "status": "invalid_annotation",
                "split": "",
                "source_path": str(image_path),
            })

            continue

        # Empty annotation
        if len(class_ids) == 0:

            empty_count += 1

            records.append({
                "image": image_path.name,
                "recording_id": recording_id,
                "classes": "",
                "class_name": "",
                "status": "empty_annotation",
                "split": "",
                "source_path": str(image_path),
            })

            continue

        # Convert IDs to class names
        class_names = [
            YOLO_CLASS_MAPPING[class_id]
            for class_id in sorted(class_ids)
        ]

        # Mixed-class image: exclude from initial single-label model
        if len(class_ids) > 1:

            mixed_count += 1

            records.append({
                "image": image_path.name,
                "recording_id": recording_id,
                "classes": ",".join(class_names),
                "class_name": "",
                "status": "mixed",
                "split": "",
                "source_path": str(image_path),
            })

            continue

        # Pure single-class image
        class_id = next(iter(class_ids))
        class_name = YOLO_CLASS_MAPPING[class_id]

        # Check recording split
        if recording_id not in RECORDING_SPLIT:

            records.append({
                "image": image_path.name,
                "recording_id": recording_id,
                "classes": class_name,
                "class_name": class_name,
                "status": "unknown_recording",
                "split": "",
                "source_path": str(image_path),
            })

            continue

        split_name = RECORDING_SPLIT[recording_id]

        record = {
            "image": image_path.name,
            "recording_id": recording_id,
            "classes": class_name,
            "class_name": class_name,
            "status": "pure",
            "split": split_name,
            "source_path": str(image_path),
        }

        pure_records.append(record)
        records.append(record)

    # ---------- Verify split configuration ----------
    used_groups = {
        record["recording_id"]
        for record in pure_records
    }

    missing_group_mapping = (
        used_groups - set(RECORDING_SPLIT.keys())
    )

    if missing_group_mapping:
        raise ValueError(
            "Recording groups missing from split configuration: "
            f"{missing_group_mapping}"
        )

    # ---------- Print audit results ----------
    print("\n" + "=" * 65)
    print("ANNOTATION AUDIT")
    print("=" * 65)
    print("Pure images:", len(pure_records))
    print("Mixed images:", mixed_count)
    print("Missing annotation files:", missing_annotation_count)
    print("Invalid annotations:", invalid_count)
    print("Empty annotations:", empty_count)
    print("Images with unrecognized recording names:", unknown_group_count)

    # ---------- Build DataFrame ----------
    pure_df = pd.DataFrame(pure_records)

    if pure_df.empty:
        print("\nNo valid pure images found. Exiting.")
        return

    # ---------- Recording split summary ----------
    print("\n" + "=" * 65)
    print("RECORDING SPLIT")
    print("=" * 65)

    for recording_id, split_name in RECORDING_SPLIT.items():

        group_count = sum(
            record["recording_id"] == recording_id
            for record in pure_records
        )

        print(
            f"{recording_id} -> {split_name}: "
            f"{group_count} pure images"
        )

    # ---------- Prepare output folder ----------
    # Remove only the generated driver_risk_behavior folder.
    # Original raw images and annotations are untouched.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # ---------- Copy pure images ----------
    copied_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    class_counts = {
        split_name: {
            class_name: 0
            for class_name in CLASS_NAMES
        }
        for split_name in copied_counts
    }

    missing_images = []

    for record in pure_records:

        source_image = Path(record["source_path"])
        split_name = record["split"]
        class_name = record["class_name"]

        destination_folder = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_image = (
            destination_folder
            / source_image.name
        )

        # Check source exists before copying.
        if not source_image.exists():

            missing_images.append(str(source_image))
            continue

        shutil.copy2(
            source_image,
            destination_image
        )

        copied_counts[split_name] += 1
        class_counts[split_name][class_name] += 1

    # ---------- Save manifest ----------
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"

    pd.DataFrame(records).to_csv(
        manifest_path,
        index=False
    )

    # ---------- Save class-count report ----------
    report_rows = []

    for split_name in ["train", "val", "test"]:

        for class_name in CLASS_NAMES:

            report_rows.append({
                "split": split_name,
                "class_name": class_name,
                "image_count": class_counts[split_name][class_name],
            })

    report_path = OUTPUT_DIR / "class_counts.csv"

    pd.DataFrame(report_rows).to_csv(
        report_path,
        index=False
    )

    # ---------- Save missing images report ----------
    if missing_images:

        missing_path = OUTPUT_DIR / "missing_images.txt"

        missing_path.write_text(
            "\n".join(missing_images),
            encoding="utf-8"
        )

    # ---------- Final summary ----------
    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 65)

    print("Train images:", copied_counts["train"])
    print("Validation images:", copied_counts["val"])
    print("Test images:", copied_counts["test"])
    print("Total copied:", sum(copied_counts.values()))
    print("Missing source images:", len(missing_images))

    print("\nClass counts by split:")

    for split_name in ["train", "val", "test"]:

        print(f"\n{split_name.upper()}")

        for class_name in CLASS_NAMES:

            print(
                f"{class_name:20s}: "
                f"{class_counts[split_name][class_name]}"
            )

    print("\nOutput folder:")
    print(OUTPUT_DIR)

    print("\nManifest file:")
    print(manifest_path)

    print("\nClass count report:")
    print(report_path)


if __name__ == "__main__":
    main()
