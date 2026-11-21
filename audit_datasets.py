
from pathlib import Path
import csv

from config import RAW_DIR

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def audit_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    print("\n" + "=" * 65)
    print(f"DATASET: {dataset_path.name}")
    print("=" * 65)

    if not dataset_path.exists():
        print("Folder does not exist yet.")
        return

    # Count all images recursively
    images = [
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("Total images:", len(images))

    # Count images by their immediate parent directory
    folder_counts = {}

    for image in images:
        folder = image.parent.name
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    print("\nImage counts by folder:")

    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder:30} {count:8}")

    # Find CSV files and inspect headers
    csv_files = list(dataset_path.rglob("*.csv"))

    print("\nCSV files found:", len(csv_files))

    for csv_file in csv_files:
        print("\nCSV:", csv_file.relative_to(dataset_path))

        try:
            with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)

            print("  Columns:", header)

        except Exception as error:
            print("  Could not read CSV:", error)

    # File size
    total_size = sum(p.stat().st_size for p in images)
    size_gb = total_size / (1024 ** 3)

    print(f"\nImage storage size: {size_gb:.2f} GB")


def main():
    print("MULTI-DATASET AUDIT TOOL")
    print("Raw datasets location:", RAW_DIR)

    if not RAW_DIR.exists():
        print("Raw dataset folder is missing.")
        return

    dataset_folders = [
        p for p in RAW_DIR.iterdir()
        if p.is_dir()
    ]

    if not dataset_folders:
        print("\nNo datasets found yet.")
        print("Download datasets into datasets/raw/")
        return

    for dataset_folder in sorted(dataset_folders):
        audit_dataset(dataset_folder)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
