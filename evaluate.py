
# ============================================================
# DRIVER BEHAVIOR CLASSIFICATION - MODEL EVALUATION
# ============================================================

import csv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report

from config import (
    PROCESSED_DIR,
    MODEL_DIR,
    REPORT_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

device = torch.device(DEVICE)

DATASET_PATH = PROCESSED_DIR / "driver_risk_behavior" / "test"
MODEL_PATH = MODEL_DIR / "driver_resnet18_best.pth"

CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.csv"
METRICS_PATH = REPORT_DIR / "test_metrics.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DRIVER BEHAVIOR MODEL - EVALUATION")
print("=" * 65)

# ============================================================
# 2. CHECK PATHS
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {DATASET_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

# ============================================================
# 3. IMAGE TRANSFORMATIONS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# 4. LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("\nTest dataset :", DATASET_PATH)
print("Test images :", len(test_dataset))
print("ImageFolder classes:", test_dataset.classes)
print("Model class order :", CLASS_NAMES)

# ============================================================
# 5. MAP IMAGEFOLDER LABELS TO MODEL LABELS
# ============================================================

# ImageFolder assigns class IDs alphabetically.
# Our trained model uses the order defined in config.py.

imagefolder_to_model = {
    test_dataset.class_to_idx[class_name]: CLASS_NAMES.index(class_name)
    for class_name in test_dataset.classes
}

print("Label mapping:", imagefolder_to_model)

# ============================================================
# 6. LOAD RESNET18 MODEL
# ============================================================

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print("\nModel loaded successfully!")

if "epoch" in checkpoint:
    print("Checkpoint epoch:", checkpoint["epoch"])

if "best_val_accuracy" in checkpoint:
    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )

# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

print("\nEvaluating model on test data...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        # Convert ImageFolder labels to model class order
        mapped_labels = [
            imagefolder_to_model[label.item()]
            for label in labels
        ]

        all_labels.extend(mapped_labels)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

# ============================================================
# 8. CALCULATE METRICS
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES)))
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

accuracy = report["accuracy"]

# ============================================================
# 9. PRINT EVALUATION RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TEST EVALUATION RESULTS")
print("=" * 65)

print("Total test images:", len(test_dataset))
print(f"Test accuracy    : {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print("-" * 65)
print("Rows = Actual classes")
print("Columns = Predicted classes")
print("\nClass order:", CLASS_NAMES)
print(cm)

# ============================================================
# 10. SAVE CONFUSION MATRIX
# ============================================================

with open(
    CONFUSION_MATRIX_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Actual / Predicted"] + CLASS_NAMES
    )

    for class_name, row in zip(CLASS_NAMES, cm):

        writer.writerow(
            [class_name] + row.tolist()
        )

# ============================================================
# 11. SAVE CLASS METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class",
        "precision",
        "recall",
        "f1_score",
        "support"
    ])

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        writer.writerow([
            class_name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
            metrics["support"]
        ])

    writer.writerow([
        "overall_accuracy",
        accuracy,
        accuracy,
        accuracy,
        len(test_dataset)
    ])

# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\nREPORTS SAVED SUCCESSFULLY")
print("-" * 65)

print("Confusion matrix:", CONFUSION_MATRIX_PATH)
print("Metrics report  :", METRICS_PATH)

print("\n" + "=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)
