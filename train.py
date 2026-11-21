
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from config import (
    PROCESSED_DIR,
    MODEL_DIR,
    LOG_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    NUM_WORKERS,
    SEED,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

torch.manual_seed(SEED)

device = torch.device(DEVICE)

DATASET_ROOT = PROCESSED_DIR / "driver_risk_behavior"
TRAIN_PATH = DATASET_ROOT / "train"
VAL_PATH = DATASET_ROOT / "val"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DRIVER BEHAVIOR CLASSIFICATION - RESNET18")
print("=" * 65)
print(f"Device       : {device}")
print(f"Classes      : {CLASS_NAMES}")
print(f"Image size   : {IMAGE_SIZE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Epochs      : {EPOCHS}")


# ============================================================
# 2. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_PATH),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=str(VAL_PATH),
    transform=val_transform
)

# ImageFolder sorts folder names alphabetically.
# We explicitly remap its labels to the order in CLASS_NAMES.
print("\nImageFolder class order:", train_dataset.classes)
print("Model class order     :", CLASS_NAMES)

if set(train_dataset.classes) != set(CLASS_NAMES):
    raise ValueError(
        "Training folders do not match CLASS_NAMES.\n"
        f"Folders found: {train_dataset.classes}\n"
        f"Expected: {CLASS_NAMES}"
    )

if set(val_dataset.classes) != set(CLASS_NAMES):
    raise ValueError(
        "Validation folders do not match CLASS_NAMES.\n"
        f"Folders found: {val_dataset.classes}\n"
        f"Expected: {CLASS_NAMES}"
    )

# ImageFolder label -> model label
train_label_map = torch.tensor([
    CLASS_NAMES.index(name)
    for name in train_dataset.classes
], dtype=torch.long)

val_label_map = torch.tensor([
    CLASS_NAMES.index(name)
    for name in val_dataset.classes
], dtype=torch.long)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"\nTraining images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Training batches  : {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# ============================================================
# 4. CLASS COUNTS AND CLASS-WEIGHTED LOSS
# ============================================================

class_counts = [0] * len(CLASS_NAMES)

for _, original_label in train_dataset.samples:
    class_name = train_dataset.classes[original_label]
    model_label = CLASS_NAMES.index(class_name)
    class_counts[model_label] += 1

print("\nTraining class distribution:")

for name, count in zip(CLASS_NAMES, class_counts):
    print(f"{name:20s}: {count}")

# Balanced class weights
total_samples = sum(class_counts)

class_weights = [
    total_samples / (len(CLASS_NAMES) * count)
    for count in class_counts
]

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ============================================================
# 5. LOAD RESNET18
# ============================================================

print("\nLoading ResNet18...")

try:
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    print("Pretrained ImageNet weights loaded.")

except Exception as error:
    print("Pretrained weights could not be loaded.")
    print("Reason:", error)
    print("Creating ResNet18 without pretrained weights.")

    model = models.resnet18(weights=None)

# Replace original 1000-class output with 4 classes
model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model = model.to(device)


# ============================================================
# 6. OPTIMIZER AND SCHEDULER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, original_labels in train_loader:

        images = images.to(device)

        # Remap labels to the model's class order
        labels = train_label_map[original_labels].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size_actual = images.size(0)

        total_loss += loss.item() * batch_size_actual

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 8. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate_one_epoch():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, original_labels in val_loader:

        images = images.to(device)

        # Remap validation labels to model class order
        labels = val_label_map[original_labels].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size_actual = images.size(0)

        total_loss += loss.item() * batch_size_actual

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 9. TRAINING LOOP
# ============================================================

BEST_MODEL_PATH = MODEL_DIR / "driver_resnet18_best.pth"
LAST_MODEL_PATH = MODEL_DIR / "driver_resnet18_last.pth"
HISTORY_PATH = LOG_DIR / "training_history.csv"

best_val_accuracy = -1.0
history = []

print("\n" + "=" * 65)
print("STARTING TRAINING")
print("=" * 65)

start_time = time.time()

for epoch in range(EPOCHS):

    epoch_start = time.time()

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate_one_epoch()

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print("-" * 55)
    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Train Accuracy  : {train_accuracy * 100:.2f}%")
    print(f"Val Loss        : {val_loss:.4f}")
    print(f"Val Accuracy    : {val_accuracy * 100:.2f}%")
    print(f"Learning Rate   : {current_lr:.6f}")
    print(f"Epoch Time     : {epoch_time:.1f} seconds")

    # Record history
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })

    # Save best model (also works if first accuracy is 0%)
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": best_val_accuracy,
            "epoch": epoch + 1
        }, BEST_MODEL_PATH)

        print("Best model saved!")

    # Save latest checkpoint every epoch
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "epoch": epoch + 1
    }, LAST_MODEL_PATH)

    # Update CSV training history
    with open(HISTORY_PATH, "w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate"
            ]
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 10. TRAINING COMPLETED
# ============================================================

total_time = time.time() - start_time

print("\n" + "=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Total training time    : {total_time / 60:.2f} minutes")

print("\nSaved files:")
print(f"Best model : {BEST_MODEL_PATH}")
print(f"Last model: {LAST_MODEL_PATH}")
print(f"History   : {HISTORY_PATH}")

print("\nNext step: Run evaluate.py to test the model.")
