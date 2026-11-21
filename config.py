
# ==========================================
# DRIVER DISTRACTION DETECTION - CONFIGURATION
# ==========================================

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- Dataset paths ----------
DATASETS_DIR = ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"

STATEFARM_DIR = RAW_DIR / "statefarm"
AUC_DIR = RAW_DIR / "auc"
ADDITIONAL_DIR = RAW_DIR / "additional"

# New dataset: Driver Risk Behavior Dataset
DRIVER_RISK_DIR = (
    ADDITIONAL_DIR
    / "driver_risk_behavior"
    / "dataset_files"
    / "Images_Labels"
)

# Processed classification dataset
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

# ---------- Model / output paths ----------
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

# ---------- Driver behavior classes ----------
CLASS_NAMES = [
    "neutral",
    "microsueno",
    "distraccion",
    "uso_celular",
]

CLASS_LABELS = {
    "neutral": "Neutral driving",
    "microsueno": "Micro-sleep",
    "distraccion": "Distraction",
    "uso_celular": "Mobile phone usage",
}

NUM_CLASSES = len(CLASS_NAMES)

# ---------- YOLO annotation class mapping ----------
YOLO_CLASS_MAPPING = {
    0: "neutral",
    1: "microsueno",
    2: "distraccion",
    3: "uso_celular",
}

# ---------- Training settings ----------
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# ---------- Hardware ----------
# Your current PyTorch installation is CPU-based
DEVICE = "cpu"
NUM_WORKERS = 0

# ---------- Reproducibility ----------
SEED = 42

# ---------- Create folders automatically ----------
REQUIRED_DIRS = [
    RAW_DIR,
    STATEFARM_DIR,
    AUC_DIR,
    ADDITIONAL_DIR,
    DRIVER_RISK_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_DIR,
    REPORT_DIR,
    LOG_DIR,
]

for folder in REQUIRED_DIRS:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 45)
print("DRIVER MONITORING CONFIGURATION LOADED")
print("=" * 45)
print("Project root :", ROOT)
print("Dataset     :", DRIVER_RISK_DIR)
print("Classes     :", NUM_CLASSES)
print("Class names :", CLASS_NAMES)
print("Image size  :", IMAGE_SIZE)
print("Batch size  :", BATCH_SIZE)
print("Epochs      :", EPOCHS)
print("Device      :", DEVICE)
print("=" * 45)
