# ============================================================
# DRIVER MONITORING SYSTEM - REAL-TIME WEBCAM INFERENCE
# Model: ResNet18
# ============================================================

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from config import (
    MODEL_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    DEVICE,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = MODEL_DIR / "driver_resnet18_best.pth"

device = torch.device(DEVICE)

DISPLAY_NAMES = {
    "neutral": "NORMAL DRIVING",
    "microsueno": "MICRO-SLEEP",
    "distraccion": "DISTRACTION",
    "uso_celular": "PHONE USAGE",
}

# BGR colors for OpenCV
COLORS = {
    "neutral": (0, 200, 0),        # Green
    "microsueno": (0, 0, 255),     # Red
    "distraccion": (0, 165, 255),   # Orange
    "uso_celular": (0, 0, 255),    # Red
}

# ============================================================
# 2. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("DRIVER MONITORING - WEBCAM INFERENCE")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Validate checkpoint class order
checkpoint_classes = checkpoint.get("class_names", CLASS_NAMES)

if list(checkpoint_classes) != list(CLASS_NAMES):
    raise ValueError(
        f"Checkpoint classes {checkpoint_classes} "
        f"do not match config classes {CLASS_NAMES}"
    )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("Model loaded successfully!")
print("Classes:", CLASS_NAMES)
print("Device:", device)

# ============================================================
# 4. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam.")

# Optional camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nWebcam started.")
print("Press Q to quit.")
print("=" * 60)

# ============================================================
# 5. REAL-TIME INFERENCE LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Mirror the webcam for a natural viewing experience
        frame = cv2.flip(frame, 1)

        # OpenCV frame is BGR -> convert to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert frame to PIL image
        pil_image = Image.fromarray(rgb_frame)

        # Apply preprocessing
        input_tensor = transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = CLASS_NAMES[
            predicted_index.item()
        ]

        confidence_value = confidence.item() * 100

        behavior_text = DISPLAY_NAMES.get(
            predicted_class,
            predicted_class.upper()
        )

        color = COLORS.get(
            predicted_class,
            (255, 255, 255)
        )

        # ====================================================
        # DISPLAY OVERLAY
        # ====================================================

        # Dark header background
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 110),
            (30, 30, 30),
            -1
        )

        # Predicted behavior
        cv2.putText(
            frame,
            f"BEHAVIOR: {behavior_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

        # Confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence_value:.2f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # ALERT MESSAGE
        # ====================================================

        if predicted_class == "neutral":

            alert_text = "STATUS: NORMAL"

            alert_color = (0, 200, 0)

        elif predicted_class == "microsueno":

            alert_text = "WARNING: MICROSLEEP DETECTED!"

            alert_color = (0, 0, 255)

        elif predicted_class == "distraccion":

            alert_text = "WARNING: DRIVER DISTRACTED!"

            alert_color = (0, 165, 255)

        elif predicted_class == "uso_celular":

            alert_text = "WARNING: PHONE USAGE!"

            alert_color = (0, 0, 255)

        else:

            alert_text = "UNKNOWN BEHAVIOR"

            alert_color = (255, 255, 255)

        # Alert banner
        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 55),
            (frame.shape[1], frame.shape[0]),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            alert_text,
            (20, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            alert_color,
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "AI Driver Monitoring System - ResNet18",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ============================================================
# 6. CLEANUP
# ============================================================

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("\nWebcam stopped.")
    print("Resources released.")