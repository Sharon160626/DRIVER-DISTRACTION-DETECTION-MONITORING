
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from config import MODEL_DIR, IMAGE_SIZE, DEVICE, CLASS_LABELS


# Correct checkpoint filename
CHECKPOINT_PATH = MODEL_DIR / "driver_resnet18_best.pth"


def load_trained_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    # Correct checkpoint keys
    classes = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    # Rebuild ResNet18
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, transform, classes, device


def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print("Image file not found:", image_path)
        return

    model, transform, classes, device = load_trained_model()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print("Could not open image:", e)
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        top_k = min(3, len(classes))

        scores, indices = torch.topk(probabilities, top_k)

    print("\n" + "=" * 45)
    print("DRIVER BEHAVIOR PREDICTION")
    print("=" * 45)

    for rank, (score, index) in enumerate(
        zip(scores, indices), start=1
    ):

        class_name = classes[index.item()]
        label = CLASS_LABELS.get(class_name, class_name)

        print(
            f"{rank}. {label}: "
            f"{score.item() * 100:.2f}%"
        )

    predicted_class = classes[indices[0].item()]

    predicted_label = CLASS_LABELS.get(
        predicted_class, predicted_class
    )

    confidence = scores[0].item() * 100

    print("-" * 45)
    print("Predicted activity:", predicted_label)
    print(f"Confidence: {confidence:.2f}%")

    # Correct class name: neutral (not c0)
    if predicted_class == "neutral":
        print("Status: SAFE-CLASS ACTIVITY")
    else:
        print("Status: DRIVER DISTRACTION / RISK ACTIVITY")

    print("=" * 45)

    return predicted_class, confidence


if __name__ == "__main__":

    image_path = input(
        "Enter the full path of a driver image: "
    ).strip().strip('"')

    predict_image(image_path)
