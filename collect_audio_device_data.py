
import cv2
import time
from pathlib import Path

# Dataset directory
BASE_DIR = Path("datasets/audio_devices/images")

CLASSES = {
    "1": "earbud",
    "2": "wired_earphone",
    "3": "headphone",
    "4": "headset",
    "5": "cell_phone",
    "6": "no_device",
}

# Create folders if they do not exist
for class_name in CLASSES.values():
    (BASE_DIR / class_name).mkdir(parents=True, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

selected_class = None
image_count = 0

def count_images(class_name):
    folder = BASE_DIR / class_name
    return len(list(folder.glob("*.jpg")))

print("\n========== AUDIO DEVICE DATA COLLECTOR ==========")
print("1 - Earbud")
print("2 - Wired earphone")
print("3 - Headphone")
print("4 - Headset")
print("5 - Cell phone")
print("6 - No device")
print("SPACE - Capture image")
print("Q - Quit")
print("=================================================\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam view
    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # Display instructions
    cv2.putText(
        display,
        "AUDIO DEVICE DATA COLLECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if selected_class is None:
        status = "Select a class using keys 1-6"
        count_text = ""
    else:
        image_count = count_images(selected_class)
        status = f"Selected: {selected_class}"
        count_text = f"Saved images: {image_count}"

    cv2.putText(
        display,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        count_text,
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "SPACE: Capture | 1-6: Change class | Q: Quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Audio Device Dataset Collector", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif chr(key) in CLASSES if key < 128 else False:
        selected_class = CLASSES[chr(key)]
        print(f"\nSelected class: {selected_class}")

    elif key == 32:
        if selected_class is None:
            print("Select a class first!")
            continue

        folder = BASE_DIR / selected_class
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{selected_class}_{timestamp}_{time.time_ns()}.jpg"
        filepath = folder / filename

        # Save the original camera frame (mirrored)
        cv2.imwrite(str(filepath), frame)

        image_count = count_images(selected_class)

        print(
            f"Saved: {filename} | "
            f"Class: {selected_class} | "
            f"Total: {image_count}"
        )

cap.release()
cv2.destroyAllWindows()

print("\nData collection stopped.")
