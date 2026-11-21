import cv2
import time
import threading
import numpy as np
import winsound
from collections import deque
from pathlib import Path

from monitoring import DriverFaceMonitor
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

# Alarm timing
RISK_DURATION = 6.0
PHONE_DURATION = 6.0
AUDIO_DEVICE_DURATION = 6.0
ALARM_COOLDOWN = 3.0

# Face thresholds
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
YAW_THRESHOLD = 18.0
PITCH_DOWN_THRESHOLD = 20.0

# YOLO settings
YOLO_INTERVAL = 4
PHONE_CONFIDENCE = 0.18
AUDIO_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640

# Detection persistence
PHONE_PERSISTENCE = 1.0
AUDIO_PERSISTENCE = 1.0

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Dashboard
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 880

# Models
PHONE_MODEL_PATH = "yolov8n.pt"

# Optional custom model. It must be trained before it can work.
AUDIO_DEVICE_MODEL_PATH = "models/audio_device_yolo.pt"


# ============================================================
# COLORS (BGR)
# ============================================================

BG = (18, 20, 28)
PANEL = (30, 34, 45)
PANEL_LIGHT = (42, 47, 60)

WHITE = (240, 243, 250)
GRAY = (155, 164, 180)

GREEN = (60, 210, 120)
YELLOW = (0, 210, 255)
ORANGE = (0, 145, 255)
RED = (60, 60, 240)
BLUE = (235, 150, 70)
CYAN = (220, 220, 50)
PURPLE = (200, 100, 230)


# ============================================================
# GLOBAL ALARM CONTROL
# ============================================================

alarm_lock = threading.Lock()
last_alarm_time = 0.0
alarm_active = False


def play_alarm(reason):
    """
    Plays three beeps with a global cooldown.
    """

    global last_alarm_time, alarm_active

    now = time.time()

    with alarm_lock:

        if alarm_active:
            return

        if now - last_alarm_time < ALARM_COOLDOWN:
            return

        last_alarm_time = now
        alarm_active = True

    def alarm_worker():

        global alarm_active

        try:

            print(f"\n[ALARM] {reason}")

            for _ in range(3):
                winsound.Beep(1800, 450)
                time.sleep(0.12)

        except Exception as e:

            print("[ALARM ERROR]", e)

        finally:

            with alarm_lock:
                alarm_active = False

    threading.Thread(
        target=alarm_worker,
        daemon=True
    ).start()


# ============================================================
# YOLO HELPERS
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def make_rois(frame):

    height, width = frame.shape[:2]

    left_x2 = int(width * 0.65)
    right_x1 = int(width * 0.35)
    lower_y1 = int(height * 0.30)

    return [
        (frame, 0, 0, "FULL"),
        (frame[:, :left_x2], 0, 0, "LEFT"),
        (frame[:, right_x1:], right_x1, 0, "RIGHT"),
        (frame[lower_y1:, :], 0, lower_y1, "LOWER"),
    ]


def run_multicrop_detection(frame, model, confidence, target_names=None):

    """
    Runs YOLO on full frame + overlapping crops.

    target_names:
        None = return every class
        set/list = only return those class names

    Coordinates are translated to original full-frame positions.
    Duplicate boxes from overlapping crops are filtered.
    """

    rois = make_rois(frame)

    images = [
        roi_info[0]
        for roi_info in rois
    ]

    results = model.predict(
        source=images,
        imgsz=YOLO_IMAGE_SIZE,
        conf=confidence,
        iou=0.45,
        max_det=30,
        device="cpu",
        verbose=False
    )

    detections = []

    for result, roi_info in zip(results, rois):

        _, offset_x, offset_y, region_name = roi_info

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            class_name = str(class_name).lower().strip()

            if (
                target_names is not None
                and class_name not in target_names
            ):
                continue

            confidence_value = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            full_box = (
                int(x1 + offset_x),
                int(y1 + offset_y),
                int(x2 + offset_x),
                int(y2 + offset_y)
            )

            detections.append({
                "box": full_box,
                "confidence": confidence_value,
                "class_name": class_name,
                "region": region_name
            })

    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    # Duplicate filtering
    filtered = []

    for detection in detections:

        duplicate = False

        for kept in filtered:

            same_class = (
                detection["class_name"]
                == kept["class_name"]
            )

            if (
                same_class
                and calculate_iou(
                    detection["box"],
                    kept["box"]
                ) > 0.45
            ):

                duplicate = True
                break

        if not duplicate:
            filtered.append(detection)

    return filtered




# ============================================================
# BACKGROUND YOLO WORKER
# ============================================================

class YOLOBackgroundWorker:

    def __init__(self, phone_model, audio_model):

        self.phone_model = phone_model
        self.audio_model = audio_model

        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

        self.latest_frame = None
        self.latest_result = None

        self.busy = False
        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.thread.start()

    def submit_frame(self, frame):

        with self.lock:

            if not self.running or self.busy:
                return

            self.latest_frame = frame.copy()
            self.busy = True

        self.new_frame_event.set()

    def get_result(self):

        with self.lock:

            result = self.latest_result
            self.latest_result = None

        return result

    def _worker_loop(self):

        while self.running:

            self.new_frame_event.wait()

            if not self.running:
                break

            self.new_frame_event.clear()

            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None

            if frame is None:

                with self.lock:
                    self.busy = False

                continue

            phone_detections = []
            audio_detections = []

            try:

                try:

                    phone_detections = run_multicrop_detection(
                        frame,
                        self.phone_model,
                        PHONE_CONFIDENCE,
                        target_names={
                            "cell phone",
                            "mobile phone"
                        }
                    )

                except Exception as e:

                    print("\n[PHONE YOLO ERROR]", e)

                if self.audio_model is not None:

                    try:

                        audio_detections = run_multicrop_detection(
                            frame,
                            self.audio_model,
                            AUDIO_CONFIDENCE,
                            target_names={
                                "earbud",
                                "earbuds",
                                "wireless earbud",
                                "wired earphone",
                                "earphone",
                                "headphone",
                                "headphones",
                                "headset"
                            }
                        )

                    except Exception as e:

                        print("\n[AUDIO YOLO ERROR]", e)

                result = {
                    "phone_boxes": phone_detections,
                    "audio_boxes": audio_detections,
                    "timestamp": time.time()
                }

                with self.lock:
                    self.latest_result = result

            except Exception as e:

                print("\n[YOLO WORKER ERROR]", e)

            finally:

                with self.lock:
                    self.busy = False

    def stop(self):

        self.running = False
        self.new_frame_event.set()

        if self.thread.is_alive():
            self.thread.join(timeout=10)


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(img, text, x, y, scale=0.55,
              color=WHITE, thickness=1):

    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_panel(img, x, y, w, h, color=PANEL):

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        color,
        -1
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        PANEL_LIGHT,
        1
    )


def draw_metric_card(img, x, y, w, h,
                    title, value, subtitle,
                    accent=CYAN):

    draw_panel(img, x, y, w, h)

    cv2.rectangle(
        img,
        (x, y),
        (x + 5, y + h),
        accent,
        -1
    )

    draw_text(
        img, title,
        x + 18, y + 28,
        0.50, GRAY, 1
    )

    draw_text(
        img, value,
        x + 18, y + 67,
        0.88, WHITE, 2
    )

    draw_text(
        img, subtitle,
        x + 18, y + h - 16,
        0.43, GRAY, 1
    )


def draw_progress_bar(img, x, y, w, h,
                     progress, color):

    progress = max(
        0.0,
        min(1.0, progress)
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        PANEL_LIGHT,
        -1
    )

    fill_width = int(w * progress)

    if fill_width > 0:

        cv2.rectangle(
            img,
            (x, y),
            (x + fill_width, y + h),
            color,
            -1
        )


def draw_status_dot(img, x, y, color, radius=6):

    cv2.circle(
        img,
        (x, y),
        radius,
        color,
        -1
    )


def format_seconds(seconds):

    return f"{max(0.0, seconds):04.1f} s"


def draw_detection_boxes(frame, detections, color, prefix=""):

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]
        name = detection["class_name"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{prefix}{name} {confidence:.2f}"

        label_y = max(20, y1 - 8)

        cv2.rectangle(
            frame,
            (x1, max(0, label_y - 24)),
            (x1 + min(250, len(label) * 10), label_y + 3),
            color,
            -1
        )

        draw_text(
            frame,
            label,
            x1 + 4,
            label_y,
            0.46,
            WHITE,
            1
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("AI DRIVER MONITORING SYSTEM")
    print("=" * 65)

    # --------------------------------------------------------
    # Face monitor
    # --------------------------------------------------------

    print("[INFO] Loading face monitor...")

    face_monitor = DriverFaceMonitor()

    # --------------------------------------------------------
    # Phone model
    # --------------------------------------------------------

    print("[INFO] Loading phone detection model...")

    phone_model = YOLO(PHONE_MODEL_PATH)

    # --------------------------------------------------------
    # Optional custom audio device model
    # --------------------------------------------------------

    audio_model = None

    audio_model_file = Path(
        AUDIO_DEVICE_MODEL_PATH
    )

    if audio_model_file.exists():

        print(
            "[INFO] Loading custom audio-device model:",
            AUDIO_DEVICE_MODEL_PATH
        )

        audio_model = YOLO(
            AUDIO_DEVICE_MODEL_PATH
        )

        print(
            "[INFO] Custom audio-device model loaded."
        )

    else:

        print(
            "[INFO] Custom earphone/headphone model not found."
        )

        print(
            "[INFO] Earphone/headphone detection is disabled "
            "until custom training is completed."
        )

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    print("[INFO] Opening webcam...")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():

        print("[ERROR] Webcam could not be opened.")

        face_monitor.close()
        return

    # ========================================================
    # STATE
    # ========================================================

    frame_count = 0

    # Driver-risk timer
    risk_start = None
    risk_elapsed = 0.0
    risk_triggered = False

    # Phone timer
    phone_start = None
    phone_elapsed = 0.0
    phone_triggered = False

    last_phone_seen = 0.0
    phone_detected = False
    phone_confidence = 0.0
    phone_boxes = []

    # Audio-device timer
    audio_start = None
    audio_elapsed = 0.0
    audio_triggered = False

    last_audio_seen = 0.0
    audio_detected = False
    audio_boxes = []
    audio_label = "NOT TRAINED"

    # Detection update timing
    last_yolo_time = 0.0

    # FPS
    fps_history = deque(maxlen=30)
    previous_frame_time = time.time()

    # Event history
    events = deque(maxlen=6)

    def add_event(message):

        timestamp = time.strftime("%H:%M:%S")

        events.appendleft(
            f"{timestamp} | {message}"
        )

        print(
            f"\n[EVENT {timestamp}] {message}"
        )

    # --------------------------------------------------------
    # Start background YOLO worker
    # --------------------------------------------------------

    yolo_worker = YOLOBackgroundWorker(
        phone_model,
        audio_model
    )

    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("[ERROR] Camera frame failed.")
                break

            frame_count += 1

            frame = cv2.flip(frame, 1)

            now = time.time()

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            delta = now - previous_frame_time

            if delta > 0:
                fps_history.append(1.0 / delta)

            previous_frame_time = now

            fps = (
                sum(fps_history) / len(fps_history)
                if fps_history
                else 0.0
            )

            # ------------------------------------------------
            # FACE ANALYSIS
            # ------------------------------------------------

            face_data = face_monitor.process_frame(frame)

            face_detected = face_data.get(
                "face_detected", False
            )

            ear = face_data.get("ear", 0.0)
            mar = face_data.get("mar", 0.0)
            yaw = face_data.get("yaw", 0.0)
            pitch = face_data.get("pitch", 0.0)

            eyes_closed = (
                face_detected
                and ear > 0
                and ear < EAR_THRESHOLD
            )

            # Updated yaw threshold
            head_turned = (
                face_detected
                and abs(yaw) >= YAW_THRESHOLD
            )

            head_down = (
                face_detected
                and pitch >= PITCH_DOWN_THRESHOLD
            )

            mouth_open = (
                face_detected
                and mar >= MAR_THRESHOLD
            )

            face_missing = not face_detected

            # Live terminal head-angle monitor
            print(
                f"\rYaw: {yaw:7.2f} | "
                f"Pitch: {pitch:7.2f} | "
                f"Face: {face_detected} | "
                f"Turn: {head_turned}",
                end="",
                flush=True
            )

            # ------------------------------------------------
            # BACKGROUND YOLO DETECTION
            # ------------------------------------------------

            # Submit a camera frame periodically.
            # If inference is still running, this frame is skipped.
            if frame_count % YOLO_INTERVAL == 0:

                yolo_worker.submit_frame(frame)

            # Read completed detections without blocking the camera.
            yolo_result = yolo_worker.get_result()

            if yolo_result is not None:

                detection_time = yolo_result["timestamp"]

                # PHONE RESULTS
                phone_boxes = yolo_result["phone_boxes"]

                if phone_boxes:

                    last_phone_seen = detection_time

                    phone_confidence = max(
                        d["confidence"]
                        for d in phone_boxes
                    )

                else:

                    phone_confidence = 0.0

                # AUDIO DEVICE RESULTS
                if audio_model is not None:

                    audio_boxes = yolo_result["audio_boxes"]

                    if audio_boxes:

                        last_audio_seen = detection_time

                        audio_label = (
                            audio_boxes[0]["class_name"]
                        )

                    else:

                        audio_label = "NONE"

                last_yolo_time = detection_time

            # ------------------------------------------------
            # PHONE DETECTION PERSISTENCE            # ------------------------------------------------
            # PHONE DETECTION PERSISTENCE
            # ------------------------------------------------

            if (
                now - last_phone_seen
                <= PHONE_PERSISTENCE
            ):

                phone_detected = True

            else:

                phone_detected = False
                phone_boxes = []

            # ------------------------------------------------
            # AUDIO DETECTION PERSISTENCE
            # ------------------------------------------------

            if audio_model is not None:

                if (
                    now - last_audio_seen
                    <= AUDIO_PERSISTENCE
                ):

                    audio_detected = True

                else:

                    audio_detected = False
                    audio_boxes = []
                    audio_label = "NONE"

            # ------------------------------------------------
            # DRIVER RISK TIMER
            # ------------------------------------------------

            risk_active = (
                eyes_closed
                or head_turned
                or head_down
                or face_missing
            )

            if risk_active:

                if risk_start is None:

                    risk_start = now
                    risk_triggered = False

                    if head_turned:

                        add_event(
                            f"HEAD TURN DETECTED | Yaw {yaw:.1f}°"
                        )

                    elif eyes_closed:

                        add_event(
                            "EYE CLOSURE DETECTED"
                        )

                    elif head_down:

                        add_event(
                            "HEAD DOWN DETECTED"
                        )

                    elif face_missing:

                        add_event(
                            "FACE NOT DETECTED"
                        )

                risk_elapsed = now - risk_start

                if (
                    risk_elapsed >= RISK_DURATION
                    and not risk_triggered
                ):

                    risk_triggered = True

                    play_alarm(
                        "DRIVER RISK - 6 SECOND TRIGGER"
                    )

                    add_event(
                        "DRIVER RISK ALARM TRIGGERED"
                    )

            else:

                risk_start = None
                risk_elapsed = 0.0
                risk_triggered = False

            # ------------------------------------------------
            # PHONE TIMER
            # ------------------------------------------------

            if phone_detected:

                if phone_start is None:

                    phone_start = now
                    phone_triggered = False

                    add_event(
                        "CELL PHONE DETECTED"
                    )

                phone_elapsed = now - phone_start

                if (
                    phone_elapsed >= PHONE_DURATION
                    and not phone_triggered
                ):

                    phone_triggered = True

                    play_alarm(
                        "PHONE - 6 SECOND TRIGGER"
                    )

                    add_event(
                        "PHONE ALARM TRIGGERED"
                    )

            else:

                phone_start = None
                phone_elapsed = 0.0
                phone_triggered = False

            # ------------------------------------------------
            # AUDIO DEVICE TIMER
            # ------------------------------------------------

            if audio_model is not None:

                if audio_detected:

                    if audio_start is None:

                        audio_start = now
                        audio_triggered = False

                        add_event(
                            f"AUDIO DEVICE DETECTED: {audio_label}"
                        )

                    audio_elapsed = now - audio_start

                    if (
                        audio_elapsed >= AUDIO_DEVICE_DURATION
                        and not audio_triggered
                    ):

                        audio_triggered = True

                        play_alarm(
                            "AUDIO DEVICE - 6 SECOND TRIGGER"
                        )

                        add_event(
                            "AUDIO DEVICE ALARM TRIGGERED"
                        )

                else:

                    audio_start = None
                    audio_elapsed = 0.0
                    audio_triggered = False

            # ------------------------------------------------
            # DRAW DETECTIONS
            # ------------------------------------------------

            draw_detection_boxes(
                frame,
                phone_boxes,
                RED,
                prefix="PHONE: "
            )

            draw_detection_boxes(
                frame,
                audio_boxes,
                PURPLE,
                prefix="DEVICE: "
            )

            # ------------------------------------------------
            # CAMERA STATUS
            # ------------------------------------------------

            if not face_detected:

                camera_status = "FACE NOT DETECTED"
                camera_color = RED

            elif risk_active:

                camera_status = "DRIVER RISK"
                camera_color = ORANGE

            else:

                camera_status = "DRIVER NORMAL"
                camera_color = GREEN

            cv2.rectangle(
                frame,
                (10, 10),
                (350, 48),
                BG,
                -1
            )

            draw_text(
                frame,
                camera_status,
                20,
                36,
                0.65,
                camera_color,
                2
            )

            # =================================================
            # DASHBOARD
            # =================================================

            dashboard = np.zeros(
                (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
                dtype=np.uint8
            )

            dashboard[:] = BG

            # HEADER
            draw_panel(
                dashboard,
                20, 15,
                WINDOW_WIDTH - 40, 70
            )

            draw_text(
                dashboard,
                "AI DRIVER MONITORING SYSTEM",
                42, 49,
                0.85,
                WHITE,
                2
            )

            draw_text(
                dashboard,
                "REAL-TIME DRIVER SAFETY ANALYTICS",
                650, 48,
                0.48,
                GRAY,
                1
            )

            draw_text(
                dashboard,
                f"FPS: {fps:.1f}",
                1110, 49,
                0.55,
                CYAN,
                1
            )

            # ------------------------------------------------
            # LIVE CAMERA
            # ------------------------------------------------

            cam_x = 20
            cam_y = 105
            cam_w = 740
            cam_h = 510

            draw_panel(
                dashboard,
                cam_x, cam_y,
                cam_w, cam_h
            )

            draw_text(
                dashboard,
                "LIVE CAMERA FEED",
                cam_x + 18,
                cam_y + 30,
                0.55,
                CYAN,
                1
            )

            display_w = 704
            display_h = 450

            resized_frame = cv2.resize(
                frame,
                (display_w, display_h)
            )

            dashboard[
                cam_y + 45:cam_y + 45 + display_h,
                cam_x + 18:cam_x + 18 + display_w
            ] = resized_frame

            # ------------------------------------------------
            # DRIVER STATUS
            # ------------------------------------------------

            side_x = 780
            side_w = 480

            draw_panel(
                dashboard,
                side_x, 105,
                side_w, 155
            )

            draw_text(
                dashboard,
                "DRIVER STATUS",
                side_x + 20,
                137,
                0.55,
                GRAY,
                1
            )

            draw_status_dot(
                dashboard,
                side_x + 35,
                180,
                camera_color,
                10
            )

            draw_text(
                dashboard,
                camera_status,
                side_x + 58,
                188,
                0.72,
                camera_color,
                2
            )

            draw_text(
                dashboard,
                (
                    "HEAD TURN DETECTED"
                    if head_turned
                    else "HEAD DIRECTION NORMAL"
                ),
                side_x + 20,
                226,
                0.52,
                ORANGE if head_turned else GREEN,
                1
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            card_y = 275
            card_w = 225
            card_h = 105
            gap = 20

            draw_metric_card(
                dashboard,
                side_x,
                card_y,
                card_w,
                card_h,
                "EYE RATIO (EAR)",
                f"{ear:.3f}" if face_detected else "--",
                "Lower value = eye closure",
                RED if eyes_closed else GREEN
            )

            draw_metric_card(
                dashboard,
                side_x + card_w + gap,
                card_y,
                card_w,
                card_h,
                "MOUTH RATIO (MAR)",
                f"{mar:.3f}" if face_detected else "--",
                "Mouth opening indicator",
                ORANGE if mouth_open else CYAN
            )

            draw_metric_card(
                dashboard,
                side_x,
                card_y + card_h + 15,
                card_w,
                card_h,
                "HEAD YAW",
                f"{yaw:.1f}°" if face_detected else "--",
                f"Alarm threshold: {YAW_THRESHOLD:.0f}°",
                ORANGE if head_turned else GREEN
            )

            draw_metric_card(
                dashboard,
                side_x + card_w + gap,
                card_y + card_h + 15,
                card_w,
                card_h,
                "HEAD PITCH",
                f"{pitch:.1f}°" if face_detected else "--",
                "Vertical head orientation",
                ORANGE if head_down else GREEN
            )

            # ------------------------------------------------
            # RISK TIMER
            # ------------------------------------------------

            timer_y = 510

            draw_panel(
                dashboard,
                side_x, timer_y,
                side_w, 115
            )

            draw_text(
                dashboard,
                "DRIVER RISK TIMER",
                side_x + 20,
                timer_y + 28,
                0.52,
                GRAY,
                1
            )

            draw_text(
                dashboard,
                format_seconds(risk_elapsed),
                side_x + 20,
                timer_y + 65,
                0.85,
                ORANGE if risk_active else GREEN,
                2
            )

            draw_progress_bar(
                dashboard,
                side_x + 20,
                timer_y + 82,
                side_w - 40,
                12,
                risk_elapsed / RISK_DURATION,
                ORANGE if risk_active else GREEN
            )

            draw_text(
                dashboard,
                "ALARM!" if risk_triggered else "6s",
                side_x + 355,
                timer_y + 66,
                0.45,
                RED if risk_triggered else GRAY,
                1
            )

            # ------------------------------------------------
            # PHONE STATUS PANEL
            # ------------------------------------------------

            phone_y = 640

            draw_panel(
                dashboard,
                20, phone_y,
                740, 125
            )

            draw_text(
                dashboard,
                "PHONE DETECTION",
                40,
                phone_y + 30,
                0.55,
                GRAY,
                1
            )

            phone_color = RED if phone_detected else GREEN

            draw_status_dot(
                dashboard,
                55,
                phone_y + 65,
                phone_color,
                9
            )

            draw_text(
                dashboard,
                (
                    "PHONE DETECTED"
                    if phone_detected
                    else "NO PHONE DETECTED"
                ),
                75,
                phone_y + 71,
                0.65,
                phone_color,
                2
            )

            draw_text(
                dashboard,
                f"Confidence: {phone_confidence:.2f}",
                40,
                phone_y + 103,
                0.48,
                GRAY,
                1
            )

            # ------------------------------------------------
            # PHONE TIMER PANEL
            # ------------------------------------------------

            draw_panel(
                dashboard,
                780, phone_y,
                480, 125
            )

            draw_text(
                dashboard,
                "PHONE ALARM TIMER",
                800,
                phone_y + 30,
                0.55,
                GRAY,
                1
            )

            draw_text(
                dashboard,
                format_seconds(phone_elapsed),
                800,
                phone_y + 65,
                0.85,
                RED if phone_detected else GREEN,
                2
            )

            draw_progress_bar(
                dashboard,
                800,
                phone_y + 82,
                440,
                12,
                phone_elapsed / PHONE_DURATION,
                RED if phone_detected else GREEN
            )

            draw_text(
                dashboard,
                "ALARM!" if phone_triggered else "6s",
                1170,
                phone_y + 66,
                0.45,
                RED if phone_triggered else GRAY,
                1
            )

            # ------------------------------------------------
            # AUDIO DEVICE PANEL
            # ------------------------------------------------

            audio_y = 780

            draw_panel(
                dashboard,
                20, audio_y,
                WINDOW_WIDTH - 40, 65
            )

            draw_text(
                dashboard,
                "EARBUD / EARPHONE / HEADPHONE / HEADSET",
                40,
                audio_y + 26,
                0.50,
                GRAY,
                1
            )

            if audio_model is None:

                audio_status_text = (
                    "CUSTOM MODEL NOT TRAINED / NOT LOADED"
                )

                audio_status_color = YELLOW

            elif audio_detected:

                audio_status_text = (
                    f"DEVICE: {audio_label.upper()} | "
                    f"{format_seconds(audio_elapsed)}"
                )

                audio_status_color = PURPLE

            else:

                audio_status_text = (
                    "NO AUDIO DEVICE DETECTED"
                )

                audio_status_color = GREEN

            draw_text(
                dashboard,
                audio_status_text,
                40,
                audio_y + 51,
                0.55,
                audio_status_color,
                1
            )

            # ------------------------------------------------
            # FOOTER
            # ------------------------------------------------

            draw_text(
                dashboard,
                "Q = EXIT",
                1110,
                865,
                0.45,
                GRAY,
                1
            )

            draw_text(
                dashboard,
                "EXPERIMENTAL DEMO - NOT FOR USE WHILE DRIVING",
                40,
                865,
                0.42,
                GRAY,
                1
            )

            # ------------------------------------------------
            # SHOW DASHBOARD
            # ------------------------------------------------

            cv2.imshow(
                "AI Driver Monitoring Dashboard",
                dashboard
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print("\n[INFO] Exit requested.")
                break

    except KeyboardInterrupt:

        print("\n[INFO] Interrupted.")

    except Exception as e:

        print("\n[ERROR]", e)

        import traceback
        traceback.print_exc()

    finally:

        print("\n[INFO] Stopping YOLO worker...")
        yolo_worker.stop()

        print("\n[INFO] Releasing webcam...")

        cap.release()

        print("[INFO] Closing face monitor...")

        face_monitor.close()

        cv2.destroyAllWindows()

        print("[INFO] Monitoring stopped.")
        print("=" * 65)


if __name__ == "__main__":
    main()