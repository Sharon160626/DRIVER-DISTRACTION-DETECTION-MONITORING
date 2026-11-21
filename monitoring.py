
# ============================================================
# DRIVER MONITORING - FACIAL ANALYSIS ENGINE
# MediaPipe Face Landmarker | EAR | MAR | Head Pose
# Windows EXE Deployment Compatible
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# DEPLOYMENT PATH CONFIGURATION
# ============================================================

# When running normally:
# Uses the project directory.
#
# When packaged with PyInstaller:
# Uses the temporary extraction directory (_MEIPASS).

BASE_DIR = Path(
    getattr(
        sys,
        "_MEIPASS",
        Path(__file__).resolve().parent
    )
)

MODEL_PATH = str(
    BASE_DIR / "models" / "face_landmarker.task"
)


# ============================================================
# CONFIGURATION
# ============================================================

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60

HEAD_YAW_THRESHOLD = 25
HEAD_PITCH_DOWN_THRESHOLD = 20


# ============================================================
# FACE LANDMARK INDICES
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# DRIVER FACE MONITOR
# ============================================================

class DriverFaceMonitor:

    def __init__(self):

        # ----------------------------------------------------
        # LOAD MEDIAPIPE FACE LANDMARKER MODEL
        # ----------------------------------------------------

        if not Path(MODEL_PATH).is_file():
            raise FileNotFoundError(
                f"MediaPipe face model not found: {MODEL_PATH}"
            )

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

        self.start_time = cv2.getTickCount()

        self.camera_matrix = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64
        )

    # --------------------------------------------------------
    # EAR - EYE ASPECT RATIO
    # --------------------------------------------------------

    def calculate_ear(self, points):

        vertical_1 = np.linalg.norm(
            points[1] - points[5]
        )

        vertical_2 = np.linalg.norm(
            points[2] - points[4]
        )

        horizontal = np.linalg.norm(
            points[0] - points[3]
        )

        if horizontal == 0:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    # --------------------------------------------------------
    # MAR - MOUTH ASPECT RATIO
    # --------------------------------------------------------

    def calculate_mar(self, landmarks):

        top = landmarks[13]
        bottom = landmarks[14]

        left = landmarks[78]
        right = landmarks[308]

        vertical = np.linalg.norm(
            top - bottom
        )

        horizontal = np.linalg.norm(
            left - right
        )

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    # --------------------------------------------------------
    # HEAD POSE ESTIMATION
    # --------------------------------------------------------

    def calculate_head_pose(
        self,
        landmarks,
        width,
        height
    ):

        image_points = np.array([
            landmarks[1],
            landmarks[152],
            landmarks[33],
            landmarks[263],
            landmarks[61],
            landmarks[291]
        ], dtype=np.float64)

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        focal_length = width

        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:
            return 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = float(angles[0])
        yaw = float(angles[1])

        return yaw, pitch

    # --------------------------------------------------------
    # PROCESS WEBCAM FRAME
    # --------------------------------------------------------

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            (cv2.getTickCount() - self.start_time)
            / cv2.getTickFrequency() * 1000
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        if not result.face_landmarks:

            return {
                "face_detected": False,
                "ear": 0.0,
                "mar": 0.0,
                "yaw": 0.0,
                "pitch": 0.0
            }

        face = result.face_landmarks[0]

        landmarks = np.array([
            [
                point.x * width,
                point.y * height
            ]
            for point in face
        ], dtype=np.float64)

        left_eye_points = landmarks[LEFT_EYE]
        right_eye_points = landmarks[RIGHT_EYE]

        left_ear = self.calculate_ear(
            left_eye_points
        )

        right_ear = self.calculate_ear(
            right_eye_points
        )

        ear = (
            left_ear + right_ear
        ) / 2.0

        mar = self.calculate_mar(
            landmarks
        )

        yaw, pitch = self.calculate_head_pose(
            landmarks,
            width,
            height
        )

        # ----------------------------------------------------
        # DRAW FACE LANDMARK DOTS
        # ----------------------------------------------------

        for point in landmarks[::5]:

            cv2.circle(
                frame,
                (
                    int(point[0]),
                    int(point[1])
                ),
                1,
                (0, 255, 0),
                -1
            )

        return {
            "face_detected": True,
            "ear": float(ear),
            "mar": float(mar),
            "yaw": float(yaw),
            "pitch": float(pitch)
        }

    # --------------------------------------------------------
    # CLOSE MEDIAPIPE LANDMARKER
    # --------------------------------------------------------

    def close(self):

        self.landmarker.close()
