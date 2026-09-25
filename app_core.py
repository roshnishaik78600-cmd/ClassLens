import math
import os
import pickle
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from deepface import DeepFace


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = "shaik_embeddings.pkl"
ATTENDANCE_FILE = "attendance.csv"
MODEL_FILE = "models/face_landmarker.task"

RECOGNITION_INTERVAL = 10
REQUIRED_MATCHES = 3
RECOGNITION_THRESHOLD = 5.0

EAR_THRESHOLD = 0.20


# ============================================================
# LOAD SAVED EMBEDDINGS
# ============================================================

with open(EMBEDDINGS_FILE, "rb") as file:
    shaik_embeddings = pickle.load(file)

shaik_embeddings = [
    np.array(embedding, dtype=np.float32)
    for embedding in shaik_embeddings
]

print(f"Loaded Shaik embeddings: {len(shaik_embeddings)}")


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# LANDMARK DISTANCE
# ============================================================

def distance(p1, p2):
    """Calculate Euclidean distance between two face landmarks."""

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(landmarks):
    """Calculate average Eye Aspect Ratio for both eyes."""

    left_vertical = distance(
        landmarks[159],
        landmarks[145]
    )

    left_horizontal = distance(
        landmarks[33],
        landmarks[133]
    )

    right_vertical = distance(
        landmarks[386],
        landmarks[374]
    )

    right_horizontal = distance(
        landmarks[362],
        landmarks[263]
    )

    left_ear = (
        left_vertical /
        max(left_horizontal, 0.0001)
    )

    right_ear = (
        right_vertical /
        max(right_horizontal, 0.0001)
    )

    return (left_ear + right_ear) / 2


# ============================================================
# FACE BOX
# ============================================================

def get_face_box(landmarks, width, height):
    """Create a bounding box around the detected face."""

    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]

    x1 = int(min(xs) * width)
    y1 = int(min(ys) * height)
    x2 = int(max(xs) * width)
    y2 = int(max(ys) * height)

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width < 80 or face_height < 80:
        return None

    padding_x = int(face_width * 0.15)
    padding_y = int(face_height * 0.15)

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(width, x2 + padding_x)
    y2 = min(height, y2 + padding_y)

    return x1, y1, x2, y2


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(frame):
    """Recognize a face using DeepFace FaceNet embeddings."""

    try:
        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )

        if not result:
            return "Unknown", None

        current_embedding = np.array(
            result[0]["embedding"],
            dtype=np.float32
        )

        distances = []

        for saved_embedding in shaik_embeddings:
            distance_value = np.linalg.norm(
                current_embedding - saved_embedding
            )

            distances.append(distance_value)

        distances.sort()

        # Use the three closest saved embeddings.
        top_matches = distances[:3]

        average_distance = float(
            np.mean(top_matches)
        )

        if average_distance < RECOGNITION_THRESHOLD:
            return "Shaik", average_distance

        return "Unknown", average_distance

    except Exception:
        return "Unknown", None


# ============================================================
# ATTENDANCE
# ============================================================

def mark_attendance(name):
    """Mark attendance once per person per day."""

    if name != "Shaik":
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame({
            "Name": [name],
            "Date": [today],
            "Time": [current_time],
            "Status": ["Present"]
        })

        df.to_csv(
            ATTENDANCE_FILE,
            index=False
        )

        print(f"Attendance marked for {name}")
        return True

    try:
        df = pd.read_csv(ATTENDANCE_FILE)

    except Exception:
        df = pd.DataFrame(
            columns=[
                "Name",
                "Date",
                "Time",
                "Status"
            ]
        )

    already_marked = (
        (df["Name"].astype(str) == name) &
        (df["Date"].astype(str) == today)
    ).any()

    if already_marked:
        return False

    new_row = pd.DataFrame({
        "Name": [name],
        "Date": [today],
        "Time": [current_time],
        "Status": ["Present"]
    })

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    df.to_csv(
        ATTENDANCE_FILE,
        index=False
    )

    print(f"Attendance marked for {name}")

    return True


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# RUNTIME VARIABLES
# ============================================================

frame_count = 0
timestamp_ms = 0

recognition_count = 0

last_name = "Unknown"
last_distance = None

eyes_closed = False
liveness_verified = False

message = ""
message_until = 0


# ============================================================
# START MEDIAPIPE
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read webcam frame.")
            break

        frame_count += 1

        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # CONVERT TO RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # DETECT FACE
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        timestamp_ms += 33

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if not result.face_landmarks:

            last_name = "No Face"
            recognition_count = 0
            liveness_verified = False
            eyes_closed = False
            last_distance = None

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # ----------------------------------------------------
        # FACE FOUND
        # ----------------------------------------------------

        else:

            landmarks = result.face_landmarks[0]

            face_box = get_face_box(
                landmarks,
                width,
                height
            )

            if face_box is not None:

                x1, y1, x2, y2 = face_box

                # ------------------------------------------------
                # FACE BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # BLINK DETECTION
                # ------------------------------------------------

                ear = calculate_ear(landmarks)

                if ear < EAR_THRESHOLD:

                    eyes_closed = True

                elif eyes_closed:

                    eyes_closed = False
                    liveness_verified = True

                    print(
                        "Blink detected - liveness verified"
                    )

                # ------------------------------------------------
                # FACE RECOGNITION
                # ------------------------------------------------

                if (
                    liveness_verified
                    and
                    frame_count % RECOGNITION_INTERVAL == 0
                ):

                    name, distance_value = recognize_face(
                        frame
                    )

                    last_distance = distance_value

                    if name == "Shaik":

                        recognition_count += 1

                        if distance_value is not None:
                            print(
                                "Shaik match:",
                                round(distance_value, 2),
                                "| Count:",
                                recognition_count
                            )

                        if (
                            recognition_count
                            >= REQUIRED_MATCHES
                        ):
                            last_name = "Shaik"

                    else:

                        recognition_count = 0
                        last_name = "Unknown"

                        if distance_value is not None:
                            print(
                                "Unknown:",
                                round(distance_value, 2)
                            )

                # ------------------------------------------------
                # ATTENDANCE
                # ------------------------------------------------

                if (
                    last_name == "Shaik"
                    and
                    liveness_verified
                    and
                    recognition_count >= REQUIRED_MATCHES
                ):

                    marked = mark_attendance("Shaik")

                    if marked:

                        message = "Attendance Marked"

                        message_until = (
                            time.time() + 3
                        )

                        print(
                            "SUCCESS: Attendance marked"
                        )

                        # Require another blink before
                        # another recognition cycle.
                        liveness_verified = False
                        recognition_count = 0

                # ------------------------------------------------
                # LIVENESS STATUS
                # ------------------------------------------------

                if liveness_verified:

                    status = "Liveness: VERIFIED"

                else:

                    status = "Blink to Verify"

                cv2.putText(
                    frame,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # NAME
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    last_name,
                    (
                        x1,
                        max(y1 - 10, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # DISTANCE
                # ------------------------------------------------

                if last_distance is not None:

                    cv2.putText(
                        frame,
                        f"Distance: {last_distance:.2f}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

        # ----------------------------------------------------
        # ATTENDANCE MESSAGE
        # ----------------------------------------------------

        if time.time() < message_until:

            cv2.putText(
                frame,
                message,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            "ClassLens - AI Attendance",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()