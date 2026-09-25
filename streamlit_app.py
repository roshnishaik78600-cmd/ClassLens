
import os
import csv
import pickle
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENTS_FILE = os.path.join(BASE_DIR, "students.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

SAMPLES_NEEDED = 20
RECOGNITION_THRESHOLD = 5.0


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def make_landmarker():

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )

    return FaceLandmarker.create_from_options(options)


def get_face(image, landmarker):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    h, w = image.shape[:2]

    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(w, max(xs))
    y2 = min(h, max(ys))

    width = x2 - x1
    height = y2 - y1

    if width < 100 or height < 100:
        return None

    # Add a little padding
    px = int(width * 0.12)
    py = int(height * 0.12)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    return image[y1:y2, x1:x2].copy()


# =========================================================
# STUDENT STORAGE
# =========================================================

def load_students():

    if not os.path.exists(STUDENTS_FILE):
        return {}

    try:

        with open(
            STUDENTS_FILE,
            "rb"
        ) as file:

            return pickle.load(file)

    except Exception:

        return {}


def save_students(students):

    with open(
        STUDENTS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            students,
            file
        )


# =========================================================
# FACENET
# =========================================================

def create_embedding(face):

    try:

        result = DeepFace.represent(
            img_path=face,
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False
        )

        embedding = np.array(
            result[0]["embedding"],
            dtype=np.float32
        )

        return embedding

    except Exception:

        return None


# =========================================================
# WEBCAM ENROLLMENT
# =========================================================

class EnrollmentProcessor(VideoProcessorBase):

    def __init__(self):

        self.samples = []
        self.frame_count = 0
        self.landmarker = make_landmarker()

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        display = image.copy()

        self.frame_count += 1

        # Capture every 5th frame
        if (
            self.frame_count % 5 == 0
            and len(self.samples) < SAMPLES_NEEDED
        ):

            face = get_face(
                image,
                self.landmarker
            )

            if face is not None:

                self.samples.append(
                    face
                )

        # Face detection for live display
        face = get_face(
            image,
            self.landmarker
        )

        if face is not None:

            cv2.putText(
                display,
                "FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "MOVE INTO CAMERA",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            display,
            f"Samples: {len(self.samples)}/{SAMPLES_NEEDED}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if len(self.samples) >= SAMPLES_NEEDED:

            cv2.putText(
                display,
                "READY - CLICK SAVE STUDENT",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "Slowly move: left / right / up / down",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            display,
            format="bgr24"
        )


# =========================================================
# ATTENDANCE
# =========================================================

def mark_attendance(name):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # Create CSV if necessary
    if not os.path.exists(ATTENDANCE_FILE):

        with open(
            ATTENDANCE_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "Name",
                    "Date",
                    "Time",
                    "Status"
                ]
            )

    # Check duplicate
    with open(
        ATTENDANCE_FILE,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["Name"] == name
                and row["Date"] == today
            ):

                return False

    now = datetime.now()

    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                name,
                today,
                now.strftime("%H:%M:%S"),
                "Present"
            ]
        )

    return True


# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="ClassLens",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 ClassLens")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Enroll Student",
        "Mark Attendance",
        "View Records"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "Home":

    students = load_students()

    st.header(
        "AI Face Recognition Attendance"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Students",
        len(students)
    )

    col2.metric(
        "Recognition",
        "FaceNet"
    )

    col3.metric(
        "Face Detection",
        "MediaPipe"
    )

    st.write(
        "Live facial recognition with FaceNet "
        "embeddings and automated attendance."
    )


# =========================================================
# ENROLL STUDENT
# =========================================================

elif page == "Enroll Student":

    st.header("👤 Enroll Student")

    name = st.text_input(
        "Student Name"
    ).strip()

    if not name:

        st.info(
            "Enter the student's name."
        )

    else:

        st.info(
            "Start the webcam. Keep your face visible "
            "and slowly move left, right, up and down."
        )

        ctx = webrtc_streamer(
            key="classlens-enrollment",
            video_processor_factory=EnrollmentProcessor,
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480,
                    "frameRate": 20
                },
                "audio": False
            },
            async_processing=True
        )

        if ctx.video_processor:

            count = len(
                ctx.video_processor.samples
            )

            st.write(
                f"Captured: {count}/{SAMPLES_NEEDED}"
            )

            if count >= SAMPLES_NEEDED:

                st.success(
                    "Enough face samples captured."
                )

                if st.button(
                    "Save Student",
                    type="primary"
                ):

                    samples = list(
                        ctx.video_processor.samples
                    )

                    embeddings = []

                    progress = st.progress(0)

                    student_folder = os.path.join(
                        DATASET_DIR,
                        name
                    )

                    os.makedirs(
                        student_folder,
                        exist_ok=True
                    )

                    for i, face in enumerate(samples):

                        # Save face sample
                        cv2.imwrite(
                            os.path.join(
                                student_folder,
                                f"{i + 1}.jpg"
                            ),
                            face
                        )

                        # Generate FaceNet embedding
                        embedding = create_embedding(
                            face
                        )

                        if embedding is not None:

                            embeddings.append(
                                embedding
                            )

                        progress.progress(
                            (i + 1) / len(samples)
                        )

                    if len(embeddings) >= 5:

                        students = load_students()

                        students[name] = embeddings

                        save_students(
                            students
                        )

                        st.success(
                            f"{name} enrolled successfully "
                            f"with {len(embeddings)} embeddings."
                        )

                    else:

                        st.error(
                            "Face embeddings could not be "
                            "generated reliably. Please enroll again."
                        )


# =========================================================
# MARK ATTENDANCE
# =========================================================

elif page == "Mark Attendance":

    st.header("📸 Mark Attendance")

    students = load_students()

    if not students:

        st.warning(
            "No students enrolled."
        )

    else:

        picture = st.camera_input(
            "Look directly at the camera"
        )

        if picture:

            data = np.frombuffer(
                picture.getvalue(),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                data,
                cv2.IMREAD_COLOR
            )

            landmarker = make_landmarker()

            face = get_face(
                image,
                landmarker
            )

            landmarker.close()

            if face is None:

                st.error(
                    "No clear face detected."
                )

            else:

                with st.spinner(
                    "Recognizing..."
                ):

                    embedding = create_embedding(
                        face
                    )

                if embedding is None:

                    st.error(
                        "Face embedding failed."
                    )

                else:

                    best_name = "Unknown"
                    best_distance = float("inf")

                    # Compare against every student's
                    # enrolled embeddings
                    for student, saved_embeddings in students.items():

                        distances = []

                        for saved in saved_embeddings:

                            saved = np.asarray(
                                saved,
                                dtype=np.float32
                            )

                            distance = np.linalg.norm(
                                embedding - saved
                            )

                            distances.append(
                                distance
                            )

                        distances.sort()

                        # Use closest 3 samples
                        closest = distances[:3]

                        student_distance = float(
                            np.mean(closest)
                        )

                        if student_distance < best_distance:

                            best_distance = (
                                student_distance
                            )

                            best_name = student

                    if (
                        best_name != "Unknown"
                        and best_distance < RECOGNITION_THRESHOLD
                    ):

                        st.success(
                            f"Recognized: {best_name}"
                        )

                        st.write(
                            f"Face distance: "
                            f"{best_distance:.2f}"
                        )

                        if mark_attendance(
                            best_name
                        ):

                            st.success(
                                "✅ Attendance marked."
                            )

                        else:

                            st.info(
                                "Attendance already marked "
                                "for today."
                            )

                    else:

                        st.error(
                            "❌ Unknown person"
                        )

                        st.write(
                            f"Closest distance: "
                            f"{best_distance:.2f}"
                        )


# =========================================================
# VIEW RECORDS
# =========================================================

elif page == "View Records":

    st.header("📊 Attendance Records")

    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        st.info(
            "No attendance records yet."
        )

    else:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline=""
        ) as file:

            records = list(
                csv.DictReader(file)
            )

        if records:

            st.dataframe(
                records,
                use_container_width=True
            )

        else:

            st.info(
                "No attendance records yet."
            )

