import streamlit as st
import cv2
import numpy as np
import pickle
import os
import csv
from datetime import datetime


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ClassLens",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STUDENT_FILE = os.path.join(
    BASE_DIR,
    "students.pkl"
)

OLD_EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "shaik_embeddings.pkl"
)

ATTENDANCE_FILE = os.path.join(
    BASE_DIR,
    "attendance.csv"
)

CASCADE_FILE = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

THRESHOLD = 5.0


# =========================================================
# LOAD OPENCV CASCADE
# =========================================================

face_cascade = None
CASCADE_OK = False

try:

    if os.path.exists(CASCADE_FILE):

        face_cascade = cv2.CascadeClassifier(
            CASCADE_FILE
        )

        if not face_cascade.empty():
            CASCADE_OK = True

except Exception:
    CASCADE_OK = False


# =========================================================
# DEEPFACE
# =========================================================

DeepFace = None
DEEPFACE_OK = False
DEEPFACE_ERROR = ""

try:

    from deepface import DeepFace

    DEEPFACE_OK = True

except Exception as e:

    DEEPFACE_ERROR = str(e)


# =========================================================
# LOAD STUDENTS
# =========================================================

def load_students():

    if os.path.exists(STUDENT_FILE):

        try:

            with open(
                STUDENT_FILE,
                "rb"
            ) as file:

                data = pickle.load(file)

            if isinstance(data, dict):
                return data

        except Exception:
            pass

    # Load old Shaik embeddings
    if os.path.exists(OLD_EMBEDDING_FILE):

        try:

            with open(
                OLD_EMBEDDING_FILE,
                "rb"
            ) as file:

                old_data = pickle.load(file)

            if isinstance(old_data, list):

                return {
                    "Shaik": old_data
                }

            if isinstance(old_data, np.ndarray):

                return {
                    "Shaik": [
                        old_data.tolist()
                    ]
                }

            return {
                "Shaik": [
                    old_data
                ]
            }

        except Exception:
            pass

    return {}


# =========================================================
# SAVE STUDENTS
# =========================================================

def save_students(students):

    try:

        with open(
            STUDENT_FILE,
            "wb"
        ) as file:

            pickle.dump(
                students,
                file
            )

        return True

    except Exception:
        return False


# =========================================================
# FACE DETECTION
# =========================================================

def detect_face(image):

    if image is None:
        return None

    if not CASCADE_OK:
        return None

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if faces is None:
            return None

        if len(faces) == 0:
            return None

        largest = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        x, y, w, h = largest

        padding = 30

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            image.shape[1],
            x + w + padding
        )

        y2 = min(
            image.shape[0],
            y + h + padding
        )

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            return None

        return face

    except Exception:
        return None


# =========================================================
# CREATE EMBEDDING
# =========================================================

def create_embedding(face):

    if face is None:
        return None

    if not DEEPFACE_OK:
        return None

    try:

        result = DeepFace.represent(
            img_path=face,
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False
        )

        if not result:
            return None

        embedding = np.array(
            result[0]["embedding"],
            dtype=np.float32
        )

        return embedding

    except Exception:
        return None


# =========================================================
# RECOGNITION
# =========================================================

def recognize_face(
    image,
    students
):

    face = detect_face(image)

    if face is None:
        return None, None

    embedding = create_embedding(face)

    if embedding is None:
        return None, None

    best_name = "Unknown"

    best_distance = float("inf")

    for name, saved_embeddings in students.items():

        if not isinstance(
            saved_embeddings,
            list
        ):

            saved_embeddings = [
                saved_embeddings
            ]

        for saved in saved_embeddings:

            try:

                saved = np.array(
                    saved,
                    dtype=np.float32
                )

                if saved.shape != embedding.shape:
                    continue

                distance = np.linalg.norm(
                    embedding - saved
                )

                if distance < best_distance:

                    best_distance = distance
                    best_name = name

            except Exception:
                continue

    if (
        best_name != "Unknown"
        and best_distance < THRESHOLD
    ):

        return (
            best_name,
            best_distance
        )

    return (
        "Unknown",
        best_distance
    )


# =========================================================
# ATTENDANCE
# =========================================================

def create_attendance_file():

    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        try:

            with open(
                ATTENDANCE_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Name",
                    "Date",
                    "Time",
                    "Status"
                ])

        except Exception:
            pass


def load_attendance():

    create_attendance_file()

    records = []

    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                records.append(row)

    except Exception:
        pass

    return records


def mark_attendance(name):

    records = load_attendance()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    for record in records:

        if (
            record.get("Name") == name
            and record.get("Date") == today
        ):

            return False

    try:

        with open(
            ATTENDANCE_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                name,
                today,
                datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "Present"
            ])

        return True

    except Exception:
        return False


# =========================================================
# INITIALIZATION
# =========================================================

students = load_students()

create_attendance_file()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 ClassLens")

st.sidebar.write(
    "AI-Powered Face Recognition Attendance"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📷 Mark Attendance",
        "👤 Enroll Student",
        "📊 View Records"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🎓 ClassLens")

    st.subheader(
        "AI-Powered Face Recognition Attendance System"
    )

    st.write(
        "Automatically recognize students "
        "and record classroom attendance."
    )

    st.divider()

    records = load_attendance()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    today_count = sum(
        1
        for record in records
        if record.get("Date") == today
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👥 Students",
            len(students)
        )

    with col2:

        st.metric(
            "📅 Present Today",
            today_count
        )

    with col3:

        if CASCADE_OK and DEEPFACE_OK:

            st.metric(
                "🟢 System",
                "Ready"
            )

        elif CASCADE_OK:

            st.metric(
                "🟡 System",
                "DeepFace Setup"
            )

        else:

            st.metric(
                "🔴 System",
                "Detector Error"
            )

    st.divider()

    st.info(
        "Use the sidebar to enroll students, "
        "mark attendance, or view records."
    )

    if not CASCADE_OK:

        st.error(
            "OpenCV face detector is unavailable."
        )

    if not DEEPFACE_OK:

        st.warning(
            "DeepFace is not currently available. "
            "Dashboard functions can still be viewed."
        )


# =========================================================
# ENROLL STUDENT
# =========================================================

elif page == "👤 Enroll Student":

    st.title("👤 Enroll Student")

    if not CASCADE_OK:

        st.error(
            "❌ Face detector is unavailable."
        )

        st.stop()

    if not DEEPFACE_OK:

        st.error(
            "❌ DeepFace is unavailable."
        )

        st.stop()

    name = st.text_input(
        "Student Name"
    )

    photo = st.camera_input(
        "Capture Student Face"
    )

    if photo is not None:

        data = photo.getvalue()

        array = np.frombuffer(
            data,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            st.error(
                "Could not read camera image."
            )

        else:

            st.image(
                frame,
                channels="BGR",
                caption="Captured Image"
            )

            face = detect_face(frame)

            if face is None:

                st.error(
                    "❌ Face not detected."
                )

            else:

                st.success(
                    "✅ Face detected!"
                )

                st.image(
                    face,
                    channels="BGR",
                    caption="Detected Face"
                )

                if st.button(
                    "💾 Enroll Student",
                    type="primary"
                ):

                    if not name.strip():

                        st.warning(
                            "Enter the student's name."
                        )

                    else:

                        with st.spinner(
                            "Creating face embedding..."
                        ):

                            embedding = create_embedding(
                                face
                            )

                        if embedding is None:

                            st.error(
                                "Could not create face embedding."
                            )

                        else:

                            clean_name = name.strip()

                            if clean_name not in students:

                                students[
                                    clean_name
                                ] = []

                            students[
                                clean_name
                            ].append(
                                embedding.tolist()
                            )

                            if save_students(students):

                                st.success(
                                    f"✅ {clean_name} "
                                    "enrolled successfully!"
                                )

                            else:

                                st.error(
                                    "Could not save student."
                                )


# =========================================================
# MARK ATTENDANCE
# =========================================================

elif page == "📷 Mark Attendance":

    st.title("📷 Mark Attendance")

    if not CASCADE_OK:

        st.error(
            "❌ Face detector is unavailable."
        )

        st.stop()

    if not DEEPFACE_OK:

        st.error(
            "❌ DeepFace is unavailable."
        )

        st.stop()

    if len(students) == 0:

        st.warning(
            "No students are enrolled."
        )

    photo = st.camera_input(
        "Capture Face"
    )

    if photo is not None:

        data = photo.getvalue()

        array = np.frombuffer(
            data,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            st.error(
                "Could not read camera image."
            )

        else:

            st.image(
                frame,
                channels="BGR",
                caption="Captured Image"
            )

            face = detect_face(frame)

            if face is None:

                st.error(
                    "❌ Face not detected."
                )

            else:

                st.success(
                    "✅ Face detected!"
                )

                st.image(
                    face,
                    channels="BGR",
                    caption="Detected Face"
                )

                if st.button(
                    "🔍 Recognize Face",
                    type="primary"
                ):

                    with st.spinner(
                        "Recognizing..."
                    ):

                        name, distance = recognize_face(
                            frame,
                            students
                        )

                    if name is None:

                        st.error(
                            "Face recognition failed."
                        )

                    elif name == "Unknown":

                        st.error(
                            "❌ Unknown Person"
                        )

                        if distance != float("inf"):

                            st.write(
                                f"Distance: {distance:.2f}"
                            )

                    else:

                        st.success(
                            f"✅ {name} recognized!"
                        )

                        st.write(
                            f"Distance: {distance:.2f}"
                        )

                        if mark_attendance(name):

                            st.success(
                                f"🟢 Attendance marked "
                                f"for {name}."
                            )

                        else:

                            st.info(
                                f"ℹ️ {name} is already "
                                "marked present today."
                            )


# =========================================================
# VIEW RECORDS
# =========================================================

elif page == "📊 View Records":

    st.title("📊 Attendance Records")

    records = load_attendance()

    if len(records) == 0:

        st.info(
            "No attendance records yet."
        )

    else:

        for record in records:

            name = record.get(
                "Name",
                ""
            )

            date = record.get(
                "Date",
                ""
            )

            time = record.get(
                "Time",
                ""
            )

            status = record.get(
                "Status",
                ""
            )

            st.write(
                f"👤 {name}   |   "
                f"📅 {date}   |   "
                f"⏰ {time}   |   "
                f"🟢 {status}"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Records",
                len(records)
            )

        unique_students = set()

        for record in records:

            if record.get("Name"):

                unique_students.add(
                    record.get("Name")
                )

        with col2:

            st.metric(
                "Students Present",
                len(unique_students)
            )