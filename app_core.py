import cv2
import pickle
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from deepface import DeepFace


# ==========================================
# 1. LOAD SHAIK'S SAVED EMBEDDINGS
# ==========================================

with open("shaik_embeddings.pkl", "rb") as file:
    shaik_embeddings = pickle.load(file)

print("Loaded embeddings:", len(shaik_embeddings))


# ==========================================
# 2. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()


# ==========================================
# 3. LOAD FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# 4. VARIABLES
# ==========================================

frame_count = 0

# Most recent recognition result
last_name = "Unknown"

# Recognition threshold
threshold = 5

# Attendance message
attendance_message = ""
attendance_message_until = 0


# ==========================================
# 5. MARK ATTENDANCE FUNCTION
# ==========================================

def mark_attendance(name):

    # Don't mark unknown people
    if name == "Unknown":
        return False

    # Get current date and time
    now = datetime.now()

    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Attendance file
    file_name = "attendance.csv"


    # --------------------------------------
    # If CSV doesn't exist, create it
    # --------------------------------------

    if not os.path.exists(file_name):

        data = {
            "Name": [name],
            "Date": [today],
            "Time": [current_time],
            "Status": ["Present"]
        }

        df = pd.DataFrame(data)

        df.to_csv(
            file_name,
            index=False
        )

        print("Attendance marked for", name)

        return True


    # --------------------------------------
    # Read existing attendance
    # --------------------------------------

    df = pd.read_csv(file_name)


    # --------------------------------------
    # Check if already marked today
    # --------------------------------------

    already_marked = (
        (df["Name"] == name) &
        (df["Date"] == today)
    ).any()


    if already_marked:

        print(name, "already marked today")

        return False


    # --------------------------------------
    # Create new attendance row
    # --------------------------------------

    new_row = pd.DataFrame({
        "Name": [name],
        "Date": [today],
        "Time": [current_time],
        "Status": ["Present"]
    })


    # Add new row
    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )


    # Save CSV
    df.to_csv(
        file_name,
        index=False
    )


    print("Attendance marked for", name)

    return True


# ==========================================
# 6. MAIN WEBCAM LOOP
# ==========================================

while True:

    # Get webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break


    # Increase frame counter
    frame_count += 1


    # ======================================
    # 7. FACE DETECTION
    #    EVERY FRAME
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )


    # ======================================
    # 8. PROCESS EACH FACE
    # ======================================

    for x, y, w, h in faces:


        # ----------------------------------
        # Draw face rectangle
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ==================================
        # 9. RECOGNITION
        #    EVERY 3RD FRAME
        # ==================================

        if frame_count % 3 == 0:


            # Crop face
            face = frame[
                y:y + h,
                x:x + w
            ]


            try:

                # Generate FaceNet embedding
                result = DeepFace.represent(
                    img_path=face,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )


                current_embedding = np.array(
                    result[0]["embedding"]
                )


                # ----------------------------------
                # Compare with saved embeddings
                # ----------------------------------

                distances = []


                for saved_embedding in shaik_embeddings:

                    saved_embedding = np.array(
                        saved_embedding
                    )


                    distance = np.linalg.norm(
                        current_embedding -
                        saved_embedding
                    )


                    distances.append(distance)


                # Find closest embedding
                min_distance = min(distances)


                # ==================================
                # 10. RECOGNITION DECISION
                # ==================================

                if min_distance < threshold:

                    last_name = "Shaik"


                    # ----------------------------------
                    # Mark attendance
                    # ----------------------------------

                    marked = mark_attendance(
                        last_name
                    )


                    # ----------------------------------
                    # Show message if newly marked
                    # ----------------------------------

                    if marked:

                        attendance_message = (
                            "Attendance Marked"
                        )

                        attendance_message_until = (
                            time.time() + 2
                        )


                else:

                    last_name = "Unknown"


                # Print distance
                print(
                    "Frame:",
                    frame_count,
                    "| Distance:",
                    round(min_distance, 2),
                    "| Result:",
                    last_name
                )


            except Exception as e:

                print(
                    "Recognition error:",
                    e
                )

                last_name = "Unknown"


        # ==================================
        # 11. DISPLAY NAME
        # ==================================

        cv2.putText(
            frame,
            last_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ======================================
    # 12. ATTENDANCE MESSAGE
    # ======================================

    if time.time() < attendance_message_until:

        cv2.putText(
            frame,
            attendance_message,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    # ======================================
    # 13. DISPLAY WEBCAM
    # ======================================

    cv2.imshow(
        "ClassLens - Attendance",
        frame
    )


    # ======================================
    # 14. PRESS Q TO EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 15. CLEAN UP
# ==========================================

cap.release()

cv2.destroyAllWindows()