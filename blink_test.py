import cv2
import mediapipe as mp
import math

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 1

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Left eye
            left_vertical = distance(
                landmarks[159],
                landmarks[145]
            )

            left_horizontal = distance(
                landmarks[33],
                landmarks[133]
            )

            left_ear = left_vertical / left_horizontal

            # Right eye
            right_vertical = distance(
                landmarks[386],
                landmarks[374]
            )

            right_horizontal = distance(
                landmarks[362],
                landmarks[263]
            )

            right_ear = right_vertical / right_horizontal

            ear = (left_ear + right_ear) / 2

            if ear < 0.20:
                text = "BLINK"
            else:
                text = "EYES OPEN"

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Blink Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()