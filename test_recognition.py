import cv2
import pickle
import numpy as np
from deepface import DeepFace


# Load saved embeddings
with open("shaik_embeddings.pkl", "rb") as file:
    saved_embeddings = pickle.load(file)

print("Saved embeddings:", len(saved_embeddings))


# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    raise SystemExit


print("\nLook at the camera.")
print("Press SPACE to test your face.")
print("Press Q to quit.\n")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow(
        "Recognition Calibration",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    if key == 32:  # SPACE

        print("\nGenerating FaceNet embedding...")

        try:

            result = DeepFace.represent(
                img_path=frame,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=True
            )


            current = np.array(
                result[0]["embedding"],
                dtype=np.float32
            )


            distances = []


            for saved in saved_embeddings:

                saved = np.array(
                    saved,
                    dtype=np.float32
                )


                distance = np.linalg.norm(
                    current - saved
                )


                distances.append(distance)


            distances.sort()


            print("\nYour FaceNet distances:")

            print(
                [round(x, 3) for x in distances]
            )


            print(
                "\nMinimum:",
                round(min(distances), 3)
            )


            print(
                "Top 3 average:",
                round(
                    np.mean(distances[:3]),
                    3
                )
            )


            print(
                "Maximum:",
                round(max(distances), 3)
            )


        except Exception as error:

            print(
                "\nRecognition error:",
                error
            )


cap.release()

cv2.destroyAllWindows()