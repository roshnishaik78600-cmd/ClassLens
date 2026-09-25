import cv2
from deepface import DeepFace


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    raise SystemExit


print("Look at the camera.")
print("Press SPACE to test.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow(
        "DeepFace Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    if key == 32:

        print("\nTesting your face...")

        try:

            result = DeepFace.represent(
                img_path=frame,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=True
            )

            print(
                "SUCCESS - FaceNet embedding generated."
            )

            print(
                "Embedding length:",
                len(result[0]["embedding"])
            )

        except Exception as error:

            print(
                "ERROR:",
                error
            )


cap.release()

cv2.destroyAllWindows()