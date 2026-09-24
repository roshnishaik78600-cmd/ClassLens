import cv2
import os

# Person's name
name = "Shaik"

# Create folder if it doesn't exist
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for x, y, w, h in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Save face image
        if count < 10:
            filename = f"{save_path}/{count + 1}.jpg"
            cv2.imwrite(filename, face)
            count += 1

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    # Show counter
    cv2.putText(
        frame,
        f"Images: {count}/10",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ClassLens - Capture Faces", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} images for {name}")