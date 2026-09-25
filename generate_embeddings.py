import os
import pickle

import numpy as np
from deepface import DeepFace


DATASET_FOLDER = "dataset/Shaik"
OUTPUT_FILE = "shaik_embeddings.pkl"


def generate_embeddings():
    """Generate FaceNet embeddings from the Shaik dataset."""

    embeddings = []

    print("Generating Shaik embeddings...\n")

    for filename in sorted(os.listdir(DATASET_FOLDER)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(
            DATASET_FOLDER,
            filename
        )

        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=True
            )

            embedding = np.array(
                result[0]["embedding"],
                dtype=np.float32
            )

            embeddings.append(embedding)

            print(f"OK: {filename}")

        except Exception as error:
            print(f"FAILED: {filename} -> {error}")

    if not embeddings:
        print("\nNo embeddings generated.")
        return

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(embeddings, file)

    print(
        f"\nSuccessfully saved {len(embeddings)} embeddings."
    )
    print(f"File: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_embeddings()