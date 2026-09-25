import os
import pickle
import numpy as np
from deepface import DeepFace


DATASET_FOLDER = "dataset/Shaik"


# Load saved embeddings
with open("shaik_embeddings.pkl", "rb") as file:
    saved_embeddings = pickle.load(file)


print("Saved embeddings:", len(saved_embeddings))
print()


for filename in sorted(os.listdir(DATASET_FOLDER)):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
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


        current_embedding = np.array(
            result[0]["embedding"],
            dtype=np.float32
        )


        distances = []


        for saved_embedding in saved_embeddings:

            saved_embedding = np.array(
                saved_embedding,
                dtype=np.float32
            )


            distance = np.linalg.norm(
                current_embedding -
                saved_embedding
            )


            distances.append(distance)


        distances.sort()


        print(
            filename,
            "| Minimum:",
            round(distances[0], 3),
            "| Top 3 avg:",
            round(
                np.mean(distances[:3]),
                3
            )
        )


    except Exception as error:

        print(
            filename,
            "| ERROR:",
            error
        )