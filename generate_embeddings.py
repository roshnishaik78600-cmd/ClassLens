from deepface import DeepFace
import os
import pickle

# Folder containing Shaik's face images
dataset_path = "dataset/Shaik"

# Store all embeddings here
embeddings = []

# Go through every image in the folder
for filename in os.listdir(dataset_path):

    if filename.endswith(".jpg"):

        image_path = os.path.join(dataset_path, filename)

        print("Processing:", filename)

        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend="opencv"
        )

        embedding = result[0]["embedding"]

        embeddings.append(embedding)

# Save embeddings
with open("shaik_embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)

print("Done!")
print("Total embeddings:", len(embeddings))