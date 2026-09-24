from deepface import DeepFace

result = DeepFace.represent(
    img_path="dataset/Shaik/1.jpg",
    model_name="Facenet",
    detector_backend="opencv"
)

embedding = result[0]["embedding"]

print("Embedding generated!")
print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])