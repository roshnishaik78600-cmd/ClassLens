# 🎓 ClassLens

### AI-Powered Face Recognition Attendance System

ClassLens is an AI-powered attendance system that automatically **detects and recognizes students from camera input** and records attendance through an interactive **Streamlit dashboard**.

## ✨ Features

* 🎥 Real-time face detection with **OpenCV**
* 🧠 Face recognition using **DeepFace + FaceNet**
* 👤 Student enrollment and embedding generation
* ✅ Automated attendance marking
* 📊 Attendance record management
* 🌐 Interactive **Streamlit** dashboard
* 🔐 Local processing with sensitive biometric data excluded from Git

## 🏗️ Architecture

```text
Camera
  ↓
OpenCV Face Detection
  ↓
Face Extraction
  ↓
FaceNet Embedding
  ↓
Embedding Comparison
  ↓
Student Recognition
  ↓
Attendance Recording
```

## 🛠️ Tech Stack

**Python · OpenCV · DeepFace · FaceNet · NumPy · Streamlit · CSV · Pickle**

## 🚀 Run Locally

```bash
git clone https://github.com/roshnishaik78600-cmd/ClassLens.git
cd ClassLens

python -m venv classlens_env
classlens_env\Scripts\activate

pip install streamlit opencv-python numpy pandas deepface tensorflow tf-keras

python -m streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## 📂 Core Files

```text
streamlit_app.py          → Streamlit dashboard
app_core.py               → Core application logic
capture_faces.py          → Face image capture
detect.py                 → Face detection
generate_embeddings.py    → Face embedding generation
haarcascade_*.xml         → Haar Cascade face detector
```

## 🔮 Future Scope

* Multi-face recognition
* Liveness / anti-spoofing
* Database integration
* Attendance analytics
* Authentication and role-based access
* Cloud deployment

## 👨‍💻 Author

**Roshni Shaik**

[GitHub](https://github.com/roshnishaik78600-cmd)
