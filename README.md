# ClassLens

### AI-Powered Face Recognition Attendance System

ClassLens is a real-time **AI attendance system** that uses computer vision and deep facial embeddings to automatically recognize students and record attendance through a webcam.

It combines **MediaPipe** for fast face detection with **DeepFace/FaceNet** for facial embedding generation and **Euclidean-distance matching** for identity recognition.

---

## ✨ Features

* 🎥 Live webcam enrollment
* 👤 Multi-angle face sample collection
* 🧠 FaceNet facial embeddings
* ⚡ MediaPipe face detection
* 🔍 Unknown-person detection
* ✅ Automated attendance
* 🛡️ Duplicate attendance prevention
* 📊 Streamlit dashboard
* 💾 CSV attendance records

---

## 🏗️ Architecture

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Face Detection
   ↓
Face Crop
   ↓
FaceNet Embedding
   ↓
Euclidean Distance
   ↓
Student / Unknown
   ↓
Attendance
   ↓
CSV
```

---

## 🧠 How It Works

### Enrollment

Students enroll using the live webcam while changing their face position and angle.

```text
Webcam
  ↓
Face Detection
  ↓
Multiple Face Samples
  ↓
FaceNet Embeddings
  ↓
Student Profile
```

Multiple embeddings help represent the student's face under different poses and conditions.

### Recognition

A live face is converted into an embedding and compared with enrolled embeddings using Euclidean distance.

```text
Live Face
   ↓
FaceNet
   ↓
Embedding
   ↓
Distance Matching
   ↓
Student / Unknown
```

If the distance satisfies the configured threshold, the student is recognized.

### Attendance

After recognition, ClassLens checks whether attendance has already been recorded for that student on the current date.

```text
Recognized
    ↓
Already Marked?
   ↙       ↘
 Yes        No
 ↓          ↓
Ignore    Mark Present
             ↓
            CSV
```

---

## 🛠️ Tech Stack

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python 3.12      | Application               |
| OpenCV           | Webcam & image processing |
| MediaPipe        | Face detection            |
| DeepFace         | Face recognition          |
| FaceNet          | Facial embeddings         |
| NumPy            | Numerical computation     |
| Streamlit        | Dashboard                 |
| Streamlit-WebRTC | Live webcam               |
| PyAV             | Video frames              |
| CSV              | Attendance storage        |

---

## 📁 Project Structure

```text
ClassLens/
│
├── streamlit_app.py
├── app_core.py
├── generate_embeddings.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── face_landmarker.task
│
└── screenshots/
```

Personal datasets, embeddings, and attendance records are excluded from GitHub.

---

## ⚙️ Setup

**Python:** 3.12

```bash
pip install -r requirements.txt
```

Make sure the MediaPipe model exists:

```text
models/face_landmarker.task
```

Run:

```bash
python -m streamlit run streamlit_app.py
```

---

## 🔑 Key Design Decisions

**MediaPipe instead of Haar Cascade**
Modern, efficient face detection suitable for real-time processing.

**FaceNet instead of raw image comparison**
Converts faces into numerical embeddings, making similarity comparison more robust to changes in pose, lighting, and expression.

**Multiple enrollment samples**
Provides the recognition system with more facial variation than a single image.

**Detection separated from recognition**
MediaPipe handles lightweight detection while FaceNet is used for the more expensive embedding step.

---

## 🚀 Future Improvements

* Database-backed student management
* Recognition threshold calibration
* Liveness / anti-spoofing
* Attendance analytics
* Authentication and role management
* Cloud deployment

---

## 💼 Resume Highlight

**ClassLens — AI Face Recognition Attendance System**
`Python | OpenCV | MediaPipe | DeepFace | FaceNet | Streamlit`

> Built a real-time AI attendance system using MediaPipe face detection and FaceNet facial embeddings, implementing multi-angle enrollment, Euclidean-distance recognition, unknown-person detection, and automated duplicate-safe attendance tracking.

