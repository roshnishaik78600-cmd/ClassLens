# 🎓 ClassLens

### AI-Powered Face Recognition Attendance System

ClassLens is an AI-powered attendance management system that uses **computer vision and deep learning** to detect and recognize students from camera input and automatically record attendance.

The application combines **OpenCV, DeepFace, FaceNet, NumPy, Pandas and Streamlit** to provide an interactive local dashboard for student enrollment, face recognition and attendance management.

---

## 🚀 Overview

Traditional attendance systems can be time-consuming and require manual record keeping.

ClassLens automates this process through a computer-vision pipeline:

```text
Webcam
   ↓
Face Detection
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
   ↓
Streamlit Dashboard
```

The system is designed to demonstrate how **AI + Computer Vision + Web Applications** can be combined into a practical real-world application.

---

## ✨ Features

* 🎥 Real-time webcam-based face detection
* 🧠 Deep learning-based face recognition
* 🔬 Face embeddings using FaceNet
* 👤 Student enrollment
* 📸 Face image capture
* 🔍 Face detection using OpenCV
* ✅ Automatic attendance marking
* 📊 Attendance record management
* 🌐 Interactive Streamlit dashboard
* 💾 Local attendance storage
* 🔐 Local processing of biometric information
* 🧩 Modular Python project structure

---

## 🏗️ System Architecture

```text
                 ┌───────────────┐
                 │    Webcam     │
                 └───────┬───────┘
                         │
                         ▼
              ┌────────────────────┐
              │ OpenCV Face         │
              │ Detection           │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Face Extraction    │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ FaceNet Embedding  │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Embedding          │
              │ Comparison         │
              └─────────┬──────────┘
                        │
                 ┌──────┴──────┐
                 │             │
              Match          No Match
                 │             │
                 ▼             ▼
          ┌────────────┐   ┌─────────┐
          │ Recognized │   │ Unknown │
          │  Student   │   │  Face   │
          └─────┬──────┘   └─────────┘
                │
                ▼
        ┌─────────────────┐
        │ Attendance      │
        │ Recording       │
        └─────────────────┘
```

---

## 🧠 How It Works

### 1. Student Enrollment

A student's face is captured through the camera.

```text
Student
   ↓
Camera
   ↓
Face Detection
   ↓
Face Image
   ↓
Face Embedding
   ↓
Stored Representation
```

The resulting facial representation can later be used for recognition.

### 2. Face Detection

ClassLens uses **OpenCV** for detecting faces in camera frames.

The Haar Cascade classifier identifies regions of an image that contain faces.

### 3. Face Recognition

Detected faces are processed through **DeepFace / FaceNet** to generate numerical face embeddings.

Conceptually:

```text
Face Image
    ↓
Deep Learning Model
    ↓
Embedding Vector
```

The generated embedding is compared with previously enrolled student embeddings.

### 4. Recognition

If the embedding is sufficiently similar to an enrolled representation:

```text
Face → Student Identified
```

Otherwise:

```text
Face → Unknown
```

### 5. Attendance

When a student is recognized, ClassLens records the attendance information.

---

## 🛠️ Tech Stack

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Core programming language          |
| OpenCV       | Computer vision and face detection |
| DeepFace     | Face recognition framework         |
| FaceNet      | Facial embedding generation        |
| TensorFlow   | Deep learning backend              |
| NumPy        | Numerical operations               |
| Pandas       | Attendance data handling           |
| Streamlit    | Interactive web dashboard          |
| CSV / Pickle | Local data storage                 |

---

## 📂 Project Structure

```text
ClassLens/
│
├── dataset/
│
├── screenshots/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── streamlit_app.py
├── app_core.py
├── capture_faces.py
├── detect.py
├── encode_faces.py
├── generate_embedding.py
├── generate_embeddings.py
├── webcam.py
│
└── haarcascade_frontalface_default.xml
```

### Core Files

| File                                  | Description                    |
| ------------------------------------- | ------------------------------ |
| `streamlit_app.py`                    | Main Streamlit dashboard       |
| `app_core.py`                         | Core application functionality |
| `capture_faces.py`                    | Captures student face images   |
| `detect.py`                           | Face detection functionality   |
| `encode_faces.py`                     | Face encoding functionality    |
| `generate_embedding.py`               | Generates face embeddings      |
| `generate_embeddings.py`              | Processes multiple embeddings  |
| `webcam.py`                           | Webcam-related functionality   |
| `haarcascade_frontalface_default.xml` | Haar Cascade face detector     |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/roshnishaik78600-cmd/ClassLens.git
```

### 2. Enter the project directory

```bash
cd ClassLens
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv classlens_env
```

### 4. Activate the environment

```bash
classlens_env\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start ClassLens

```bash
python -m streamlit run streamlit_app.py
```

### 7. Open the application

Streamlit will provide a local address, typically:

```text
http://localhost:8501
```

---

## 📸 Screenshots

Add screenshots of the working application here.

### Dashboard

![ClassLens Dashboard](screenshots/dashboard.png)

### Student Enrollment

![Student Enrollment](screenshots/enrollment.png)

### Attendance

![Attendance Dashboard](screenshots/attendance.png)

> Replace the screenshot filenames above with the actual images you add to the repository.

---

## 🔐 Privacy & Security

ClassLens is intended as a local demonstration project.

Face images, generated embeddings and attendance records may contain sensitive information and **should not be committed to a public GitHub repository**.

The repository therefore uses `.gitignore` rules to prevent sensitive or generated files from being uploaded.

For a production deployment, additional safeguards would be required, including:

* Secure authentication
* Encrypted biometric storage
* Database access controls
* Role-based authorization
* Data retention policies
* Consent management
* Liveness / anti-spoofing
* Secure API communication
* Audit logging

---

## 🔮 Future Scope

Potential improvements include:

* 👥 Multi-face recognition
* 🛡️ Liveness detection / anti-spoofing
* 🗄️ Database integration
* 📊 Advanced attendance analytics
* 🔐 Authentication and role-based access control
* ☁️ Cloud deployment
* 📱 Mobile-friendly interface
* 📈 Attendance reports and visualizations
* 🔔 Automated attendance notifications
* 🧠 Improved recognition accuracy
* ⚡ Real-time multi-student processing

---

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Computer Vision
* Face Detection
* Face Recognition
* Deep Learning
* Face Embeddings
* Python Application Development
* Streamlit
* Data Processing
* Real-time Webcam Processing
* Local AI/ML deployment

---

## ⚠️ Disclaimer

ClassLens is an educational and portfolio project demonstrating face recognition and automated attendance concepts.

It should not be deployed in real-world educational environments without appropriate testing, security controls, privacy protections, consent mechanisms and compliance with applicable laws and institutional policies.

---

## 👨‍💻 Author

### Roshni Shaik

GitHub:
https://github.com/roshnishaik78600-cmd

---

## ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

