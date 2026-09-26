<div align="center">

# ClassLens

### Real-Time AI Face Recognition Attendance System

Automated, contactless attendance using computer vision and deep facial embeddings.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

[Overview](#overview) • [Features](#features) • [Architecture](#architecture) • [Setup](#setup) • [Tech Stack](#tech-stack) • [Roadmap](#roadmap)

</div>

---

## Overview

**ClassLens** is a real-time attendance system that identifies students through a webcam feed and automatically logs their presence — no ID cards, no manual roll calls, no proxy attendance.

It pairs **MediaPipe** for fast, lightweight face detection with **DeepFace / FaceNet** for generating facial embeddings, then uses distance-based similarity matching to identify enrolled students and mark attendance in real time.

The project was built to explore an end-to-end applied computer vision pipeline: detection → representation learning → similarity search → business logic (duplicate-safe attendance) → a usable interface.

---

## Features

| | |
|---|---|
| 🎥 | Live webcam-based face enrollment |
| 👤 | Multi-angle sample capture for robust student profiles |
| 🧠 | FaceNet embeddings for facial representation |
| ⚡ | MediaPipe for real-time, low-latency face detection |
| 🔍 | Unknown-person detection for unenrolled faces |
| 🛡️ | Basic liveness / anti-spoofing checks (blink + landmark analysis) |
| ✅ | Automated, duplicate-safe attendance marking |
| 📊 | Interactive Streamlit dashboard |
| 💾 | Lightweight CSV-based attendance storage |

---

## Architecture

The system is split into two independent pipelines — **enrollment** (building a student's facial profile) and **recognition** (matching a live face against that profile) — which converge at attendance validation.

```text
                         ┌────────────┐
                         │   Webcam   │
                         └─────┬──────┘
                               │
                     OpenCV / Streamlit-WebRTC
                               │
                     MediaPipe Face Detection
                               │
                          Face Crop
                               │
                     DeepFace / FaceNet
                               │
                       Facial Embedding
                               │
                     Distance-Based Matching
                               │
                    ┌──────────┴──────────┐
                    │                     │
              Known Student           Unknown
                    │
           Already marked today?
              ┌─────┴─────┐
             Yes           No
              │             │
           Ignore     Mark Present → Attendance CSV
```

### 1. Enrollment
A student's face is captured from multiple angles and expressions via webcam. Each sample is converted into a FaceNet embedding, and the set is stored as that student's reference profile — improving robustness to pose and lighting variation at recognition time.

### 2. Recognition
A live frame is passed through MediaPipe for detection, cropped, and embedded via FaceNet. The resulting vector is compared against all enrolled profiles using distance-based similarity to identify the closest match (or flag the face as unknown).

### 3. Attendance Validation
On a successful match, ClassLens checks whether the student has already been marked present that day before writing a new record — preventing duplicate entries from repeated camera passes.

### 4. Liveness Check
Before a recognition result is accepted, a lightweight liveness layer (facial landmarks + blink detection) helps filter out simple photo-based spoofing attempts.

> **Note:** Liveness detection here is a basic deterrent, not an enterprise-grade biometric security guarantee — see [Roadmap](#roadmap).

---

## Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | **Python** | Core application logic |
| Vision | **OpenCV** | Webcam capture and image processing |
| Detection | **MediaPipe** | Real-time face detection and landmarks |
| Recognition | **DeepFace / FaceNet** | Facial embedding generation |
| Numerical | **NumPy** | Vector operations and distance computation |
| Interface | **Streamlit** | Web dashboard |
| Streaming | **Streamlit-WebRTC** | Real-time browser-based webcam streaming |
| Video I/O | **PyAV** | Frame decoding/processing |
| Storage | **CSV** | Lightweight attendance persistence |

---

## Project Structure

```text
ClassLens/
│
├── models/
│   └── face_landmarker.task        # MediaPipe landmark model
│
├── streamlit_app.py                # Main dashboard entry point
├── app_core.py                     # Core application logic
├── enroll.py                       # Student enrollment pipeline
├── generate_embeddings.py          # Embedding generation
│
├── liveness_test.py                # Liveness/anti-spoofing checks
├── blink_test.py                   # Blink detection utility
├── test_landmarker.py              # Landmark detection tests
├── test_recognition.py             # Recognition accuracy tests
├── verify_embeddings.py            # Embedding integrity checks
├── live_deepface_test.py           # Live DeepFace pipeline test
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Personal datasets, generated embeddings, attendance records, and environment files are excluded from version control via `.gitignore`.

---

## Setup

### Prerequisites
- Python 3.9+
- A working webcam
- pip / virtualenv

### 1. Clone the repository
```bash
git clone https://github.com/roshnishaik78600-cmd/ClassLens.git
cd ClassLens
```

### 2. Create and activate a virtual environment
```bash
python -m venv classlens_env
```

**Windows (PowerShell):**
```powershell
.\classlens_env\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source classlens_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify the MediaPipe model
Ensure the following file is present:
```text
models/face_landmarker.task
```

### 5. Launch the app
```bash
python -m streamlit run streamlit_app.py
```
The dashboard will open automatically in your default browser.

---

## Key Design Decisions

**MediaPipe for detection, FaceNet for recognition.**
Detection and recognition are deliberately decoupled: MediaPipe handles the cheap, high-frequency task of locating a face in the frame, while the more expensive FaceNet embedding step only runs on confirmed face crops. This keeps the pipeline responsive in real time.

**Embeddings over raw image comparison.**
Rather than comparing pixels, faces are projected into a fixed-length embedding space where distance correlates with facial similarity — enabling fast, scalable matching that generalizes across lighting and minor pose changes.

**Multi-sample enrollment.**
Capturing several samples per student at enrollment reduces sensitivity to any single angle, expression, or lighting condition, improving real-world recognition accuracy.

**Duplicate-safe attendance logic.**
Attendance writes are gated behind a same-day lookup, so a student walking past the camera multiple times is only marked present once.

---

## Roadmap

- [ ] Database-backed student management (replace CSV with a proper DB)
- [ ] Automatic recognition-threshold calibration
- [ ] Stronger, production-grade liveness / anti-spoofing detection
- [ ] Attendance analytics and reporting dashboard
- [ ] Authentication and role-based access control
- [ ] Cloud deployment (Docker + managed hosting)
- [ ] Batch / multi-student enrollment workflow

---


## License

This project is available under the [MIT License](LICENSE).

<div align="center">

Built with ❤️ by [Roshni Shaik](https://github.com/roshnishaik78600-cmd)

</div>
