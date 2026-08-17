# Advertisement Gaze Analytics 👁️

A deep learning based computer vision system for detecting, tracking, and analyzing viewers' gaze and attention toward digital advertisements.

This project combines MediaPipe Face Landmarker and L2CS-Net to estimate facial landmarks, head pose, and gaze direction in real time. The system then maps the estimated gaze to the screen, detects attention sessions, stores gaze events, and generates analytical data such as gaze points and heatmaps.

More than just detecting whether someone is looking at a screen, the goal is to understand **where people look, for how long they look, and how much attention an advertisement receives**.

---

## See it live and in action 📺 - Click the image!

Link to be added.

# Setup 🪛

### 1. Clone the repository

```bash
git clone https://github.com/your-username/advertisement-gaze-analytics.git
cd advertisement-gaze-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\\Scripts\\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

# Models 🧠

This project uses pretrained deep learning models instead of training the complete system from scratch.

### MediaPipe Face Landmarker

Used for:

- Face detection
- 478 facial landmarks
- Eye landmarks
- Facial transformation matrix
- Face geometry
- Head pose estimation

Place the model here:

```text
models/mediapipe/face_landmarker.task
```

### L2CS-Net

Used for deep learning based gaze estimation.

The model predicts:

- Yaw
- Pitch

Place the pretrained model here:

```text
models/l2cs/L2CSNet_gaze360.pkl
```

# Running 🚀

## Run with Webcam

```bash
python main.py
```

By default the application uses camera index `0`.

If your machine has multiple cameras, specify the camera:

```bash
python main.py --camera 1
```

## Run with a Video

```bash
python main.py --source video --video data/input/test.mp4
```

## Run without Face Recognition

For testing the gaze pipeline without the additional recognition model:

```bash
python main.py --no-recognition
```

This is useful when testing:

- Face Landmarks
- Head Pose
- L2CS
- Gaze Fusion
- Attention

# What the system does 🔍

The complete processing pipeline is:

```text
Camera / Video
      ↓
MediaPipe Face Landmarker
      ↓
Face Landmarks
      ↓
Head Pose Estimation
      ↓
Face Tracking
      ↓
Face Recognition
      ↓
L2CS-Net
      ↓
Gaze Yaw / Pitch
      ↓
Gaze Fusion
      ↓
Screen Calibration
      ↓
Gaze Point
      ↓
Attention Analysis
      ↓
Look Session
      ↓
Analytics
      ↓
SQLite / Heatmap
```

# Face Detection & Landmarks 👤

MediaPipe Face Landmarker is responsible for extracting detailed facial information.

The system uses:

- 478 3D facial landmarks
- Eye landmarks
- Facial transformation matrix
- Head orientation information

These features are used by the gaze estimation pipeline.

# Gaze Estimation 👁️

L2CS-Net is used as the main gaze estimation model.

The model receives a face crop and predicts:

```text
Yaw
Pitch
```

The predicted angles are then converted into a gaze direction vector.

```text
Face
 ↓
L2CS-Net
 ↓
Yaw + Pitch
 ↓
Gaze Direction
```

The project uses the pretrained Gaze360 model provided by L2CS-Net.

# Gaze Fusion 🔀

Gaze estimation is combined with head pose information.

```text
L2CS Gaze
    +
Head Pose
    ↓
Gaze Fusion
    ↓
Final Gaze Direction
```

This provides a more stable representation of the viewer's attention direction than relying on a single signal.

# Screen Calibration 🎯

The system uses a calibration stage to map gaze direction to normalized screen coordinates.

Example:

```text
Gaze Direction
      ↓
Calibration Model
      ↓
Screen Coordinate
      ↓
(x, y)
```

The predicted gaze point is represented using normalized coordinates:

```text
x = 0.0 → 1.0
y = 0.0 → 1.0
```

Calibration data can be stored in:

```text
data/calibration/screen_calibration.json
```

# Attention Analysis 🧠

The system does not consider every detected face as a viewer looking at the advertisement.

Attention is determined using several signals:

- Gaze direction
- Gaze confidence
- Head pose
- Screen gaze point
- Attention score

The result is:

```text
LOOKING
```

or:

```text
NOT LOOKING
```

# Look Sessions ⏱️

Continuous attention frames are grouped into a single look session.

For example:

```text
Frame 1 → Looking
Frame 2 → Looking
Frame 3 → Looking
Frame 4 → Looking
Frame 5 → Not Looking
```

can become:

```text
Look Session
Start: 10.2s
End:   14.7s
Duration: 4.5s
```

Short interruptions are tolerated so that one continuous look is not split into multiple events.

# Face Tracking 🎯

Every detected face receives a temporary `Track ID`.

```text
Face
 ↓
Tracker
 ↓
Track ID
```

Example:

```text
Track ID = 7
```

The Track ID is intentionally different from the persistent identity assigned by face recognition.

```text
Track ID
   ↓
Face Recognition
   ↓
Person ID
```

# Face Recognition 🧑

InsightFace / ArcFace is used to generate face embeddings.

```text
Face
 ↓
ArcFace
 ↓
Embedding
 ↓
Similarity Search
 ↓
Person ID
```

Known identities are stored in:

```text
data/faces/gallery.json
```

Face recognition can also be disabled during development:

```bash
python main.py --no-recognition
```

# Database 💾

The system uses SQLite to store analytical data.

Main tables include:

```text
persons
ads
analytics_sessions
look_events
gaze_points
person_statistics
ad_statistics
```

The database is created automatically when the application starts.

Default location:

```text
data/output/gaze_analytics.db
```

# Gaze Heatmaps 🔥

Every valid calibrated gaze point can be stored in the database.

```text
Gaze Points
     ↓
Density Estimation
     ↓
Heatmap
```

Heatmaps can be generated for:

- A specific advertisement
- A specific person
- An entire dataset

Example output:

```text
Advertisement
      ↓
Gaze Points
      ↓
Heatmap
```

Generated heatmaps can be stored in:

```text
data/output/
```

# Project Structure 📁

```text
advertisement-gaze-analytics/
│
├── main.py
│
├── src/
│   ├── config/
│   ├── input/
│   ├── vision/
│   ├── gaze/
│   ├── tracking/
│   ├── recognition/
│   ├── calibration/
│   ├── attention/
│   ├── analytics/
│   ├── storage/
│   └── visualization/
│
├── models/
│   ├── mediapipe/
│   └── l2cs/
│
├── data/
│   ├── input/
│   ├── calibration/
│   ├── faces/
│   └── output/
│
├── tests/
│
├── scripts/
│
├── requirements.txt
└── README.md
```

# Technologies 🛠️

- Python
- PyTorch
- OpenCV
- MediaPipe
- L2CS-Net
- InsightFace
- ArcFace
- ONNX Runtime
- NumPy
- SQLite

# Current Features ✅

- Real-time webcam processing
- Video file processing
- Face detection
- 478 facial landmarks
- Head pose estimation
- Deep learning based gaze estimation
- Multi-face tracking
- Face recognition
- Person identification
- Gaze fusion
- Screen calibration
- Gaze point estimation
- Attention detection
- Look session management
- SQLite storage
- Gaze point storage
- Advertisement statistics
- Person statistics
- Gaze heatmaps
- Real-time visualization

# Development Roadmap 🚧

### Phase 1 — Core Gaze Pipeline

- MediaPipe integration
- L2CS-Net integration
- Head pose
- Gaze fusion
- Real-time visualization

### Phase 2 — Attention Analytics

- Screen calibration
- Gaze point estimation
- Attention scoring
- Look sessions
- Advertisement events

### Phase 3 — Recognition & Tracking

- Face recognition
- Persistent identities
- Improved multi-object tracking
- Identity stability

### Phase 4 — Advanced Analytics

- Advertisement heatmaps
- ROI analysis
- Attention over time
- Viewer comparison
- Advertisement ranking

### Phase 5 — Optimization

- GPU inference
- ONNX optimization
- Faster face recognition
- Better tracking
- Improved gaze calibration

# Great Resources 📚

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)
- [InsightFace](https://github.com/deepinsight/insightface)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

# Important Notes ⚠️

This project currently uses pretrained models for face landmarks, face recognition, and gaze estimation.

The gaze estimation pipeline is designed as a modular system so the pretrained L2CS-Net model can later be replaced or fine-tuned with a custom gaze model and project-specific dataset.

The accuracy of screen gaze estimation depends on factors such as:

- Camera position
- Camera quality
- Lighting
- Viewer distance
- Head orientation
- Screen position
- Calibration quality

Therefore, the system should be evaluated using real-world test data before being used for production analytics.

# Who, When, Why?

👨‍💻 Author: Javid

📅 Version: 1.x

🎯 Purpose: Advertisement Gaze & Attention Analytics

🧠 Main Models: MediaPipe Face Landmarker + L2CS-Net

📜 License: To be added
