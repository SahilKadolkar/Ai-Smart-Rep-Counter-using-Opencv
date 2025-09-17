# AI Smart Rep Counter using OpenCV

This project is an **AI-powered exercise tracking system** that detects and counts repetitions for various exercises like bicep curls, squats, push-ups, lateral raises, and shoulder presses using **OpenCV** and **MediaPipe**. It provides real-time feedback with rep counts, exercise stage, and smooth angle measurements.

---

## 📂 Dataset & Input

- Real-time webcam video feed is used as input.  
- No static dataset required; the system processes your body pose live using MediaPipe Pose landmarks.

---

## 🔧 Tools & Libraries

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Flask (for web interface)  
- HTML/CSS/JavaScript (for frontend UI)  

---

## 📊 Project Highlights

### 1. Exercise Detection & Rep Counting
- Detects body landmarks using MediaPipe Pose  
- Calculates joint angles for relevant exercises  
- Counts repetitions with **hysteresis thresholds** and **smooth angle tracking**  
- Tracks the **stage** of exercise (up/down) for accurate rep counting  

### 2. Supported Exercises
- **Bicep Curl** – Elbow angle detection  
- **Squat** – Knee angle detection  
- **Push-Up** – Average elbow angle  
- **Lateral Raise** – Shoulder abduction angle  
- **Shoulder Press** – Elbow/shoulder extension  

### 3. Real-Time Feedback
- Displays live **rep count**, **stage**, and **current angle** on the video feed  
- Alerts when landmarks are not visible  

---

## 📈 Sample Interface

- Live video feed with overlays for:
  - Exercise name  
  - Reps counted  
  - Stage (up/down)  
  - Joint angle  

- Buttons to **switch between exercises** dynamically  

---

## 📌 Future Improvements
- Store full workout sessions for analysis  
- Generate visual insights like graphs of reps, sets, and exercise distribution  
- Add more exercises and custom workout routines  
- Integrate with a mobile-friendly interface  

---

## 🤝 Acknowledgements
- **MediaPipe** for pose estimation  
- **OpenCV** for video processing  

