# AI Smart Rep Counter using OpenCV and MediaPipe

## Description
AI Smart Rep Counter is a real-time exercise detection and rep-counting application built using **Python**, **OpenCV**, and **MediaPipe**. It can detect exercises such as **bicep curls, squats, push-ups, lateral raises, and shoulder presses** from your webcam feed, count repetitions, and track workout progress.  

This tool is perfect for fitness enthusiasts who want to monitor their workouts without needing a trainer or specialized equipment.

---

## Features
- Real-time exercise detection using webcam.
- Automatic **rep counting** for multiple exercises.
- Tracks **stage** of movement (up/down) for accurate counting.
- Smoothing of joint angles to reduce noise.
- Configurable exercise parameters like angle thresholds, minimum rep time, and range of motion.
- Easily extendable for new exercises.
- Web-based interface with **Flask**:
  - Live webcam feed
  - Exercise selection buttons
  - Reps and stage display

---

## Supported Exercises
| Exercise         | Description                        |
|-----------------|------------------------------------|
| Bicep Curl       | Detects arm curls                  |
| Squat            | Tracks knee angle for squats       |
| Push-Up          | Tracks elbow angle for push-ups    |
| Lateral Raise    | Tracks shoulder abduction angle    |
| Shoulder Press   | Tracks shoulder & elbow angle      |

---

## Installation

1. **Clone the repository**:

git clone https://github.com/SahilKadolkar/Ai-Smart-Rep-Counter-using-Opencv.git
cd Ai-Smart-Rep-Counter-using-Opencv

2.  **Create a virtual environment** :

python -m venv env

3. **Activate the virtual environment** :

Windows:

env\Scripts\activate


Mac/Linux:

source env/bin/activate


4. **Install dependencies** :

pip install -r requirements.txt

** Usage

1. Run the Flask app:

python app.py


2. Open your browser and go to:

http://127.0.0.1:5000/


3. Choose an exercise using the buttons.

4. The webcam feed will show:

Exercise name

Current rep count

Stage (up/down)

Angle of relevant joints

File Structure
├─ app.py                # Main Flask application
├─ templates/
│   └─ index.html        # Frontend HTML page
├─ requirements.txt      # Python dependencies
├─ README.md             # Project documentation


Note: Do not include the virtual environment (env/) in the repository. Add it to .gitignore.

Contributing

Feel free to fork the repository and add new exercises, improve detection, or enhance UI features.

License

This project is open-source and available under the MIT License.


I can also create a **.gitignore** for you that excludes the virtual environment and other large files s
