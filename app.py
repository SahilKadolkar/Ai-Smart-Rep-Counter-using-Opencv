from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ------------------ Global State ------------------
counter = 0
stage = None
current_exercise = "bicep_curl"   # default
rep_counter = None   # will be created per exercise
session_active = False   # ✅ session control

# ------------------ Utils ------------------
def calculate_angle(a, b, c):
    """Angle at point b given 2D points a, b, c."""
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def visible(lm, names, thresh=0.6):
    """All requested landmarks have sufficient visibility?"""
    return all(lm[n].visibility >= thresh for n in names)

# Quick name->enum index map for convenience
L = mp_pose.PoseLandmark
def P(lm, name_enum):
    e = name_enum
    return np.array([lm[e].x, lm[e].y, lm[e].visibility], dtype=float)

# ------------------ Smoother ------------------
class SmoothAngle:
    def __init__(self, window=5):
        self.buf = deque(maxlen=window)
        self.prev_val = None
        self.prev_time = None

    def update(self, val):
        self.buf.append(val)
        return float(np.median(self.buf))  # median = robust to outliers

# ------------------ Generic Rep Counter ------------------
class RepCounter:
    """
    Generic rep counter with:
      - low/high hysteresis thresholds
      - min ROM requirement
      - min time between reps (debounce)
      - optional hold time at extremes
      - stage tracking ('down' ~ high angle, 'up' ~ low angle)
      - count_on: 'up' or 'down'
    """
    def __init__(self, low_th, high_th, min_rom=60.0, min_rep_time=0.5, hold_time=0.0, count_on='up', smooth_window=5):
        self.low_th = low_th
        self.high_th = high_th
        self.min_rom = min_rom
        self.min_rep_time = min_rep_time
        self.hold_time = hold_time
        self.count_on = count_on
        self.smoother = SmoothAngle(window=smooth_window)

        self.stage = None
        self.counter = 0
        self.last_switch_time = None
        self.rep_start_time = None

        # ROM tracking
        self.cycle_min = 9999.0
        self.cycle_max = -9999.0

        # Hold-time detection
        self.extreme_reached_time = None

    def reset_cycle(self):
        self.cycle_min = 9999.0
        self.cycle_max = -9999.0
        self.extreme_reached_time = None

    def update(self, raw_angle):
        now = time.time()
        angle = self.smoother.update(raw_angle)

        # Track ROM
        self.cycle_min = min(self.cycle_min, angle)
        self.cycle_max = max(self.cycle_max, angle)

        # Initialize stage
        if self.stage is None:
            self.stage = 'down' if angle >= self.high_th else ('up' if angle <= self.low_th else None)
            self.rep_start_time = now
            return self.stage, self.counter, angle

        # Stage transitions
        made_switch = False
        if self.stage == 'down':
            if angle <= self.low_th:
                if self.extreme_reached_time is None:
                    self.extreme_reached_time = now
                if (now - self.extreme_reached_time) >= self.hold_time:
                    self.stage = 'up'
                    self.last_switch_time = now
                    self.extreme_reached_time = None
                    made_switch = True
            else:
                self.extreme_reached_time = None

        elif self.stage == 'up':
            if angle >= self.high_th:
                if self.extreme_reached_time is None:
                    self.extreme_reached_time = now
                if (now - self.extreme_reached_time) >= self.hold_time:
                    self.stage = 'down'
                    self.last_switch_time = now
                    self.extreme_reached_time = None
                    made_switch = True
            else:
                self.extreme_reached_time = None

        # Count rep if valid
        if made_switch:
            if self.rep_start_time is None:
                self.rep_start_time = now

            complete = (self.count_on == self.stage)
            enough_time = (now - self.rep_start_time) >= self.min_rep_time
            rom = self.cycle_max - self.cycle_min
            enough_rom = rom >= self.min_rom

            if complete and enough_time and enough_rom:
                self.counter += 1
                self.rep_start_time = now
                self.reset_cycle()
            else:
                self.rep_start_time = now
                self.reset_cycle()

        return self.stage, self.counter, angle

# ------------------ Exercise Angle Functions ------------------
def angle_bicep_curl(lm):
    needed = [L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value]
    if not visible(lm, needed):
        return None
    return calculate_angle(P(lm, L.LEFT_SHOULDER.value), P(lm, L.LEFT_ELBOW.value), P(lm, L.LEFT_WRIST.value))

def angle_pushup(lm):
    needed = [L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value,
              L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value]
    if not visible(lm, needed):
        return None
    a_left = calculate_angle(P(lm, L.LEFT_SHOULDER.value), P(lm, L.LEFT_ELBOW.value), P(lm, L.LEFT_WRIST.value))
    a_right = calculate_angle(P(lm, L.RIGHT_SHOULDER.value), P(lm, L.RIGHT_ELBOW.value), P(lm, L.RIGHT_WRIST.value))
    return (a_left + a_right) * 0.5

def angle_squat(lm):
    needed = [L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value,
              L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value]
    if not visible(lm, needed):
        return None
    a_left = calculate_angle(P(lm, L.LEFT_HIP.value), P(lm, L.LEFT_KNEE.value), P(lm, L.LEFT_ANKLE.value))
    a_right = calculate_angle(P(lm, L.RIGHT_HIP.value), P(lm, L.RIGHT_KNEE.value), P(lm, L.RIGHT_ANKLE.value))
    return (a_left + a_right) * 0.5

def angle_lateral_raise(lm):
    needed = [L.LEFT_HIP.value, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value]
    if not visible(lm, needed):
        return None
    return calculate_angle(P(lm, L.LEFT_HIP.value), P(lm, L.LEFT_SHOULDER.value), P(lm, L.LEFT_ELBOW.value))

def angle_shoulder_press(lm):
    needed = [L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value]
    if not visible(lm, needed):
        return None
    return calculate_angle(P(lm, L.LEFT_SHOULDER.value), P(lm, L.LEFT_ELBOW.value), P(lm, L.LEFT_WRIST.value))

# ------------------ Exercise Config ------------------
EXERCISE_CFG = {
    "bicep_curl": {
        "angle_fn": angle_bicep_curl,
        "low": 40, "high": 160, "min_rom": 60, "count_on": "up",
        "min_rep_time": 0.6, "hold_time": 0.05, "smooth": 5
    },
    "squat": {
        "angle_fn": angle_squat,
        "low": 90, "high": 160, "min_rom": 50, "count_on": "up",
        "min_rep_time": 0.8, "hold_time": 0.05, "smooth": 5
    },
    "pushup": {
        "angle_fn": angle_pushup,
        "low": 90, "high": 160, "min_rom": 50, "count_on": "up",
        "min_rep_time": 0.7, "hold_time": 0.05, "smooth": 5
    },
    "lateral_raise": {
        "angle_fn": angle_lateral_raise,
        "low": 30, "high": 80, "min_rom": 40, "count_on": "down",
        "min_rep_time": 0.6, "hold_time": 0.05, "smooth": 5
    },
    "shoulder_press": {
        "angle_fn": angle_shoulder_press,
        "low": 60, "high": 150, "min_rom": 60, "count_on": "down",
        "min_rep_time": 0.7, "hold_time": 0.05, "smooth": 5
    },
}

def make_counter(name):
    cfg = EXERCISE_CFG[name]
    return RepCounter(
        low_th=cfg["low"],
        high_th=cfg["high"],
        min_rom=cfg["min_rom"],
        min_rep_time=cfg["min_rep_time"],
        hold_time=cfg["hold_time"],
        count_on=cfg["count_on"],
        smooth_window=cfg["smooth"]
    )

# ------------------ Video Gen ------------------
def gen_frames():
    global counter, stage, current_exercise, rep_counter, session_active
    cap = cv2.VideoCapture(0)

    if rep_counter is None:
        rep_counter = make_counter(current_exercise)

    with mp_pose.Pose(min_detection_confidence=0.6,
                     min_tracking_confidence=0.6,
                     model_complexity=1) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            smooth_angle = 0
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                angle_fn = EXERCISE_CFG[current_exercise]["angle_fn"]
                angle = angle_fn(lm)

                if angle is not None:
                    if session_active:   # ✅ only count during session
                        stage, counter, smooth_angle = rep_counter.update(angle)
                    else:
                        stage, counter, smooth_angle = stage, rep_counter.counter, angle

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Overlay info
            cv2.putText(image, f"Exercise: {current_exercise}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Reps: {rep_counter.counter}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Stage: {stage}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f"Angle: {int(smooth_angle)}°", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            if not session_active:
                cv2.putText(image, "SESSION STOPPED", (200, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_session')
def start_session():
    global session_active, rep_counter
    rep_counter = make_counter(current_exercise)
    session_active = True
    return "Workout started!"

@app.route('/stop_session')
def stop_session():
    global session_active
    session_active = False
    return "Workout stopped!"

@app.route('/set_exercise/<exercise>')
def set_exercise(exercise):
    global current_exercise, rep_counter
    exercise = exercise.lower()
    if exercise not in EXERCISE_CFG:
        return f"Unknown exercise: {exercise}", 400
    current_exercise = exercise
    rep_counter = make_counter(current_exercise)
    return f"Exercise set to {exercise}"

if __name__ == '__main__':
    app.run(debug=True)
