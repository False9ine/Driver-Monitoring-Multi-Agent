import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

from state_builder import build_motion_state
from build_sequences import window_sequence

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANDMARK_DIR = os.path.join(BASE_DIR, "data", "landmarks")
VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
MODEL_OUT = os.path.join(BASE_DIR, "passenger", "physical_model.pkl")


LABEL_TO_RISK = {
    "normal": 0.0,
    "unusual": 0.5,
    "aggressive": 1.0
}

X, y = [], []

for label, risk in LABEL_TO_RISK.items():
    folder = os.path.join(VIDEO_DIR, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        if not file.endswith(".mp4"):
            continue

        name = os.path.splitext(file)[0]
        lm_path = os.path.join(LANDMARK_DIR, name + ".npy")

        if not os.path.exists(lm_path):
            continue

        landmarks = np.load(lm_path)
        motion = build_motion_state(landmarks)
        windows = window_sequence(motion)

        for w in windows:
            X.append(w)
            y.append(risk)

X = np.array(X)
y = np.array(y)

print(f"✅ Training samples: {len(X)}")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)

print(f"🎉 Model saved → {MODEL_OUT}")