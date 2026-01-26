import os
import librosa
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

# =========================
# CONFIG
# =========================
DATA_DIR = "data/speech/ravdess"
MODEL_OUT = "models/speech_emotion_ravdess.pkl"

# Emotion → Safety Risk Mapping
RISK_MAP = {
    "01": 0.0,  # neutral
    "02": 0.0,  # calm
    "03": 0.1,  # happy
    "04": 0.2,  # sad
    "05": 1.0,  # angry (primary threat signal)
    "06": 0.4   # fearful (stress, not aggression)
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Pitch
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch = pitch[pitch > 0]
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0.0

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [pitch_mean],
        [energy]
    ])

# =========================
# LOAD DATA
# =========================
X, y, speakers = [], [], []

for actor in os.listdir(DATA_DIR):
    if not actor.startswith("Actor_"):
        continue

    actor_path = os.path.join(DATA_DIR, actor)
    speaker_id = actor.split("_")[1]

    for file in os.listdir(actor_path):
        parts = file.split("-")
        emotion_code = parts[2]

        if emotion_code not in RISK_MAP:
            continue

        file_path = os.path.join(actor_path, file)
        features = extract_features(file_path)

        X.append(features)
        y.append(RISK_MAP[emotion_code])
        speakers.append(speaker_id)

X = np.array(X)
y = np.array(y)
speakers = np.array(speakers)

# =========================
# SPEAKER-INDEPENDENT SPLIT
# =========================
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=speakers))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# =========================
# TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

print("Train R²:", model.score(X_train, y_train))
print("Val R²:", model.score(X_val, y_val))

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_OUT)

print("✅ Speech emotion model trained & saved")
