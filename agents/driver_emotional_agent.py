import joblib
import librosa
import numpy as np
from collections import deque


class SpeechEmotionAgent:
    """
    Speech-based emotional risk estimator.
    Outputs smoothed emotional risk in [0, 1].
    """

    def __init__(self, model_path="models/speech_emotion_ravdess.pkl"):
        self.model = joblib.load(model_path)
        self.risk_buffer = deque(maxlen=10)  # temporal smoothing window

    def extract_features(self, audio, sr):
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Pitch
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch = pitch[pitch > 0]
        pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0.0

        # Energy
        energy = np.mean(librosa.feature.rms(y=audio))

        return np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            [pitch_mean],
            [energy]
        ])

    def update(self, audio, sr):
        if audio is None or len(audio) == 0:
            return 0.0

        features = self.extract_features(audio, sr)
        raw_risk = float(self.model.predict([features])[0])
        raw_risk = np.clip(raw_risk, 0.0, 1.0)

        # Temporal smoothing
        self.risk_buffer.append(raw_risk)
        return float(np.mean(self.risk_buffer))
