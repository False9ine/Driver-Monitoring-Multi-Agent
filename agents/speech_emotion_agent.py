import joblib
import librosa
import numpy as np


class SpeechEmotionAgent:
    """
    Speech-based emotional risk estimator.
    Outputs emotional risk in [0, 1].
    """

    def __init__(self, model_path="models/speech_emotion_ravdess.pkl"):
        self.model = joblib.load(model_path)

        # Baseline calibration
        self.baseline_ready = False
        self.energy_base = []
        self.pitch_base = []

    def extract_features(self, audio, sr):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch = pitch[pitch > 0]

        pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0.0
        energy = np.mean(librosa.feature.rms(y=audio))

        return np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            [pitch_mean],
            [energy]
        ])

    def update(self, audio, sr):
        if audio is None:
            return 0.0

        features = self.extract_features(audio, sr)
        raw_risk = float(self.model.predict([features])[0])
        raw_risk = np.clip(raw_risk, 0.0, 1.0)

        return raw_risk
