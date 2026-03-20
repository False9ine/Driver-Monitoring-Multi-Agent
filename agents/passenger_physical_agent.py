import os
import numpy as np
import joblib

from passenger.state_builder import build_motion_state
from passenger.build_sequences import window_sequence


class PassengerPhysicalAgent:
    """
    Physical agent that estimates passenger motion risk.
    Output: physical_risk ∈ [0, 1]
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "passenger", "physical_model.pkl")


        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Physical model not found. "
                "Run passenger/train_physical_model.py first."
            )

        self.model = joblib.load(model_path)

    def infer(self, landmarks: np.ndarray) -> float:
        """
        Args:
            landmarks (np.ndarray): shape (T, 33, 4)

        Returns:
            float: physical risk in [0, 1]
        """

        # Step 1: landmarks → motion features
        motion = build_motion_state(landmarks)

        if len(motion) == 0:
            return 0.0

        # Step 2: motion → temporal windows
        windows = window_sequence(motion)

        if len(windows) == 0:
            return 0.0

        # Step 3: predict risk per window
        risks = self.model.predict(windows)

        # Step 4: temporal aggregation
        physical_risk = float(np.mean(risks))

        # Safety clamp
        return float(np.clip(physical_risk, 0.0, 1.0))