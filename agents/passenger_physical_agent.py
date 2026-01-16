import numpy as np
import torch

from rl_training.model import DQN
from rl_training.passenger_env import PassengerSafetyEnv


class PassengerPhysicalAgent:
    """
    High-sensitivity physical anomaly detector.

    Uses a trained RL policy to detect sustained abnormal
    passenger motion from pose-based feature sequences.
    """

    def __init__(
        self,
        model_path="rl_training/passenger_dqn.pth",
        high_percentile=85,
        min_frames=20,
        persist_ratio=0.10,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(input_dim=6, num_actions=3).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        # Detection hyperparameters (FROZEN)
        self.high_percentile = high_percentile
        self.min_frames = min_frames
        self.persist_ratio = persist_ratio

    def detect_physical_anomaly(self, motion_sequence):
        """
        Detects sustained physical anomaly.

        Args:
            motion_sequence (np.ndarray):
                Shape: (T, 6)

        Returns:
            int:
                1 -> physical anomaly detected
                0 -> no anomaly
        """

        if motion_sequence is None or len(motion_sequence) == 0:
            return 0

        env = PassengerSafetyEnv(motion_sequence)
        state = env.reset()
        done = False

        risks = []

        while not done:
            state_tensor = torch.tensor(
                state, dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                q_vals = self.model(state_tensor)
                risk = torch.max(q_vals).item()

            risks.append(risk)
            state, _, done = env.step(0)

        risks = np.array(risks)
        total_frames = len(risks)

        # Extreme-motion threshold (relative)
        high_thr = np.percentile(risks, self.high_percentile)
        high_frames = np.sum(risks >= high_thr)

        # Temporal persistence constraint
        required_frames = max(
            self.min_frames,
            int(self.persist_ratio * total_frames)
        )

        return int(high_frames >= required_frames)
