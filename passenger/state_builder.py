import numpy as np

# Upper-body landmark indices (MediaPipe Pose)
UPPER_BODY = [
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16   # wrists
]


def build_motion_state(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Converts pose landmarks into motion-based features.

    Args:
        landmark_sequence (np.ndarray):
            Shape (T, 33, 4) → x, y, z, visibility

    Returns:
        np.ndarray:
            Shape (T-1, 6)
    """

    if len(landmark_sequence) < 2:
        return np.empty((0, 6))

    motion_features = []

    for t in range(1, len(landmark_sequence)):
        prev = landmark_sequence[t - 1]
        curr = landmark_sequence[t]

        velocities = []

        for idx in UPPER_BODY:
            dx = curr[idx, 0] - prev[idx, 0]
            dy = curr[idx, 1] - prev[idx, 1]
            velocities.append(np.sqrt(dx * dx + dy * dy))

        velocities = np.array(velocities)

        # 6D motion descriptor
        features = [
            np.mean(velocities),         # mean velocity
            np.max(velocities),          # max velocity
            np.std(velocities),          # motion variance
            np.sum(velocities),          # motion energy
            np.percentile(velocities, 75),
            np.percentile(velocities, 90)
        ]

        motion_features.append(features)

    return np.array(motion_features, dtype=np.float32)