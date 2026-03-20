import numpy as np


def window_sequence(
    motion_seq: np.ndarray,
    window_size: int = 20,
    overlap: int = 10
) -> np.ndarray:
    """
    Convert motion features into overlapping temporal windows.

    Args:
        motion_seq (np.ndarray): (T, 6)
        window_size (int): frames per window
        overlap (int): overlapping frames

    Returns:
        np.ndarray: (N_windows, feature_dim)
    """

    if len(motion_seq) < window_size:
        return np.empty((0, 18))

    step = window_size - overlap
    windows = []

    for start in range(0, len(motion_seq) - window_size + 1, step):
        window = motion_seq[start:start + window_size]

        # Aggregate statistics
        feat = np.concatenate([
            np.mean(window, axis=0),
            np.std(window, axis=0),
            np.max(window, axis=0),
        ])

        windows.append(feat)

    return np.array(windows, dtype=np.float32)