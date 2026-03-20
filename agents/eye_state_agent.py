import numpy as np

class EyeStateAgent:
    """
    Eye State Agent (Explainable)
    -----------------------------
    Outputs:
        score ∈ [0,1]   → normalized eye openness
        meta:
            - ear       → raw Eye Aspect Ratio
    """

    def __init__(self):
        self.EAR_MIN = 0.15   # closed
        self.EAR_MAX = 0.35   # open

        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

    def _ear(self, eye_pts):
        p = np.array(eye_pts)

        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])

        return (A + B) / (2.0 * C)

    def update(self, landmarks):

        if landmarks is None:
            return 0.5, {"ear": None}

        left = [(landmarks[i].x, landmarks[i].y) for i in self.left_eye]
        right = [(landmarks[i].x, landmarks[i].y) for i in self.right_eye]

        ear_left = self._ear(left)
        ear_right = self._ear(right)
        ear = (ear_left + ear_right) / 2.0

        score = (ear - self.EAR_MIN) / (self.EAR_MAX - self.EAR_MIN)
        score = float(np.clip(score, 0.0, 1.0))

        return float(np.clip(score, 0.0, 1.0))
