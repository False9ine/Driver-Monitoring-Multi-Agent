import time
from collections import deque
import numpy as np


class BlinkPatternAgent:
    """
    Blink Pattern Agent (Explainable)
    ---------------------------------
    Outputs:
        score ∈ [0,1]  → eye fatigue (1 = alert, 0 = fatigued)

        meta:
            - perclos
            - blink_rate (per min)
            - avg_duration (sec)
    """

    def __init__(self,
                 ear_threshold=0.20,
                 window_seconds=60):

        self.ear_threshold = ear_threshold
        self.window_seconds = window_seconds

        self.eye_closed = False
        self.current_blink_start = None

        self.blink_durations = deque()
        self.closed_frames = deque()
        self.frame_times = deque()

    def update(self, ear_value):

        now = time.time()

        self.frame_times.append(now)
        self.closed_frames.append(1 if ear_value < self.ear_threshold else 0)

        if ear_value < self.ear_threshold and not self.eye_closed:
            self.eye_closed = True
            self.current_blink_start = now

        if ear_value >= self.ear_threshold and self.eye_closed:
            self.eye_closed = False
            blink_duration = now - self.current_blink_start
            self.blink_durations.append(blink_duration)

        while self.frame_times and now - self.frame_times[0] > self.window_seconds:
            self.frame_times.popleft()
            self.closed_frames.popleft()

        while self.blink_durations and now - self.blink_durations[0] > self.window_seconds:
            self.blink_durations.popleft()

        return self._compute_fatigue_score()

    def _compute_fatigue_score(self):

        if not self.frame_times:
            return 1.0, {}

        perclos = sum(self.closed_frames) / len(self.closed_frames)
        blink_rate = len(self.blink_durations) / (self.window_seconds / 60)
        avg_duration = np.mean(self.blink_durations) if self.blink_durations else 0.0

        blink_rate_norm = min(blink_rate / 30.0, 1.0)
        duration_norm = min(avg_duration / 1.5, 1.0)

        fatigue = (0.4 * blink_rate_norm +
                   0.3 * duration_norm +
                   0.3 * perclos)

        return float(np.clip(1.0 - fatigue, 0.0, 1.0))

        meta = {
            "perclos": float(perclos),
            "blink_rate": float(blink_rate),
            "avg_duration": float(avg_duration)
        }

        return score, meta