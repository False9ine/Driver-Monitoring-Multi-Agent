import time
from collections import deque


class ExplainableFusionAgent:
    def __init__(self):

        self.w_eye = 0.4
        self.w_blink = 0.35
        self.w_head = 0.25

        self.safe_th = 0.7
        self.warning_th = 0.4

        self.history_seconds = 30
        self.times = deque()
        self.scores = deque()

    def update(self, eye, blink, head, eye_meta=None, blink_meta=None, head_meta=None):

        C_eye = self.w_eye * eye
        C_blink = self.w_blink * blink
        C_head = self.w_head * head

        final_score = C_eye + C_blink + C_head

        if final_score >= self.safe_th:
            state = "SAFE"
        elif final_score >= self.warning_th:
            state = "WARNING"
        else:
            state = "DROWSY"

        return alertness_score, state
