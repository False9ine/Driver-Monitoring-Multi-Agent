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

        now = time.time()
        self.times.append(now)
        self.scores.append(final_score)

        while self.times and now - self.times[0] > self.history_seconds:
            self.times.popleft()
            self.scores.popleft()

        explanation = self._explain(final_score, state,
                                    eye, blink, head,
                                    C_eye, C_blink, C_head,
                                    eye_meta, blink_meta, head_meta)

        return final_score, state, explanation

    # ----------------- EXPLAINABILITY -----------------

    def _explain(self, final_score, state,
                 eye, blink, head,
                 C_eye, C_blink, C_head,
                 eye_meta, blink_meta, head_meta):

        total = final_score + 1e-6

        importance = {
            "eye": C_eye / total,
            "blink": C_blink / total,
            "head": C_head / total
        }

        dominant = min({"eye": eye, "blink": blink, "head": head}, key=lambda k: {
            "eye": eye, "blink": blink, "head": head}[k])

        reasons = []
        if eye < 0.4: reasons.append("Low eye openness detected")
        if blink < 0.4: reasons.append("Abnormal blink pattern")
        if head < 0.5: reasons.append("Head pose deviating from road")

        if blink_meta:
            if blink_meta["perclos"] > 0.35: reasons.append("High PERCLOS")
            if blink_meta["avg_duration"] > 0.6: reasons.append("Long eye closure")
            if blink_meta["blink_rate"] > 25: reasons.append("High blink rate")

        if head_meta:
            if abs(head_meta["yaw"]) > 25: reasons.append("Looking away from road")
            if abs(head_meta["pitch"]) > 20: reasons.append("Looking up/down frequently")

        trend = self._trend()

        counter = self._counterfactuals(eye, blink, head)

        return {
            "state": state,
            "score": float(final_score),
            "importance": importance,
            "dominant_factor": dominant,
            "reasons": reasons,
            "temporal_trend": trend,
            "counterfactuals": counter
        }

    def _trend(self):
        if len(self.scores) < 5:
            return "Insufficient data"

        delta = self.scores[-1] - self.scores[0]

        if delta < -0.15:
            return "Alertness decreasing"
        elif delta > 0.15:
            return "Alertness improving"
        else:
            return "Alertness stable"

    def _counterfactuals(self, eye, blink, head):
        cf = []

        if eye < 0.6:
            cf.append("Increase eye openness")
        if blink < 0.6:
            cf.append("Reduce eye closure duration")
        if head < 0.6:
            cf.append("Face the road consistently")

        if not cf:
            cf.append("Maintain current behavior")

        return cf
