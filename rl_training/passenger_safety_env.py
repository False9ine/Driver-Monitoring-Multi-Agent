import numpy as np

class PassengerSafetyEnv:
    """
    RL environment for passenger safeguarding
    against potentially harmful driver behavior.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.physical = np.random.uniform(0.2, 0.6)
        self.emotional = np.random.uniform(0.2, 0.6)
        self.timestep = 0
        return self._get_state()

    def _get_state(self):
        fused = 0.6 * self.physical + 0.4 * self.emotional
        return np.array([self.physical, self.emotional, fused], dtype=np.float32)

    def step(self, action):
        """
        Actions:
        0 - NO_ACTION
        1 - VERBAL_WARNING
        2 - PROTECTIVE_INTERVENTION
        """

        reward = 0.0

        # -------------------------
        # NO ACTION
        # -------------------------
        if action == 0:
            if self.emotional < 0.3:
                reward += 2            # correct calm behavior
            else:
                reward -= 1            # ignored rising anger
            self.emotional += 0.03

        # -------------------------
        # VERBAL WARNING
        # -------------------------
        elif action == 1:
            if 0.3 <= self.emotional <= 0.6:
                reward += 3            # correct timing
            else:
                reward -= 1            # unnecessary or late warning
            self.emotional -= 0.08

        # -------------------------
        # PROTECTIVE INTERVENTION
        # -------------------------
        elif action == 2:
            if self.emotional > 0.6:
                reward += 5            # justified escalation
            else:
                reward -= 6            # FALSE ESCALATION (KEY FIX)
            self.emotional -= 0.2

        # Clamp emotional risk
        self.emotional = np.clip(self.emotional, 0.0, 1.0)

        # Severe passenger risk penalty
        if self.emotional > 0.85:
            reward -= 10

        self.timestep += 1
        done = self.emotional < 0.2 or self.timestep >= 30

        return self._get_state(), reward, done