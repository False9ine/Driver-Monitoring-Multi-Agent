import numpy as np

class PassengerSafetyEnv:
    """
    RL environment for passenger safeguarding
    against potentially harmful driver behavior.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Initial risks (can be seeded from real fusion output)
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

        # ----- DRIVER RESPONSE MODEL -----
        if action == 0:  # ignore
            self.emotional += 0.05
            reward -= 1

        elif action == 1:  # warning
            self.emotional -= 0.08
            reward += 2

        elif action == 2:  # protective action
            self.emotional -= 0.2
            reward += 5

        # Clamp
        self.emotional = np.clip(self.emotional, 0, 1)

        # Passenger harm penalty
        if self.emotional > 0.8:
            reward -= 10

        # Success condition
        done = self.emotional < 0.2 or self.timestep > 30
        self.timestep += 1

        return self._get_state(), reward