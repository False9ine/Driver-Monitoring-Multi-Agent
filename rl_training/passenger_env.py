class PassengerSafetyEnv:
    def __init__(self, sequence):
        self.sequence = sequence
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self.sequence[self.idx]

    def step(self, action):
        state = self.sequence[self.idx]
        motion = state[-1]

        reward = self._reward(motion, action)

        self.idx += 1
        done = self.idx >= len(self.sequence)

        next_state = None if done else self.sequence[self.idx]
        return next_state, reward, done

    def _reward(self, motion, action):
        if motion < 0.2 and action == 0:
            return +0.2
        if 0.2 <= motion < 0.5 and action == 1:
            return +0.5
        if motion >= 0.5 and action == 2:
            return +1.0
        if motion >= 0.5 and action == 0:
            return -1.0
        return -0.2
