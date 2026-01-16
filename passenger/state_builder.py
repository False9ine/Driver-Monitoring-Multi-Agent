import numpy as np

class PassengerStateBuilder:
    def __init__(self):
        self.prev_pose = None

    def _pt(self, lm):
        return np.array([lm.x, lm.y], dtype=np.float32)

    def _vel(self, a, b):
        return np.linalg.norm(a - b)

    def build_state(self, pose):
        # MediaPipe Pose indices
        HEAD = 0
        LS = 11
        RS = 12
        LW = 15
        RW = 16

        head = self._pt(pose[HEAD])
        ls = self._pt(pose[LS])
        rs = self._pt(pose[RS])
        lw = self._pt(pose[LW])
        rw = self._pt(pose[RW])

        # First frame
        if self.prev_pose is None:
            self.prev_pose = (head, ls, rs, lw, rw)
            return np.zeros(6, dtype=np.float32)

        ph, pls, prs, plw, prw = self.prev_pose

        head_v = self._vel(head, ph)
        torso_w = np.linalg.norm(ls - rs)
        torso_v = abs(torso_w - np.linalg.norm(pls - prs))
        left_v = self._vel(lw, plw)
        right_v = self._vel(rw, prw)

        motion_energy = head_v + torso_v + left_v + right_v

        self.prev_pose = (head, ls, rs, lw, rw)

        # 🔥 IMPORTANT: NO NORMALIZATION HERE
        return np.array([
            head_v,
            torso_w,
            torso_v,
            left_v,
            right_v,
            motion_energy
        ], dtype=np.float32)
