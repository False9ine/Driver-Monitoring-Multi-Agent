import cv2
import numpy as np


class HeadPoseAgent:
    def __init__(self):

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        self.landmark_ids = [1, 152, 33, 263, 61, 291]

        self.PITCH_THRESHOLD = 30.0
        self.YAW_THRESHOLD = 40.0

        self.base_pitch = None
        self.base_yaw = None

    def update(self, landmarks, frame_shape):
        try:
            image_points = []
            for i in self.landmark_ids:
                lm = landmarks[i]
                image_points.append((lm.x * frame_shape[1],
                                     lm.y * frame_shape[0]))

            image_points = np.array(image_points, dtype=np.float64)

            focal_length = frame_shape[1]
            center = (frame_shape[1] / 2, frame_shape[0] / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            success, rot_vec, _ = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, np.zeros((4, 1)))

            if not success:
                return self._neutral()

            rot_mat, _ = cv2.Rodrigues(rot_vec)

            sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
            pitch = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            roll = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))

            if self.base_pitch is None:
                self.base_pitch = pitch
                self.base_yaw = yaw

            pitch -= self.base_pitch
            yaw -= self.base_yaw

            pitch_score = 1 - min(abs(pitch) / self.PITCH_THRESHOLD, 1)
            yaw_score = 1 - min(abs(yaw) / self.YAW_THRESHOLD, 1)

            score = float(np.clip(min(pitch_score, yaw_score), 0, 1))

            return {
                "score": score,
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll)
            }

        except Exception as e:
            print("[HEAD POSE ERROR]", e)
            return self._neutral()

    def _neutral(self):
        return {"score": 0.5, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}