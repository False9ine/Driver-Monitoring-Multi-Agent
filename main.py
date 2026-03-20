import cv2

from perception.face_mesh import FaceMeshDetector
from agents.eye_state_agent import EyeStateAgent
from agents.blink_agent import BlinkPatternAgent
from agents.head_pose_agent import HeadPoseAgent
from agents.fusion_agent import ExplainableFusionAgent


# ---------------- DEEP COLOR PALETTE (BGR) ----------------
DEEP_RED     = (0, 0, 180)
DEEP_ORANGE  = (0, 90, 180)
DEEP_BLUE    = (180, 60, 0)
DEEP_GREEN   = (0, 150, 0)
DEEP_YELLOW  = (0, 180, 180)
DEEP_RED   = (0, 0, 180)


def main():
    # -------- Initialize components --------
    face_mesh = FaceMeshDetector()
    eye_agent = EyeStateAgent()
    blink_agent = BlinkPatternAgent()
    head_pose_agent = HeadPoseAgent()
    fusion_agent = ExplainableFusionAgent()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print("[INFO] Explainable Driver Monitoring System started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        landmarks = face_mesh.process(frame)

        if landmarks is not None:

            # -------- Eye State Agent --------
            eye_score, eye_meta = eye_agent.update(landmarks)
            ear_value = eye_meta["ear"] if eye_meta["ear"] else 0.3

            # -------- Blink Pattern Agent --------
            blink_score, blink_meta = blink_agent.update(ear_value)

            # -------- Head Pose Agent --------
            head_pose_data = head_pose_agent.update(landmarks, frame.shape)
            head_pose_score = head_pose_data["score"]

            # -------- Fusion + Explainability --------
            alertness_score, state, explanation = fusion_agent.update(
                eye_score,
                blink_score,
                head_pose_score,
                eye_meta=eye_meta,
                blink_meta=blink_meta,
                head_meta=head_pose_data
            )

            # -------- Basic Scores --------
            cv2.putText(frame, f"Eye score: {eye_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEEP_RED, 2)

            cv2.putText(frame, f"Blink score: {blink_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEEP_RED, 2)

            cv2.putText(frame, f"Head score: {head_pose_score:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEEP_RED, 2)

            cv2.putText(frame,
                        f"Yaw: {head_pose_data['yaw']:.1f}  Pitch: {head_pose_data['pitch']:.1f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DEEP_RED, 1)

            cv2.putText(frame, f"Alertness: {alertness_score:.2f}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEEP_RED, 2)

            # -------- STATE --------
            if state == "SAFE":
                state_color = DEEP_GREEN
            elif state == "WARNING":
                state_color = DEEP_YELLOW
            else:
                state_color = DEEP_RED

            cv2.putText(frame, f"STATE: {state}", (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, state_color, 3)

            # ================= EXPLAINABILITY PANEL =================

            y0 = 230

            # ---- Dominant Factor ----
            cv2.putText(frame, f"DOMINANT FACTOR: {explanation['dominant_factor']}",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, DEEP_RED, 2)

            # ---- Temporal Trend ----
            cv2.putText(frame, f"TREND: {explanation['temporal_trend']}",
                        (10, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEEP_BLUE, 2)

            # ---- Reasons ----
            cv2.putText(frame, "REASONS:",
                        (10, y0 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEEP_RED, 2)

            for i, r in enumerate(explanation["reasons"][:3]):
                cv2.putText(frame, f"- {r}",
                            (10, y0 + 95 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEEP_ORANGE, 2)

            # ---- Suggestions ----
            cv2.putText(frame, "SUGGESTIONS:",
                        (10, y0 + 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEEP_RED, 2)

            for i, c in enumerate(explanation["counterfactuals"][:2]):
                cv2.putText(frame, f"> {c}",
                            (10, y0 + 205 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEEP_GREEN, 2)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEEP_RED, 2)

        cv2.imshow("Explainable Driver Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")


if __name__ == "__main__":
    main()
