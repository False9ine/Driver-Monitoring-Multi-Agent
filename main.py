import cv2
from perception.face_mesh import FaceMeshDetector
from agents.eye_state_agent import EyeStateAgent


detector = FaceMeshDetector()
eye_agent = EyeStateAgent()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lm = detector.process(frame)

    eye_score = eye_agent.update(lm)

    cv2.putText(frame,
                f"Eye Score: {eye_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.imshow("Eye State Agent Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
