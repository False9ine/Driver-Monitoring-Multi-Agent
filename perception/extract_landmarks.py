import cv2
import mediapipe as mp
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
OUT_DIR = os.path.join(BASE_DIR, "data", "landmarks")

os.makedirs(OUT_DIR, exist_ok=True)

print("\n📂 RAW VIDEOS FOUND:")
for f in os.listdir(RAW_DIR):
    print(" -", repr(f))

def extract_landmarks(video_path, save_path):
    print(f"🔍 Opening video: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return

    # ⚠️ IMPORTANT: MediaPipe initialized PER VIDEO
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_result = pose.process(rgb)
            face_result = face_mesh.process(rgb)

            if pose_result.pose_landmarks and face_result.multi_face_landmarks:
                frames.append((
                    pose_result.pose_landmarks.landmark,
                    face_result.multi_face_landmarks[0].landmark
                ))

    cap.release()

    if len(frames) == 0:
        print(f"⚠️ No landmarks detected in {os.path.basename(video_path)}")
        return

    np.save(save_path, frames)
    print(f"✅ Saved {len(frames)} frames → {os.path.basename(save_path)}")

# -----------------------------
# PROCESS ALL VIDEOS
# -----------------------------
for video in sorted(os.listdir(RAW_DIR)):
    if video.lower().endswith(".mp4"):
        video_path = os.path.join(RAW_DIR, video)
        save_path = os.path.join(OUT_DIR, os.path.splitext(video)[0] + ".npy")

        print(f"\n🎥 Processing: {video}")
        extract_landmarks(video_path, save_path)

print("\n🎉 Landmark extraction finished for ALL videos.")
