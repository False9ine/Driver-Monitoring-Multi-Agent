import cv2
import mediapipe as mp
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
OUT_DIR = os.path.join(BASE_DIR, "data", "landmarks")

os.makedirs(OUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

def extract_landmarks(video_path, save_path):
    print(f"\n🎥 Processing video: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                if len(lm) == 33:
                    frames.append(lm)

    cap.release()

    if len(frames) == 0:
        print(f"⚠️ No pose landmarks detected in {os.path.basename(video_path)}")
        return

    np.save(save_path, frames, allow_pickle=True)
    print(f"✅ Saved {len(frames)} pose frames → {os.path.basename(save_path)}")

for video in sorted(os.listdir(RAW_DIR)):
    if video.lower().endswith(".mp4"):
        extract_landmarks(
            os.path.join(RAW_DIR, video),
            os.path.join(OUT_DIR, os.path.splitext(video)[0] + ".npy")
        )

print("\n🎉 Pose landmark extraction finished.")
