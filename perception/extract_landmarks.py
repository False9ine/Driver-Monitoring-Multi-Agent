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
    print(f"🎥 Processing: {os.path.basename(video_path)}")

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
                frame_lm = []

                for p in lm:
                    frame_lm.append([
                        p.x, p.y, p.z, p.visibility
                    ])

                frames.append(frame_lm)

    cap.release()

    if len(frames) == 0:
        print("⚠️ No landmarks detected")
        return

    np.save(save_path, np.array(frames, dtype=np.float32))
    print(f"✅ Saved {len(frames)} frames → {os.path.basename(save_path)}")


for root, _, files in os.walk(RAW_DIR):
    for video in files:
        if video.lower().endswith(".mp4"):
            video_path = os.path.join(root, video)
            name = os.path.splitext(video)[0]
            save_path = os.path.join(OUT_DIR, name + ".npy")
            extract_landmarks(video_path, save_path)

print("🎉 Landmark extraction completed.")