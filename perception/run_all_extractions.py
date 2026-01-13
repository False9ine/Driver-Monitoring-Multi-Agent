import cv2
import mediapipe as mp
import numpy as np
import os
import sys

print("🔥 Landmark Extraction Started")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
OUT_DIR = os.path.join(BASE_DIR, "data", "landmarks")

os.makedirs(OUT_DIR, exist_ok=True)

videos = sorted([
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith(".mp4")
])

if len(videos) == 0:
    print("⚠️ No videos found")
    sys.exit(0)

print(f"🎞️ Videos found: {len(videos)}")

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose, mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    for video in videos:
        video_path = os.path.join(RAW_DIR, video)
        save_path = os.path.join(
            OUT_DIR, os.path.splitext(video)[0] + ".npy"
        )

        print(f"\n🚀 Processing {video}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video:", video)
            continue

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            face_res = face_mesh.process(rgb)

            if pose_res.pose_landmarks and face_res.multi_face_landmarks:
                pose_lm = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in pose_res.pose_landmarks.landmark],
                    dtype=np.float32
                )

                face_lm = np.array(
                    [[lm.x, lm.y, lm.z]
                     for lm in face_res.multi_face_landmarks[0].landmark],
                    dtype=np.float32
                )

                frames.append({
                    "pose": pose_lm,
                    "face": face_lm
                })

        cap.release()

        if len(frames) == 0:
            print("⚠️ No landmarks detected")
            continue

        np.save(save_path, frames, allow_pickle=True)
        print(f"✅ Saved {len(frames)} frames → {save_path}")

print("\n🎯 ALL VIDEOS PROCESSED SUCCESSFULLY")
