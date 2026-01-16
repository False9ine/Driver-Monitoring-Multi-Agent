import os
import numpy as np
from state_builder import PassengerStateBuilder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARK_DIR = os.path.join(BASE_DIR, "data", "landmarks")
OUT_DIR = os.path.join(BASE_DIR, "data", "passenger_sequences")

os.makedirs(OUT_DIR, exist_ok=True)

builder = PassengerStateBuilder()

for file in sorted(os.listdir(LANDMARK_DIR)):
    if not file.endswith(".npy"):
        continue

    print(f"\n🔄 Processing {file}")
    data = np.load(os.path.join(LANDMARK_DIR, file), allow_pickle=True)

    builder.prev_pose = None
    states = []

    for pose_landmarks in data:
        state = builder.build_state(pose_landmarks)
        states.append(state)

    states = np.array(states, dtype=np.float32)

    # 🔥 SEQUENCE-LEVEL NORMALIZATION (CRITICAL)
    max_vals = np.max(states, axis=0) + 1e-6
    states = states / max_vals

    np.save(os.path.join(OUT_DIR, file), states)

    print(f"✅ Saved {states.shape} → {file}")

print("\n🎉 Passenger motion sequences rebuilt successfully.")
