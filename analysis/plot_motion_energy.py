import numpy as np
import os
import matplotlib.pyplot as plt

SEQ_DIR = "data/passenger_sequences"

plt.figure()

for file in sorted(os.listdir(SEQ_DIR)):
    if not file.endswith(".npy"):
        continue

    seq = np.load(os.path.join(SEQ_DIR, file))
    motion = seq[:, -1]  # motion_energy

    if "aggressive" in file:
        plt.plot(motion, alpha=0.7, label=file)
    elif "normal" in file:
        plt.plot(motion, alpha=0.3)
    else:
        plt.plot(motion, alpha=0.5)

plt.title("Passenger Motion Energy")
plt.xlabel("Frame")
plt.ylabel("Normalized Motion")
plt.show()
