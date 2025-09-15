import json
from pathlib import Path
import subprocess, os
from tqdm import tqdm
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


target_glosses = [
  "hello","thank you","please","yes","no","eat","drink","like","want",
  "who","what","where","when","why","sleep","help","good","bad","see",
  "again","understand","me","you","milk"
]


meta = json.load(open("WLASL_v0.3.json", "r")) 
target_set = set(target_glosses)

filtered = []
for item in meta:
    if item.get("gloss") in target_set:
        for inst in item.get("instances", []):
            filtered.append({
                "gloss": item["gloss"],
                "video_id": inst.get("video_id", inst.get("youtube_id")),
                "url": inst["url"],
                "start_frame": inst.get("frame_start", 0),
                "end_frame": inst.get("frame_end", None)
            })

print(f"Found {len(filtered)} clips for your selected glosses")




# out_root = Path("videos")
# out_root.mkdir(exist_ok=True)
# for item in tqdm(filtered):
#     gloss = item["gloss"]
#     url = item["url"]
#     vid = item["video_id"]
#     outdir = out_root / gloss
#     outdir.mkdir(parents=True, exist_ok=True)
#     outpath = outdir / f"{vid}.mp4"
#     if outpath.exists():
#         continue
#     # yt-dlp command
#     cmd = ["yt-dlp", "-f", "mp4", "-o", str(outpath), url]
#     subprocess.run(cmd)   # add error handling / retries in production


# extract landmarks
VIDEO_ROOT = Path("videos") 
OUTPUT_ROOT = Path("landmarks")
MAX_FRAMES = 48
OUTPUT_ROOT.mkdir(exist_ok=True)

# Setup mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

X = []
y = []

for gloss_folder in VIDEO_ROOT.iterdir():
    if not gloss_folder.is_dir():
        continue
    label = gloss_folder.name
    out_gloss_folder = OUTPUT_ROOT / label
    out_gloss_folder.mkdir(parents=True, exist_ok=True)

    for video_file in tqdm(list(gloss_folder.glob("*.mp4")), desc=f"Processing {label}"):
        cap = cv2.VideoCapture(str(video_file))
        sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                sequence.append(coords)

        cap.release()

        # Skip videos with no hand detected
        if len(sequence) == 0:
            continue

        # Pad or truncate to MAX_FRAMES
        seq_len = len(sequence)
        if seq_len < MAX_FRAMES:
            padding = np.zeros((MAX_FRAMES - seq_len, 21*3))
            sequence = np.vstack([sequence, padding])
        elif seq_len > MAX_FRAMES:
            sequence = sequence[:MAX_FRAMES]

        sequence = np.array(sequence)
        X.append(sequence)
        y.append(label)

        # Save each video sequence
        np.save(out_gloss_folder / f"{video_file.stem}.npy", sequence)

# --- ENCODE LABELS ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Save full dataset for training
np.save(OUTPUT_ROOT / "X.npy", np.array(X))
np.save(OUTPUT_ROOT / "y.npy", y_cat)
np.save(OUTPUT_ROOT / "labels.npy", le.classes_)

print("✅ Landmarks extraction done. Dataset ready for training!")