# main.py
from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import threading, time
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# CONFIG
SEQ_LEN = 48
SESSION_MAX_AGE = 60  # seconds, cleanup idle sessions
ALLOWED_ORIGINS = [
    "http://localhost:5173", 
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    ]  # React dev server

# Khmer translations for your labels
khmer_translations = {
    "hello": "សួស្តី",
    "thanks": "អរគុណ",
    "yes": "បាទ",
    "no": "ទេ",
    "please": "សូម",
    "sorry": "សុំទោស",
    "help": "ជួយ",
    "water": "ទឹក",
    "food": "ម្ហូប",
    "home": "ផ្ទះ",
    "school": "សាលា",
    "work": "ការងារ",
    "friend": "មិត្តភក្តិ",
    "family": "គ្រួសារ",
    "love": "ស្រលាញ់",
    "happy": "សប្បាយ",
    "sad": "ទូលាយ",
    "tired": "ហត់",
    "sick": "ឈឺ",
    "goodbye": "លាហើយ"
}

# app + CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model + labels
model = load_model("sign_model.h5")
labels = np.load("landmarks/labels.npy")     # adjust path if needed
FEATURE_DIM = None  # set after first landmark extraction

# mediapipe hands
mp_hands = mp.solutions.hands
mp_hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                   min_detection_confidence=0.5)

# session buffers: client_id -> {"deque": deque, "last": timestamp}
session_buffers = {}
buffers_lock = threading.Lock()

def extract_hand_landmarks_from_bgr(img_bgr):
    """Return flattened (21*3,) numpy array or None if no hand."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = mp_hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...), x_client_id: str = Header(None)):
    """
    Accepts a single image/frame. Header 'X-Client-Id' must be provided by the client to maintain per-client buffer.
    Returns JSON: {word, confidence, status, have, khmer_translation}
    """
    if not x_client_id:
        return {"error": "Missing X-Client-Id header (unique client id)"}

    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    lm = extract_hand_landmarks_from_bgr(img)
    if lm is None:
        return {"word": None, "confidence": 0.0, "status": "no_hand", "have": 0, "khmer_translation": ""}

    global FEATURE_DIM
    if FEATURE_DIM is None:
        FEATURE_DIM = lm.shape[0]

    with buffers_lock:
        sess = session_buffers.get(x_client_id)
        if sess is None:
            sess = {"deque": deque(maxlen=SEQ_LEN), "last": time.time()}
            session_buffers[x_client_id] = sess
        sess["deque"].append(lm)
        sess["last"] = time.time()
        have = len(sess["deque"])
        

    if have < SEQ_LEN:
        print(f"[DEBUG] Collecting frames ({have}/{SEQ_LEN}) from {x_client_id}")
        return {"word": None, "confidence": 0.0, "status": "collecting", "have": have, "khmer_translation": ""}
    else:
        print(f"[DEBUG] Ready to predict for {x_client_id} (frames={have})")
    
    # Build input batch and predict
    with buffers_lock:
        seq = np.array(sess["deque"], dtype=np.float32)  # shape (SEQ_LEN, FEATURE_DIM)
    seq = seq.reshape(1, SEQ_LEN, -1)
    preds = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = str(labels[idx])
    conf = float(preds[idx])
    
    # Get Khmer translation
    khmer_word = khmer_translations.get(label, "Translation not available")

    return {"word": label, "confidence": conf, "status": "predicted", "have": have, "khmer_translation": khmer_word}

@app.get(f"/reset/{client_id}")
async def reset_client(client_id: str):
    """Reset the buffer for a specific client"""
    with buffers_lock:
        if client_id in session_buffers:
            session_buffers[client_id]["deque"].clear()
            return {"message": f"Buffer reset for client {client_id}"}
        else:
            return {"message": f"No buffer found for client {client_id}"}

# Background cleanup thread to remove idle sessions
def cleanup_loop():
    while True:
        time.sleep(10)
        now = time.time()
        with buffers_lock:
            to_delete = [k for k,v in session_buffers.items() if now - v["last"] > SESSION_MAX_AGE]
            for k in to_delete:
                del session_buffers[k]
                print(f"Cleaned up session for client {k}")

threading.Thread(target=cleanup_loop, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)