import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
from gtts import gTTS
import os
import pygame
from pygame import mixer
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk
import threading
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description='Hand Gesture Recognition')
parser.add_argument('--model', type=str, default='sign_model.h5', help='Path to the trained model')
parser.add_argument('--labels', type=str, default='landmarks/labels.npy', help='Path to labels file')
parser.add_argument('--camera', type=int, default=0, help='Camera index')
args = parser.parse_args()

# Load model and labels
try:
    model = load_model(args.model)
    print(f"Model loaded successfully from {args.model}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    labels = np.load(args.labels, allow_pickle=True).tolist()
    print(f"Labels loaded: {labels}")
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

# Khmer translations for the labels
khmer_translations = {
    "again": "ម្តងទៀត",
    "bad": "អាក្រក់",
    "drink": "ផឹក",
    "eat": "ញ៉ាំ",
    "good": "ល្អ",
    "hello": "សួស្តី",
    "help": "ជួយ",
    "like": "ចូលចិត្ត",
    "me": "ខ្ញុំ",
    "milk": "ទឹកដោះគោ",
    "no": "ទេ",
    "please": "សូម",
    "see": "ឃើញ",
    "sleep": "គេង",
    "thank you": "អរគុណ",
    "understand": "យល់",
    "want": "ចង់",
    "what": "អ្វី",
    "when": "ពេលណា",
    "where": "ឯណា",
    "who": "អ្នកណា",
    "why": "ហេតុអ្វី",
    "yes": "បាទ",
    "you": "អ្នក"
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 150)  # Speed of speech
except:
    engine = None
    print("Text-to-speech engine not available")

# Initialize pygame for audio playback
mixer.init()

# Global variables
sequence = []
predictions = []
threshold = 0.7
last_prediction = ""
show_translation = False
is_running = True

def speak_text(text, language='en'):
    """Convert text to speech and play it"""
    try:
        if language == 'km' and text in khmer_translations:
            text_to_speak = khmer_translations[text]
        else:
            text_to_speak = text
            
        if engine:
            engine.say(text_to_speak)
            engine.runAndWait()
        else:
            # Use gTTS as fallback
            tts = gTTS(text=text_to_speak, lang='km' if language == 'km' else 'en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                mixer.music.load(fp.name)
                mixer.music.play()
                while mixer.music.get_busy():
                    continue
                os.unlink(fp.name)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def process_landmarks(hand_landmarks):
    """Process hand landmarks to create feature vector"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

class GestureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x700")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Open video capture
        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            self.window.destroy()
            return
        
        # Get model input shape
        _, self.seq_len, self.feature_dim = model.input_shape
        print(f"Model expects sequence length: {self.seq_len}, feature dimension: {self.feature_dim}")
        
        # Create a canvas that can fit the video frame
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(pady=10)
        
        # Create label for prediction
        self.prediction_label = tk.Label(window, text="Word: ", font=("Arial", 24))
        self.prediction_label.pack(pady=5)
        
        # Create label for confidence
        self.confidence_label = tk.Label(window, text="Confidence: ", font=("Arial", 16))
        self.confidence_label.pack(pady=5)
        
        # Create label for Khmer translation
        self.translation_label = tk.Label(window, text="Khmer: ", font=("Arial", 20))
        self.translation_label.pack(pady=5)
        
        # Create label for sequence progress
        self.sequence_label = tk.Label(window, text="Sequence: 0/30", font=("Arial", 14))
        self.sequence_label.pack(pady=5)
        
        # Create buttons
        button_frame = tk.Frame(window)
        button_frame.pack(pady=10)
        
        self.translate_btn = tk.Button(button_frame, text="Toggle Translation", command=self.toggle_translation)
        self.translate_btn.pack(side=tk.LEFT, padx=5)
        
        self.speak_en_btn = tk.Button(button_frame, text="Speak English", command=self.speak_english)
        self.speak_en_btn.pack(side=tk.LEFT, padx=5)
        
        self.speak_km_btn = tk.Button(button_frame, text="Speak Khmer", command=self.speak_khmer)
        self.speak_km_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(button_frame, text="Quit", command=self.on_closing)
        self.quit_btn.pack(side=tk.LEFT, padx=5)
        
        # Start video loop
        self.update()
        
        self.window.mainloop()
    
    def toggle_translation(self):
        global show_translation
        show_translation = not show_translation
        self.update_translation()
    
    def speak_english(self):
        if last_prediction:
            speak_text(last_prediction, 'en')
    
    def speak_khmer(self):
        if last_prediction:
            speak_text(last_prediction, 'km')
    
    def update_translation(self):
        global last_prediction, show_translation
        if last_prediction and show_translation and last_prediction in khmer_translations:
            self.translation_label.config(text=f"Khmer: {khmer_translations[last_prediction]}")
        else:
            self.translation_label.config(text="Khmer: ")
    
    def update(self):
        global sequence, predictions, last_prediction
        
        if is_running:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Process hand landmarks for prediction
                if results.multi_hand_landmarks:
                    landmarks = process_landmarks(results.multi_hand_landmarks[0])
                    
                    # Add to sequence
                    sequence.append(landmarks)
                    
                    # Keep only the required sequence length
                    if len(sequence) > self.seq_len:
                        sequence = sequence[-self.seq_len:]
                    
                    # Update sequence label
                    self.sequence_label.config(text=f"Sequence: {len(sequence)}/{self.seq_len}")
                    
                    # Make prediction when we have enough frames
                    if len(sequence) == self.seq_len:
                        # Prepare input data
                        input_data = np.expand_dims(sequence, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(input_data, verbose=0)[0]
                        confidence = np.max(prediction)
                        predicted_idx = np.argmax(prediction)
                        
                        if confidence > threshold:
                            predicted_word = labels[predicted_idx] if predicted_idx < len(labels) else str(predicted_idx)
                            
                            # Update predictions list
                            predictions.append(predicted_word)
                            
                            # Keep only last 5 predictions
                            if len(predictions) > 5:
                                predictions = predictions[-5:]
                            
                            # Get most frequent prediction
                            from collections import Counter
                            if predictions:
                                most_common = Counter(predictions).most_common(1)[0][0]
                                last_prediction = most_common
                                
                                # Update UI
                                self.prediction_label.config(text=f"Word: {last_prediction}")
                                self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
                                self.update_translation()
                else:
                    # Reset sequence if no hand is detected
                    sequence = []
                    self.sequence_label.config(text=f"Sequence: {len(sequence)}/{self.seq_len}")
                
                # Convert frame to PhotoImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk
            
            # Schedule next update
            self.window.after(10, self.update)
    
    def on_closing(self):
        global is_running
        is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

def main():
    # Create a window and pass it to the Application object
    root = tk.Tk()
    app = GestureApp(root, "Hand Gesture Recognition")
    
    print("Application started")
    print("Controls available in the GUI:")
    print("- Toggle Translation: Show/hide Khmer translation")
    print("- Speak English: Speak the detected word in English")
    print("- Speak Khmer: Speak the detected word in Khmer")
    print("- Quit: Close the application")

if __name__ == "__main__":
    main()