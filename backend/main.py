# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import base64
# import cv2
# import json

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load your pre-trained model
# # Replace with the path to your .h5 file
# model = load_model('sign_model.h5')

# # Define your gesture classes
# GESTURE_CLASSES = [
#    "again",
# "bad",
# "drink",
# "eat",
# "good",
# "hello",
# "help",
# "like",
# "me",
# "milk",
# "no",
# "please",
# "see",
# "sleep",
# "thank you",
# "understand",
# "want",
# "what",
# "when",
# "where",
# "who",
# "why",
# "yes",
# "you"
# ]

# def preprocess_keypoints(keypoints):
#     """
#     Preprocess the keypoints for your model
#     Adjust this function based on your model's requirements
#     """
#     # Convert to numpy array
#     points = np.array([[kp['x'], kp['y'], kp['z']] for kp in keypoints])
    
#     # Normalize coordinates (adjust based on your training data)
#     points[:, 0] = points[:, 0] / 640  # Normalize x by video width
#     points[:, 1] = points[:, 1] / 480  # Normalize y by video height
    
#     # Flatten and reshape for model input
#     processed = points.flatten()
#     processed = np.expand_dims(processed, axis=0)
    
#     return processed

# @app.route('/recognize', methods=['POST'])
# def recognize_gesture():
#     try:
#         data = request.get_json()
#         keypoints = data['keypoints']
        
#         # Preprocess the keypoints
#         processed_data = preprocess_keypoints(keypoints)
        
#         # Make prediction
#         predictions = model.predict(processed_data)
#         predicted_class = np.argmax(predictions[0])
#         confidence = np.max(predictions[0])
        
#         # Only return result if confidence is high enough
#         if confidence > 0.7:
#             gesture = GESTURE_CLASSES[predicted_class]
#             return jsonify({
#                 'gesture': gesture,
#                 'confidence': float(confidence)
#             })
#         else:
#             return jsonify({
#                 'gesture': 'Unknown',
#                 'confidence': float(confidence)
#             })
            
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import queue
from main_window import MainWindow

def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()