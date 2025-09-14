# backend/app/services/prediction_service.py
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

class PredictionService:
    def __init__(self):
        self.model = None
        self.labels = []
        self.load_model()
    
    def load_model(self):
        try:
            # Load TensorFlow.js model
            model_path = os.path.join(os.path.dirname(__file__), '../../model_files')
            
            # Load metadata
            with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                self.labels = metadata.get('userDefinedMetadata', {}).get('labels', [])
            
            print(f"Loaded labels: {self.labels}")
            
            # For TensorFlow.js models, you might need to convert them
            # This is a simplified approach - you might need to use tensorflowjs_converter
            print("Model loading would require conversion from TensorFlow.js format")
            print("Consider converting your model to SavedModel format first")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def is_model_loaded(self):
        return self.model is not None
    
    def get_supported_words(self):
        return self.labels
    
    def preprocess_image(self, image):
        # Resize and preprocess image for your model
        image = image.resize((224, 224))  # Adjust based on your model
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    
    def predict(self, image):
        # Mock prediction - replace with actual model inference
        # For now, return a mock response
        
        if not self.is_model_loaded():
            # Mock prediction for demo
            mock_words = ["HELLO", "THANK YOU", "YES", "NO", "PLEASE"]
            mock_confidences = [0.95, 0.87, 0.92, 0.88, 0.91]
            
            import random
            idx = random.randint(0, len(mock_words) - 1)
            
            return {
                "predicted_word": mock_words[idx],
                "confidence": mock_confidences[idx]
            }
        
        # Actual prediction code would go here
        processed_image = self.preprocess_image(image)
        # predictions = self.model.predict(processed_image)
        # predicted_class = np.argmax(predictions[0])
        # confidence = float(predictions[0][predicted_class])
        
        # For now, return mock data
        return {
            "predicted_word": "HELLO",
            "confidence": 0.95
        }