import cv2
import numpy as np
import math

try:
    from cvzone.HandTrackingModule import HandDetector
    from cvzone.ClassificationModule import Classifier
    CVZONE_AVAILABLE = True
except ImportError:
    CVZONE_AVAILABLE = False

class GestureDetector:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.space = 20
        self.imgSize = 300
        
        # Labels mapping
        self.labels = [
            "again", "bad", "drink", "eat", "good", "hello", "help", "like", 
            "me", "milk", "no", "please", "see", "sleep", "thank you", 
            "understand", "want", "what", "when", "where", "who", "why", "yes", "you"
        ]
        
        # English to Khmer translations
        self.translations = {
            "again": "ម្តងទៀត",
            "bad": "មិនល្អ",
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
            "see": "មើល",
            "sleep": "គេង",
            "thank you": "អរគុណ",
            "understand": "យល់",
            "want": "ចង់",
            "what": "អ្វី",
            "when": "ពេលណា",
            "where": "ណា",
            "who": "នរណា",
            "why": "ហេតុអ្វី",
            "yes": "បាទ/ចាស",
            "you": "អ្នក"
        }
        
        if CVZONE_AVAILABLE:
            try:
                self.hand_detector = HandDetector(maxHands=1)
                self.classifier = Classifier("model_train_my_hands/keras_model.h5", 
                                           "model_train_my_hands/labels.txt")
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model_loaded = False
        else:
            print("CVZone not available. Install with: pip install cvzone")
            self.model_loaded = False
            
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for gesture detection"""
        self.confidence_threshold = threshold
        
    def get_gesture_text(self, index, language="english"):
        """Get gesture text in specified language"""
        if 0 <= index < len(self.labels):
            gesture_word = self.labels[index]
            if language == "khmer":
                return self.translations.get(gesture_word, gesture_word)
            return gesture_word
        return "Unknown"
        
    def process_frame(self, frame):
        """Process frame and detect gestures with improved accuracy"""
        if not self.model_loaded:
            return frame, None, 0.0
            
        try:
            hands, processed_frame = self.hand_detector.findHands(frame)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Improved cropping with better boundary handling
                y1 = max(0, y - self.space)
                y2 = min(frame.shape[0], y + h + self.space)
                x1 = max(0, x - self.space)
                x2 = min(frame.shape[1], x + w + self.space)
                imgCrop = frame[y1:y2, x1:x2]

                if imgCrop.size != 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    # Create white background
                    white = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                    
                    # Improved aspect ratio handling
                    crop_h, crop_w = imgCrop.shape[:2]
                    ratio = crop_h / crop_w
                    
                    if ratio > 1:
                        # Height is greater than width
                        scale = self.imgSize / crop_h
                        wCal = math.ceil(scale * crop_w)
                        if wCal > 0:
                            imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                            gap = math.ceil((self.imgSize - wCal) / 2)
                            white[:, gap:gap + wCal] = imgResize
                    else:
                        # Width is greater than or equal to height
                        scale = self.imgSize / crop_w
                        hCal = math.ceil(scale * crop_h)
                        if hCal > 0:
                            imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                            gap = math.ceil((self.imgSize - hCal) / 2)
                            white[gap:gap + hCal, :] = imgResize

                    # Get prediction with confidence
                    prediction, index = self.classifier.getPrediction(white, draw=False)
                    
                    # Calculate confidence (max probability)
                    confidence = max(prediction) if prediction else 0.0
                    
                    # Only return result if confidence meets threshold
                    if confidence >= self.confidence_threshold:
                        return processed_frame, index, confidence
                        
            return processed_frame, None, 0.0
            
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            return frame, None, 0.0