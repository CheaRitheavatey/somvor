# backend/app/utils/image_processor.py
import cv2
import numpy as np
from PIL import Image

def process_image_for_prediction(image):
    """
    Process image for model prediction
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR (if using OpenCV)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize
    img_array = cv2.resize(img_array, (224, 224))
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def base64_to_image(base64_string):
    """
    Convert base64 string to PIL Image
    """
    import base64
    from io import BytesIO
    
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image.convert('RGB')