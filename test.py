import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import numpy as np
import math
import time
import os

    
# from tensorflow.keras.models import load_model
# model = load_model("modelss/keras_model.h5")
# model.save("keras_model_tf29.h5", save_format="h5")

# open camera
capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

classifier = Classifier("modelss/keras_model.h5", "modelss/labels.txt")

space = 20
imgSize = 300

# folder = "img/See_You_Later"
# if not os.path.exists(folder):
#     os.makedirs(folder)
# count = 0

labels = [
"Again",
"Bathroom",
"Eat",
"Find",
"Fine",
"Good",
"Hello",
"I_Love_You",
"Like",
"Me",
"Milk",
"No",
"Please",
"See_You_Later",
"Sleep",
"Talk",
"Thank_You",
"Understand",
"Want",
"What's_Up",
"Who",
"Why",
"Yes",
"You"

]

while True:
    success, img = capture.read()
    
    # detect hand
    hands, img = hand_detector.findHands(img)
    
    # crop and only have hand in the frame
    if hands:
        hand = hands[0] #because we only detect 1 hand first
        
        # get bounding box from the hand
        x,y, w,h = hand['bbox']
        
        # crop img based on dimension we want
        imgCrop = img[y - space: y + h + space, x - space: x + w + space]
        
        # create a hand img with white bg
        white = np.ones((imgSize,imgSize,3),np.uint8 )*255
        
        # put the hand on the white aka map all the size corner of the imgcrop to white aka overlay the hand on the bg white
        # white[0 : imgCrop.shape[0], 0: imgCrop.shape[1]] = imgCrop
        
        ratio = h/w
        # if ration < 1 -> w > h -> stretch h; else do the oppsite
        if ratio > 1:
            x = imgSize/h
            wCal = math.ceil(x * w) 
            
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            gap = math.ceil((imgSize - wCal)/2) # to make the img in the center
            white[ : , gap: wCal + gap] = imgResize
            

        else:
            x = imgSize/w
            hCal = math.ceil(x * h) 
            
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            gap = math.ceil((imgSize - hCal)/2) # to make the img in the center
            white[gap: hCal + gap , :  ] = imgResize
        prediction, index = classifier.getPrediction(white)
        print(prediction, index)

        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgwhite", white)
        
    cv2.imshow("img", img)
    cv2.waitKey(1)
    
    # if key == ord("s"):
    #     count+= 1
    #     cv2.imwrite(f"{folder}/Img_{time.time()}.jpg", white)
    #     print(count)
    # elif key == ord("q"):
    #     break

