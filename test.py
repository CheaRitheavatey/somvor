import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

from cvzone.ClassificationModule import Classifier
classifer = Classifier("model/keras_model.h5", "model/labels.txt")

# for the index below need to put the labels in arr here
labels = []
with open("model/labels.txt", "r") as file:
    for i in file:
        # split by space
        word = i.strip().split(maxsplit=1)
        if len(word) > 1:
            labels.append(word[1])
# open camera
capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

space = 20
imgSize = 300

# folder = "img/See_You_Later"
# if not os.path.exists(folder):
#     os.makedirs(folder)
# count = 0

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
            
            # now add classifier to predict it
            prediction, index = classifer.getPrediction(img)
            print(prediction, index)

        else:
            x = imgSize/w
            hCal = math.ceil(x * h) 
            
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            gap = math.ceil((imgSize - hCal)/2) # to make the img in the center
            white[gap: hCal + gap , :  ] = imgResize

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