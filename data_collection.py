import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# open camera
capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

space = 20
imgSize = 300

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
        white[0 : imgCrop.shape[0], 0: imgCrop.shape[1]] = imgCrop
        
        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgwhite", white)
        
    cv2.imshow("img", img)
    cv2.waitKey(1)