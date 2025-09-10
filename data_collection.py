import cv2
from cvzone.HandTrackingModule import HandDetector

# open camera
capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

while True:
    success, img = capture.read()
    
    # detect hand
    hand, img = hand_detector.findHands(img)
    cv2.imshow("img", img)
    cv2.waitKey(1)