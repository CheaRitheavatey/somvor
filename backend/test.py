# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier

# import numpy as np
# import math
# import time
# import os

    
# # from tensorflow.keras.models import load_model
# # model = load_model("modelss/keras_model.h5")
# # model.save("keras_model_tf29.h5", save_format="h5")

# # open camera
# capture = cv2.VideoCapture(0)
# hand_detector = HandDetector(maxHands=1)

# classifier = Classifier("modelss/keras_model.h5", "modelss/labels.txt")

# space = 20
# imgSize = 300

# # folder = "img/See_You_Later"
# # if not os.path.exists(folder):
# #     os.makedirs(folder)
# # count = 0

# labels = [
# "Again",
# "Bathroom",
# "Eat",
# "Find",
# "Fine",
# "Good",
# "Hello",
# "I_Love_You",
# "Like",
# "Me",
# "Milk",
# "No",
# "Please",
# "See_You_Later",
# "Sleep",
# "Talk",
# "Thank_You",
# "Understand",
# "Want",
# "What's_Up",
# "Who",
# "Why",
# "Yes",
# "You"

# ]

# while True:
#     success, img = capture.read()
    
#     # detect hand
#     hands, img = hand_detector.findHands(img)
    
#     # crop and only have hand in the frame
#     if hands:
#         hand = hands[0] #because we only detect 1 hand first
        
#         # get bounding box from the hand
#         x,y, w,h = hand['bbox']
        
#         # crop img based on dimension we want
#         imgCrop = img[y - space: y + h + space, x - space: x + w + space]
        
#         # create a hand img with white bg
#         white = np.ones((imgSize,imgSize,3),np.uint8 )*255
        
#         # put the hand on the white aka map all the size corner of the imgcrop to white aka overlay the hand on the bg white
#         # white[0 : imgCrop.shape[0], 0: imgCrop.shape[1]] = imgCrop
        
#         ratio = h/w
#         # if ration < 1 -> w > h -> stretch h; else do the oppsite
#         if ratio > 1:
#             x = imgSize/h
#             wCal = math.ceil(x * w) 
            
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             gap = math.ceil((imgSize - wCal)/2) # to make the img in the center
#             white[ : , gap: wCal + gap] = imgResize
            

#         else:
#             x = imgSize/w
#             hCal = math.ceil(x * h) 
            
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             gap = math.ceil((imgSize - hCal)/2) # to make the img in the center
#             white[gap: hCal + gap , :  ] = imgResize
#         prediction, index = classifier.getPrediction(white)
#         print(prediction, index)

#         cv2.imshow("imgCrop", imgCrop)
#         cv2.imshow("imgwhite", white)
        
#     cv2.imshow("img", img)
#     cv2.waitKey(1)
    
#     # if key == ord("s"):
#     #     count+= 1
#     #     cv2.imwrite(f"{folder}/Img_{time.time()}.jpg", white)
#     #     print(count)
#     # elif key == ord("q"):
#     #     break



# test with keras_model.h5

import cv2, numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

labels = np.load("landmarks/labels.npy")
model = load_model("sign_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)
sequence = []
MAX_FRAMES = 48

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        coords = []
        for pt in lm.landmark:
            coords.extend([pt.x, pt.y, pt.z])
        sequence.append(coords)

        if len(sequence) > MAX_FRAMES:
            sequence = sequence[-MAX_FRAMES:]

        if len(sequence) == MAX_FRAMES:
            X_input = np.expand_dims(sequence, axis=0)
            pred = model.predict(X_input, verbose=0)[0]
            word = labels[np.argmax(pred)]
            cv2.putText(frame, f"{word} ({pred.max():.2f})",
                        (30,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
