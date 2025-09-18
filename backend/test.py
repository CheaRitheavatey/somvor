import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np, math

capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)
classifier = Classifier("model_train_my_hands/keras_model.h5", "model_train_my_hands/labels.txt")

space = 20
imgSize = 300

while True:
    success, img = capture.read()
    if not success:
        break

    hands, img = hand_detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        y1 = max(0, y - space)
        y2 = min(img.shape[0], y + h + space)
        x1 = max(0, x - space)
        x2 = min(img.shape[1], x + w + space)
        imgCrop = img[y1:y2, x1:x2]

        white = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        if imgCrop.size != 0:
            ratio = h / w
            if ratio > 1:
                scale = imgSize / h
                wCal = math.ceil(scale * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                gap = math.ceil((imgSize - wCal) / 2)
                white[:, gap:wCal + gap] = imgResize
            else:
                scale = imgSize / w
                hCal = math.ceil(scale * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                gap = math.ceil((imgSize - hCal) / 2)
                white[gap:hCal + gap, :] = imgResize

            prediction, index = classifier.getPrediction(white)
            print(prediction, index)

            cv2.imshow("Hand", imgCrop)
            cv2.imshow("White", white)

    cv2.imshow("Camera", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


# test with keras_model.h5

# import cv2, numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model

# labels = np.load("landmarks/labels.npy")
# model = load_model("sign_model.h5")

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2)

# cap = cv2.VideoCapture(0)
# sequence = []
# MAX_FRAMES = 48

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     res = hands.process(rgb)

#     if res.multi_hand_landmarks:
#         lm = res.multi_hand_landmarks[0]
#         coords = []
#         for pt in lm.landmark:
#             coords.extend([pt.x, pt.y, pt.z])
#         sequence.append(coords)

#         if len(sequence) > MAX_FRAMES:
#             sequence = sequence[-MAX_FRAMES:]

#         if len(sequence) == MAX_FRAMES:
#             X_input = np.expand_dims(sequence, axis=0)
#             pred = model.predict(X_input, verbose=0)[0]
#             word = labels[np.argmax(pred)]
#             cv2.putText(frame, f"{word} ({pred.max():.2f})",
#                         (30,50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0,255,0), 2)

#     cv2.imshow("Sign Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
