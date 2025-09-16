import json
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# ---------- PARAMETERS ----------
DATA_DIR = "kaggle/"  # e.g., "ASL_Dataset"
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 10  # increase if needed
COMMON_LABELS = [
    "Again","Bathroom","Eat","Find","Fine","Good","Hello","I_Love_You",
    "Like","Me","Milk","No","Please","See_You_Later","Sleep","Talk",
    "Thank_You","Understand","Want","What's_Up","Who","Why","Yes","You"
]
# --------------------------------

# ---------- HELPER FUNCTIONS ----------
def load_data(json_file, data_dir, allowed_labels):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    images = []
    labels = []
    
    for item in data:
        label = item['label']
        if label not in allowed_labels:
            continue  # skip labels not in our common set
        img_path = os.path.join(data_dir, item['image'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    
    return np.array(images, dtype="float32"), np.array(labels)

# ---------- LOAD DATA ----------
train_images, train_labels = load_data(os.path.join(DATA_DIR, "MSASL_train.json"), DATA_DIR, COMMON_LABELS)
val_images, val_labels = load_data(os.path.join(DATA_DIR, "MSASL_val.json"), DATA_DIR, COMMON_LABELS)

# Normalize images
train_images /= 255.0
val_images /= 255.0

# Encode labels to integers
le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
val_labels_enc = le.transform(val_labels)

print("Train labels:", train_labels)
print("Encoded train labels:", train_labels_enc)
print("Number of training samples:", len(train_labels))


# One-hot encoding
train_labels_cat = to_categorical(train_labels_enc)
val_labels_cat = to_categorical(val_labels_enc)
num_classes = len(le.classes_)

# ---------- BUILD MODEL ----------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- TRAIN ----------
model.fit(train_images, train_labels_cat, validation_data=(val_images, val_labels_cat),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# ---------- SAVE MODEL ----------
model.save("asl_model_common_words.h5")
print("Model saved as asl_model_common_words.h5")
print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
