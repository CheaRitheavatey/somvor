import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# ---- Load dataset ----
X = np.load("landmarks/X.npy")
y = np.load("landmarks/y.npy")
labels = np.load("landmarks/labels.npy")

# Normalization (optional but helps)
# X shape: (samples, frames, 63) -> scale to 0-1
X = X.astype("float32")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Build LSTM model ----
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save best model while training
checkpoint = ModelCheckpoint(
    "sign_model.h5", monitor="val_accuracy",
    save_best_only=True, verbose=1
)

# ---- Train ----
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    callbacks=[checkpoint]
)

print("✅ Training complete. Best model saved as sign_model.h5")
