# train.py

import os
import matplotlib.pyplot as plt

from preprocess import load_data
from model import build_rnn_model


# -----------------------------
# Configuration
# -----------------------------
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

SEQUENCE_LENGTH = 60
EPOCHS = 25
BATCH_SIZE = 32

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.keras")


# -----------------------------
# Load Data
# -----------------------------
print("Loading stock data...")

df, scaler, X_train, y_train, X_test, y_test = load_data(
    ticker=TICKER,
    start=START_DATE,
    end=END_DATE,
    sequence_length=SEQUENCE_LENGTH
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# -----------------------------
# Build Model
# -----------------------------
model = build_rnn_model(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

model.summary()


# -----------------------------
# Train Model
# -----------------------------
print("\nTraining model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)


# -----------------------------
# Evaluate Model
# -----------------------------
loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Evaluation")
print("-----------------------")
print(f"Test Loss : {loss:.6f}")
print(f"Test MAE  : {mae:.6f}")


# -----------------------------
# Save Model
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_PATH)

print(f"\nModel saved successfully to:\n{MODEL_PATH}")


# -----------------------------
# Plot Training History
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.grid(True)

plt.show()