# predict.py

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import load_data


# -----------------------------
# Configuration
# -----------------------------
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

SEQUENCE_LENGTH = 60

MODEL_PATH = "models/rnn_model.keras"


# -----------------------------
# Load Dataset
# -----------------------------
df, scaler, X_train, y_train, X_test, y_test = load_data(
    ticker=TICKER,
    start=START_DATE,
    end=END_DATE,
    sequence_length=SEQUENCE_LENGTH
)


# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")


# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(X_test)

# Convert scaled values back to original prices
predictions = scaler.inverse_transform(predictions)

actual_prices = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# -----------------------------
# Evaluation Metrics
# -----------------------------
mae = mean_absolute_error(actual_prices, predictions)
mse = mean_squared_error(actual_prices, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predictions)

print("\nModel Performance")
print("-" * 30)
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# -----------------------------
# Next Day Prediction
# -----------------------------
last_sequence = X_test[-1].reshape(1, SEQUENCE_LENGTH, 1)

next_day_scaled = model.predict(last_sequence)

next_day_price = scaler.inverse_transform(next_day_scaled)

print("\nPredicted Next Closing Price")
print("-" * 30)
print(f"${next_day_price[0][0]:.2f}")


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    actual_prices,
    color="blue",
    label="Actual Price"
)

plt.plot(
    predictions,
    color="red",
    label="Predicted Price"
)

plt.title(f"{TICKER} Stock Price Prediction Using RNN")

plt.xlabel("Time")

plt.ylabel("Closing Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()