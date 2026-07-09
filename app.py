# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import load_data

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Stock Price Prediction using RNN",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction using RNN")
st.write("Predict stock closing prices using a Recurrent Neural Network.")

# ---------------------------------------
# Sidebar
# ---------------------------------------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Symbol", "AAPL")

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime("2025-01-01")
)

sequence_length = st.sidebar.slider(
    "Sequence Length",
    30,
    100,
    60
)

# ---------------------------------------
# Load Data
# ---------------------------------------

with st.spinner("Loading Data..."):

    df, scaler, X_train, y_train, X_test, y_test = load_data(
        ticker=ticker,
        start=str(start_date),
        end=str(end_date),
        sequence_length=sequence_length
    )

st.success("Data Loaded Successfully!")

# ---------------------------------------
# Dataset Preview
# ---------------------------------------

st.subheader("Dataset")

st.dataframe(df.tail())

# ---------------------------------------
# Historical Price Chart
# ---------------------------------------

st.subheader("Historical Closing Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# Load Model
# ---------------------------------------

model = load_model("models/rnn_model.keras")

# ---------------------------------------
# Predictions
# ---------------------------------------

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# ---------------------------------------
# Metrics
# ---------------------------------------

mae = mean_absolute_error(actual, predictions)

mse = mean_squared_error(actual, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(actual, predictions)

st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", f"{mae:.2f}")

c2.metric("MSE", f"{mse:.2f}")

c3.metric("RMSE", f"{rmse:.2f}")

c4.metric("R² Score", f"{r2:.4f}")

# ---------------------------------------
# Actual vs Predicted
# ---------------------------------------

st.subheader("Actual vs Predicted")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        y=actual.flatten(),
        mode="lines",
        name="Actual"
    )
)

fig2.add_trace(
    go.Scatter(
        y=predictions.flatten(),
        mode="lines",
        name="Predicted"
    )
)

fig2.update_layout(
    xaxis_title="Days",
    yaxis_title="Closing Price",
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------
# Next Day Prediction
# ---------------------------------------

last_sequence = X_test[-1].reshape(1, sequence_length, 1)

next_price = model.predict(last_sequence)

next_price = scaler.inverse_transform(next_price)

st.subheader("Next Day Prediction")

st.success(
    f"Predicted Closing Price: ${next_price[0][0]:.2f}"
)

# ---------------------------------------
# Download Predictions
# ---------------------------------------

result = pd.DataFrame({
    "Actual": actual.flatten(),
    "Predicted": predictions.flatten()
})

csv = result.to_csv(index=False)

st.download_button(
    label="Download Prediction Results",
    data=csv,
    file_name="stock_predictions.csv",
    mime="text/csv"
)