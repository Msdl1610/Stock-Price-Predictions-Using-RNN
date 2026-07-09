# 📈 Stock Price Prediction Using RNN

## Overview

This project predicts stock closing prices using a **Recurrent Neural Network (RNN)** built with TensorFlow/Keras. Historical stock data is downloaded from Yahoo Finance, preprocessed, and used to train an RNN model that forecasts future stock prices. An interactive Streamlit dashboard allows users to visualize historical prices, compare actual and predicted values, and view model performance metrics.

---

## Features

* Download historical stock data using **Yahoo Finance**
* Data preprocessing and normalization using **MinMaxScaler**
* Sequence generation for time-series forecasting
* RNN model built using TensorFlow/Keras
* Model training and saving
* Predict future stock prices
* Interactive Streamlit dashboard
* Historical closing price visualization
* Actual vs Predicted comparison graph
* Model evaluation using MAE, MSE, RMSE, and R² Score
* Download prediction results as a CSV file

---

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Matplotlib
* yfinance

---

## Project Structure

```
Stock_Price_RNN/
│
├── app.py
├── preprocess.py
├── model.py
├── train.py
├── predict.py
├── README.md
│
├── models/
│   └── rnn_model.keras
│
├── data/
│
└── outputs/
```

---

## Workflow

```
Historical Stock Data
          ↓
Data Preprocessing
          ↓
Normalization
          ↓
Sequence Creation
          ↓
RNN Model Training
          ↓
Prediction
          ↓
Performance Evaluation
          ↓
Streamlit Dashboard
```

---

## Model Architecture

```
Input Layer
      ↓
SimpleRNN (64 Units)
      ↓
Dropout (0.2)
      ↓
SimpleRNN (32 Units)
      ↓
Dropout (0.2)
      ↓
Dense (16)
      ↓
Output Layer (1 Unit)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Stock_Price_RNN.git
```

Move into the project directory:

```bash
cd Stock_Price_RNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

This creates:

```
models/rnn_model.keras
```

---

## Run Prediction

```bash
python predict.py
```

---

## Launch the Dashboard

```bash
streamlit run app.py
```

---

## Dashboard Features

* Select stock symbol
* Select date range
* Historical stock price chart
* Actual vs Predicted graph
* Next-day stock price prediction
* Performance metrics
* Download prediction results

---

## Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---


## Applications

* Financial forecasting
* Investment analysis
* Time-series prediction
* Educational deep learning project
* Portfolio management

---




## Author

**Deepak M**

