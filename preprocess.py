

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def download_stock_data(ticker='AAPL',
                        start='2015-01-01',
                        end='2025-01-01'):
    """
    Download stock data from Yahoo Finance.
    """

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False
    )

    df = df[['Close']]
    df.dropna(inplace=True)

    return df


def scale_data(df):
    """
    Scale closing prices between 0 and 1.
    """

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler


def create_sequences(data, sequence_length=60):
    """
    Create sequences for RNN training.
    """

    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(
        X,
        (X.shape[0], X.shape[1], 1)
    )

    return X, y


def train_test_split_data(
        scaled_data,
        sequence_length=60,
        train_ratio=0.8):
    """
    Split data into train and test sets.
    """

    train_size = int(len(scaled_data) * train_ratio)

    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size-sequence_length:]

    X_train, y_train = create_sequences(
        train_data,
        sequence_length
    )

    X_test, y_test = create_sequences(
        test_data,
        sequence_length
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


def load_data(
        ticker='AAPL',
        start='2015-01-01',
        end='2025-01-01',
        sequence_length=60):

    df = download_stock_data(
        ticker,
        start,
        end
    )

    scaled_data, scaler = scale_data(df)

    X_train, y_train, X_test, y_test = train_test_split_data(
        scaled_data,
        sequence_length
    )

    return (
        df,
        scaler,
        X_train,
        y_train,
        X_test,
        y_test
    )


if __name__ == "__main__":

    df, scaler, X_train, y_train, X_test, y_test = load_data()

    print("Dataset Shape :", df.shape)
    print("Training Samples :", X_train.shape)
    print("Testing Samples :", X_test.shape)