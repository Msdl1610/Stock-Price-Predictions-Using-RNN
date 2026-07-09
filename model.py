# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


def build_rnn_model(input_shape):
    """
    Build and compile a Simple RNN model.
    """

    model = Sequential()

    # First RNN layer
    model.add(
        SimpleRNN(
            units=64,
            return_sequences=True,
            input_shape=input_shape
        )
    )
    model.add(Dropout(0.2))

    # Second RNN layer
    model.add(
        SimpleRNN(
            units=32,
            return_sequences=False
        )
    )
    model.add(Dropout(0.2))

    # Fully connected layer
    model.add(Dense(16, activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


if __name__ == "__main__":

    model = build_rnn_model((60, 1))

    model.summary()