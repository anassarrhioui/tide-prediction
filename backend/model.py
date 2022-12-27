from tensorflow import keras
from keras import activations
import numpy as np


def get_model(sequence_length: int = 20, ):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(sequence_length,)),
        keras.layers.Dense(3, activation=activations.tanh),
        keras.layers.Dense(1, activation=activations.linear)
    ])
    loss = keras.losses.MeanSquaredError()
    optim = keras.optimizers.SGD(learning_rate=0.01)
    metrics = [keras.metrics.MeanSquaredError()]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    return model


def reshape_data(time_data, height_data, sequence_length=20):
    x_train = []
    y_train = []

    for i in range(sequence_length, len(time_data)):
        x_train.append(height_data[i - sequence_length:i])
        y_train.append(height_data[i])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return time_data[sequence_length:], x_train, y_train


def train_model(model, x_train, y_train, epochs=100):
    model.fit(x_train, y_train, epochs=epochs, verbose=1)


def predict(model, previous_data, last_date, sequence_length=20, prediction_duration=100):
    y_predicted = list(previous_data[-sequence_length:])
    dates = []

    for i in range(int(prediction_duration)):
        previous_data = [y_predicted[-sequence_length:]]
        y_predicted.append(float(model.predict(previous_data)[0][0]))
        last_date += 1
        dates.append(last_date)

    return dates, y_predicted[sequence_length:]


def perform_predictions(data, duration, sequence_length=100, prediction_duration=200):
    data = np.array(data, dtype=np.float)
    time_data, height_data = data[:, 0], data[:, 1]

    time_data = (time_data - time_data[0]) / 3600
    height_data = (height_data - 2.25) / 4.5

    time_data, x_train, y_train = reshape_data(time_data, height_data, sequence_length=sequence_length)
    last_date = time_data[-1]
    model = get_model(sequence_length)
    train_model(model, x_train, y_train, epochs=250)

    dates, y_predicted = predict(model, height_data, last_date, sequence_length=sequence_length,
                                 prediction_duration=prediction_duration)

    return [[((last_date + i) * 3600 + time_data[0]) * 1000, y_predicted[i] * 4.5 + 2.25] for i in range(prediction_duration)]
