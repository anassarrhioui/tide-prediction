import random
import numpy as np

from flask_cors import CORS
from flask import Flask, request, redirect
import csv
import io
from datetime import datetime as dt, timedelta

from tensorflow import keras
from keras import activations

app = Flask(__name__)
CORS(app)

batch_size = 50
epochs = 200

loss = keras.losses.MeanSquaredError()
optim = keras.optimizers.Adam(learning_rate=0.0001)
metrics = [keras.metrics.MeanSquaredError()]


@app.route("/predict", methods=["GET", "POST"])
def index():
    response = []
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        duration = request.form["duration"]
        print("duration")
        print(duration)
        if file.filename == "":
            return redirect(request.url)

        if file:
            t = file.stream.read()
            content = t.decode()
            file = io.StringIO(content)
            csv_data = csv.reader(file, delimiter=",")
            data = np.array(list(csv_data), dtype=np.float)
            return perform_prediction(data, duration)
            # print(data[:, 0])
            # print(data[:, 1])
            # x_train, y_train = data[:, 0], data[:, 1]
            #
            # date_prediction = dt.now().replace(microsecond=0, second=0, minute=0)

            # if int(duration) == 7:
            #     model = fit_week(x_train=x_train, y_train=y_train)
            #     response = predict_range(model=model, start_date=date_prediction, days_count=24 * 28)
            # elif int(duration) == 30:
            #     model = fit_month(x_train=x_train, y_train=y_train)
            #     response = predict_range(model=model, start_date=date_prediction, days_count=24 * 28 * 4)

        return response


def get_model( sequence_length : int = 20,  ):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(sequence_length,)),
        # keras.layers.Dense(6, activation=activations.tanh),
        keras.layers.Dense(3, activation=activations.tanh),
        keras.layers.Dense(1, activation=activations.linear)
        ])
    loss = keras.losses.MeanSquaredError()
    # optim = keras.optimizers.Adam(learning_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=0.01)
    metrics = [ keras.metrics.MeanSquaredError()]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    return model

def reshape_data(time_data, height_data, sequence_length=20):
    # -- convert time to number of hours
    # init_time = time_data[0]
    # time_data = ( time_data-init_time )/3600

    x_train = []
    y_train = []

    for i in range(sequence_length, len(time_data)):
        x_train.append(height_data[i - sequence_length:i])
        y_train.append(height_data[i])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return time_data[sequence_length:], x_train, y_train


def train_model(model, x_train, y_train, epochs=100):
    # model.evaluate( x_train, y_train )
    model.fit(x_train, y_train, epochs=epochs, verbose=1)
    # model.evaluate( x_train, y_train )


def predict(model, previous_data, last_date, sequence_length=20, prediction_duration=100):
    y_predicted = list(previous_data[-sequence_length:])
    dates = []

    for i in range(int(prediction_duration)):
        previous_data = [y_predicted[-sequence_length:]]
        y_predicted.append(float(model.predict(previous_data)[0][0]))
        last_date += 1
        dates.append(last_date)

    # y_predicted = np.array(y_predicted)
    return dates, y_predicted[sequence_length:]


def perform_prediction(data, duration):
    SEQ_LEN = 700
    PRED_DUR = 200

    data = np.array(data, dtype=np.float)
    time_data, height_data = data[:, 0], data[:, 1]

    # convert time to number of hours
    time_data = (time_data - time_data[0]) / 3600
    height_data = (height_data - 2.25) / 4.5

    time_data, x_train, y_train = reshape_data(time_data, height_data, sequence_length=SEQ_LEN)
    last_date = time_data[-1]
    model = get_model(SEQ_LEN)
    train_model(model, x_train, y_train, epochs=250)

    dates, y_predicted = predict(model, height_data, last_date, sequence_length=SEQ_LEN,
                                 prediction_duration=PRED_DUR)

    return [[((last_date + i) * 3600 + time_data[0]) * 1000, y_predicted[i] * 4.5 + 2.25] for i in range(PRED_DUR)]


app.debug = True
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7777)
