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
            print(data[:, 0])
            print(data[:, 1])
            x_train, y_train = data[:, 0], data[:, 1]

            date_prediction = dt.now().replace(microsecond=0, second=0, minute=0)

            if int(duration) == 7:
                model = fit_week(x_train=x_train, y_train=y_train)
                response = predict_range(model=model, start_date=date_prediction, days_count=24 * 28)
            elif int(duration) == 30:
                model = fit_month(x_train=x_train, y_train=y_train)
                response = predict_range(model=model, start_date=date_prediction, days_count=24 * 28 * 4)

            return response


def predict_range(model, start_date, days_count):
    response = []
    for i in range(days_count):
        timestamp_prediction = int(start_date.timestamp().real)
        prediction = model.predict([timestamp_prediction])
        response.append([
            timestamp_prediction * 1000,
            prediction.tolist()[0][0]
        ])
        start_date += timedelta(hours=1)
    print("response")
    print(response)
    return response


def fit_month(x_train, y_train):
    month_model = keras.models.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(1, activation=activations.tanh),
        keras.layers.Dense(1, activation=activations.linear)
    ])
    month_model.compile(loss=loss, optimizer=optim, metrics=metrics)
    month_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    return month_model


def fit_week(x_train, y_train):
    week_model = keras.models.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(6, activation=activations.tanh),
        keras.layers.Dense(1, activation=activations.linear)
    ])

    week_model.compile(loss=loss, optimizer=optim, metrics=metrics)
    week_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    print("Fit Done")
    return week_model


app.debug = True
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7777)
