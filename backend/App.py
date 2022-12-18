import random

from flask import Flask, request, redirect
import csv
import io
from datetime import datetime as dt, timedelta

app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def index():
    response = []
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        duration = request.form["duration"]

        if file.filename == "":
            return redirect(request.url)

        if file:

            t = file.stream.read()
            content = t.decode()
            file = io.StringIO(content)
            csv_data = csv.reader(file, delimiter=",")
            i = 0
            for row in csv_data:
                i += 1
            print(i)
            model = "Waiting for Akram"
            date_prediction = dt.now().replace(microsecond=0, second=0, minute=0)

            if int(duration) == 7:
                response = predict_range(model=model, start_date=date_prediction, days_count=24 * 28)
            elif int(duration) == 30:
                response = predict_range(model=model, start_date=date_prediction, days_count=24 * 28 * 4)

            return response


def predict_range(model, start_date, days_count):
    response = []
    for i in range(days_count):
        prediction = random.uniform(0, 5)
        response.append([
            int(start_date.timestamp().real) * 1000,
            prediction
        ])
        start_date += timedelta(hours=1)

    return response


app.debug = True
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7777)
