from flask_cors import CORS
from flask import Flask, request, redirect
import csv
import io
from datetime import datetime as dt, timedelta

from model import *

app = Flask(__name__)
CORS(app)


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

        return response



app.debug = True
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7777)
