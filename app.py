from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
import os, json

app = Flask(__name__)

# Firebase Init using environment variable for credentials
json_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
cred_dict = json.loads(json_creds)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-database.firebaseio.com/'
})

# Load models
irrigation_model = joblib.load("irrigation_model.pkl")
fertigation_model = joblib.load("random_forest_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    ref = db.reference('/sensor_data')
    sensor_data = ref.order_by_key().limit_to_last(10).get()
    if sensor_data:
        latest = list(sensor_data.values())[-1]
        timestamps = [entry['timestamp'] for entry in sensor_data.values()]
        temperatures = [entry['temperature'] for entry in sensor_data.values()]
        humidities = [entry['humidity'] for entry in sensor_data.values()]
    else:
        latest = {"temperature": None, "humidity": None}
        timestamps = temperatures = humidities = []

    if request.method == "POST":
        plant_name = request.form["plant_name"]
        current_n = float(request.form["n"])
        current_p = float(request.form["p"])
        current_k = float(request.form["k"])
        motor_type = request.form["motor_type"]
        area_value = float(request.form["area_value"])
        area_unit = request.form["area_unit"]

        # Sensor data from Firebase
        temperature = latest['temperature']
        humidity = latest['humidity']

        irrigation_percent = irrigation_model.predict([[temperature, humidity]])[0]
        fertigation_dose = fertigation_model.predict([[current_n, current_p, current_k]])[0]

        result = {
            "plant_name": plant_name,
            "temperature": temperature,
            "humidity": humidity,
            "n": current_n,
            "p": current_p,
            "k": current_k,
            "motor_type": motor_type,
            "area": area_value,
            "area_unit": area_unit,
            "predicted_irrigation": irrigation_percent,
            "recommended_fertilizer": fertigation_dose,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        db.reference('/user_inputs').push(result)

        return render_template("index.html", result=result, latest_sensor_data=latest,
                               timestamps=timestamps, temperatures=temperatures, humidities=humidities)

    return render_template("index.html", latest_sensor_data=latest,
                           timestamps=timestamps, temperatures=temperatures, humidities=humidities)

if __name__ == "__main__":
    app.run(debug=True)
