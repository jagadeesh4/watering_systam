from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained ML model
model = joblib.load("watering_model.pkl")

# WeatherAPI configurations
API_KEY = 'b0b6fbb82e714fefa7c133211241711'  # Replace with your WeatherAPI key
CITY = 'Coimbatore'
BASE_URL = 'http://api.weatherapi.com/v1/forecast.json'  # Updated to use the forecast endpoint

# ESP32 endpoint for controlling the solenoid valve
ESP32_IP = '192.168.102.1'
  # Replace with your ESP32's endpoint


def get_weather_data(city):
    """Fetch real-time weather data using WeatherAPI."""
    url = f"{BASE_URL}?key={API_KEY}&q={city}&days=1"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Extract required weather data
        temp_avg = data['current']['temp_c']
        temp_max = data['forecast']['forecastday'][0]['day']['maxtemp_c']
        temp_min = data['forecast']['forecastday'][0]['day']['mintemp_c']
        precip = data['current']['precip_mm']
        return temp_avg, temp_max, temp_min, precip
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None, None, None, None


@app.route('/predict', methods=['GET'])
def predict():
    """Predict whether plants need watering based on real-time weather data."""
    # Fetch real-time weather data
    weather_temp_avg, weather_temp_max, weather_temp_min, weather_precip = get_weather_data(CITY)

    if weather_temp_avg is None:
        return jsonify({"error": "Failed to fetch weather data"}), 500

    # Prepare data for prediction
    sensor_data = pd.DataFrame({
        "precip": [weather_precip],
        "temp_avg": [weather_temp_avg],
        "temp_max": [weather_temp_max],
        "temp_min": [weather_temp_min]
    })

    # Make prediction using the trained ML model
    try:
        prediction = model.predict(sensor_data)
        water_plants = int(prediction[0])

        # Toggle the valve based on the prediction
        toggle_valve('on' if water_plants == 1 else 'off')

        return jsonify({
            "water_plants": water_plants,
            "temperature": weather_temp_avg,
            "humidity": weather_precip
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500


def toggle_valve(state):
    """Toggle the solenoid valve via ESP32."""
    try:
        url = f"{ESP32_IP}/toggle_valve?state={state}"
        response = requests.post(url)
        if response.status_code == 200:
            print(f"Valve successfully turned {state}")
        else:
            print(f"Failed to toggle valve. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error toggling valve: {e}")


@app.errorhandler(404)
def page_not_found(e):
    """Custom handler for 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
