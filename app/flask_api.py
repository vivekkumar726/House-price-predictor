from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

BASE      = os.path.dirname(os.path.abspath(__file__))
pipeline  = joblib.load(os.path.join(BASE, "..", "model", "pipeline.pkl"))
with open(os.path.join(BASE, "..", "model", "model_info.json")) as f:
    model_info = json.load(f)

REQUIRED = [
    'area', 'bedrooms', 'bathrooms', 'stories',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
]


def to_binary(val) -> int:
    """Accept yes/no strings or 0/1 integers."""
    if isinstance(val, str):
        return 1 if val.strip().lower() == 'yes' else 0
    return int(bool(val))


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "House Price Predictor API (INR)",
        "version": "2.0",
        "endpoints": {
            "POST /predict": "Predict house price in ₹",
            "GET  /health":  "Health check",
            "GET  /info":    "Model info & metrics",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/info", methods=["GET"])
def info():
    return jsonify(model_info)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in REQUIRED if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        bedrooms = int(data['bedrooms'])
        area     = float(data['area'])

        df = pd.DataFrame([{
            'area':             area,
            'bedrooms':         bedrooms,
            'bathrooms':        int(data['bathrooms']),
            'stories':          int(data['stories']),
            'mainroad':         to_binary(data['mainroad']),
            'guestroom':        to_binary(data['guestroom']),
            'basement':         to_binary(data['basement']),
            'hotwaterheating':  to_binary(data['hotwaterheating']),
            'airconditioning':  to_binary(data['airconditioning']),
            'parking':          int(data['parking']),
            'prefarea':         to_binary(data['prefarea']),
            'furnishingstatus': str(data['furnishingstatus']),
            'area_per_bedroom': area / max(bedrooms, 1),
            'total_rooms':      bedrooms + int(data['bathrooms']),
        }])

        price = float(pipeline.predict(df)[0])

        return jsonify({
            "predicted_price": round(price, 2),
            "price_low":       round(price * 0.90, 2),
            "price_high":      round(price * 1.10, 2),
            "currency":        "INR",
            "formatted":       f"₹{price:,.0f}",
            "model":           model_info["best_model"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
