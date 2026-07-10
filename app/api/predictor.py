import os
import json
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE,
    "..",
    "..",
    "model",
    "pipeline.pkl"
)

INFO_PATH = os.path.join(
    BASE,
    "..",
    "..",
    "model",
    "model_info.json"
)

pipeline = joblib.load(MODEL_PATH)

with open(INFO_PATH) as f:
    model_info = json.load(f)


def format_inr(amount):
    amount = int(round(amount, -3))

    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Cr"

    elif amount >= 100000:
        return f"₹{amount/100000:.2f} L"

    return f"₹{amount:,}"


def predict_price(request):

    area = request.area
    bedrooms = request.bedrooms

    df = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": request.bathrooms,
        "stories": request.stories,

        "mainroad": int(request.mainroad),
        "guestroom": int(request.guestroom),
        "basement": int(request.basement),
        "hotwaterheating": int(request.hotwaterheating),
        "airconditioning": int(request.airconditioning),

        "parking": request.parking,
        "prefarea": int(request.prefarea),

        "furnishingstatus": request.furnishingstatus,

        "area_per_bedroom": area / max(bedrooms, 1),
        "total_rooms": bedrooms + request.bathrooms,
    }])

    price = float(pipeline.predict(df)[0])

    return {
        "predicted_price": round(price, 2),
        "price_low": round(price * 0.90, 2),
        "price_high": round(price * 1.10, 2),
        "currency": "INR",
        "formatted": format_inr(price),
        "model": model_info["best_model"],
    }