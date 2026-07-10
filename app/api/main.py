from fastapi import FastAPI
from .schemas import HouseRequest, PredictionResponse
from .predictor import predict_price, model_info

app = FastAPI(
    title="House Price Prediction API",
    version="3.0",
    description="Predict house prices using Machine Learning."
)


@app.get("/")
def home():

    return {
        "message": "House Price Prediction API",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/info")
def info():

    return model_info


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: HouseRequest):

    return predict_price(data)