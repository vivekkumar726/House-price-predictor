"""
tests/test_model.py — Unit + integration tests
Run: pytest tests/ -v
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import json

BASE = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(os.path.join(BASE, "model", "pipeline.pkl"))


@pytest.fixture(scope="module")
def sample_input():
    return pd.DataFrame([{
        'area': 5000, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
        'mainroad': 1, 'guestroom': 0, 'basement': 0,
        'hotwaterheating': 0, 'airconditioning': 1, 'parking': 1,
        'prefarea': 0, 'furnishingstatus': 'semi-furnished',
        'area_per_bedroom': 5000/3, 'total_rooms': 5,
    }])


def test_model_loads(pipeline):
    assert pipeline is not None


def test_prediction_returns_float(pipeline, sample_input):
    pred = pipeline.predict(sample_input)
    assert isinstance(float(pred[0]), float)


def test_prediction_in_realistic_range(pipeline, sample_input):
    pred = float(pipeline.predict(sample_input)[0])
    # INR range: 17.5L to 1.33Cr
    assert 1_000_000 < pred < 20_000_000, f"Out of range: ₹{pred:,.0f}"


def test_prediction_is_positive(pipeline, sample_input):
    pred = float(pipeline.predict(sample_input)[0])
    assert pred > 0


def test_luxury_house_higher(pipeline):
    luxury = pd.DataFrame([{
        'area': 14000, 'bedrooms': 5, 'bathrooms': 4, 'stories': 4,
        'mainroad': 1, 'guestroom': 1, 'basement': 1,
        'hotwaterheating': 1, 'airconditioning': 1, 'parking': 3,
        'prefarea': 1, 'furnishingstatus': 'furnished',
        'area_per_bedroom': 2800, 'total_rooms': 9,
    }])
    budget = pd.DataFrame([{
        'area': 2000, 'bedrooms': 2, 'bathrooms': 1, 'stories': 1,
        'mainroad': 0, 'guestroom': 0, 'basement': 0,
        'hotwaterheating': 0, 'airconditioning': 0, 'parking': 0,
        'prefarea': 0, 'furnishingstatus': 'unfurnished',
        'area_per_bedroom': 1000, 'total_rooms': 3,
    }])
    assert pipeline.predict(luxury)[0] > pipeline.predict(budget)[0]


def test_model_info_json():
    with open(os.path.join(BASE, "model", "model_info.json")) as f:
        info = json.load(f)
    assert "best_model" in info
    assert "results" in info
    best = info["best_model"]
    assert info["results"][best]["R2"] > 0.50

