"""
train.py — Train and save the best house price prediction model.
Dataset: Housing.csv (545 records, prices in INR ₹)
Run: python train.py
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "housing.csv")
MODEL_DIR = os.path.join(BASE, "model")

BINARY_COLS = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
NUM_COLS    = ['area','bedrooms','bathrooms','stories','mainroad','guestroom',
               'basement','hotwaterheating','airconditioning','parking','prefarea',
               'area_per_bedroom','total_rooms']
CAT_COLS    = ['furnishingstatus']

MODELS = {
    'Ridge':            Ridge(alpha=100),
    'RandomForest':     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42),
}


def load_and_engineer(path: str):
    df = pd.read_csv(path)
    for col in BINARY_COLS:
        df[col] = (df[col].str.lower() == 'yes').astype(int)
    df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0, 1)
    df['total_rooms']      = df['bedrooms'] + df['bathrooms']
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y


def build_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
    ])


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    return {
        'RMSE': round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        'R2':   round(float(r2_score(y_test, pred)), 4),
        'MAE':  round(float(mean_absolute_error(y_test, pred)), 2),
    }


def train():
    print("Loading data…")
    X, y = load_and_engineer(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results    = {}
    best_score = -np.inf
    best_name  = None
    best_pipe  = None

    for name, model in MODELS.items():
        print(f"Training {name}…")
        pipe = Pipeline([('pre', build_preprocessor()), ('model', model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        print(f"  RMSE=₹{metrics['RMSE']:>12,.0f}  R²={metrics['R2']:.4f}  MAE=₹{metrics['MAE']:>12,.0f}")
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_name  = name
            best_pipe  = pipe

    print(f"\n✅ Best model: {best_name}  (R²={best_score:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, 'pipeline.pkl'))
    info = {
        'best_model':   best_name,
        'results':      results,
        'num_features': NUM_COLS,
        'cat_features': CAT_COLS,
        'binary_cols':  BINARY_COLS,
    }
    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print("✅ pipeline.pkl and model_info.json saved to /model/")
    return info


if __name__ == '__main__':
    train()
