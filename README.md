# 🏠 House Price Prediction System (INR ₹)

> An end-to-end Machine Learning project that predicts house prices in **Indian Rupees (₹)** using Ridge Regression — with a REST API (Flask) and an interactive web app (Streamlit).

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikitlearn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![Tests](https://img.shields.io/badge/Tests-10%20passed-brightgreen)](#testing)
[![Currency](https://img.shields.io/badge/Currency-INR%20₹-green)](https://en.wikipedia.org/wiki/Indian_rupee)

---

## 📌 Live Demo
🚀 **[Try the app →](https://your-app.streamlit.app)** *(deploy free on Streamlit Cloud)*

---

## 🧠 Project Overview

```
Data (545 records) → EDA → Preprocessing → Feature Engineering
  → Train 3 Models → Evaluate → Best Model (Ridge, R²=0.63)
    → sklearn Pipeline → Flask API → Streamlit UI → Deploy
```

| Component | Details |
|---|---|
| Dataset | 545 real housing records, 13 features |
| Target | House price in ₹ (₹17.5L – ₹1.33Cr) |
| Features | Area, bedrooms, bathrooms, stories, amenities, furnishing |
| Best Model | Ridge Regression (R²=0.63) |
| API | Flask REST with `/predict` endpoint |
| UI | Streamlit web app with ₹ formatting |
| Tests | 10 unit + integration tests (pytest) |

---

## 📁 Project Structure

```
house-price-predictor/
│
├── data/
│   └── housing.csv              # 545 real housing records
│
├── model/
│   ├── pipeline.pkl             # Trained sklearn pipeline
│   └── model_info.json          # Metrics & metadata
│
├── app/
│   ├── streamlit_app.py         # Interactive web UI (₹)
│   └── flask_api.py             # REST API
│
├── notebooks/
│   └── eda.py                   # Full EDA with plots
│
├── tests/
│   └── test_model.py            # 10 pytest tests
│
├── train.py                     # Retrain script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain model
python train.py

# 4a. Run Streamlit UI
streamlit run app/streamlit_app.py

# 4b. OR run Flask API
python app/flask_api.py
```

---

## 🌐 API Usage

### Predict House Price

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 5000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 1,
    "prefarea": "no",
    "furnishingstatus": "semi-furnished"
  }'
```

**Response:**
```json
{
  "predicted_price": 4823500.00,
  "price_low":       4341150.00,
  "price_high":      5305850.00,
  "currency":        "INR",
  "formatted":       "₹48,23,500",
  "model":           "Ridge"
}
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/info` | Model metrics |
| POST | `/predict` | Predict price in ₹ |

---

## 📊 Model Performance

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **Ridge** ✅ | **0.6305** | **₹13.67L** | **₹9.88L** |
| GradientBoosting | 0.6128 | ₹13.99L | ₹10.33L |
| RandomForest | 0.6065 | ₹14.10L | ₹10.58L |

> R² of 0.63 on 545 records is reasonable — more data would improve performance significantly.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

```
tests/test_model.py::test_model_loads                    PASSED
tests/test_model.py::test_prediction_returns_float       PASSED
tests/test_model.py::test_prediction_in_realistic_range  PASSED
tests/test_model.py::test_prediction_is_positive         PASSED
tests/test_model.py::test_luxury_house_higher            PASSED
tests/test_model.py::test_model_info_json                PASSED
tests/test_model.py::test_flask_api_predict              PASSED
tests/test_model.py::test_flask_api_missing_field        PASSED
tests/test_model.py::test_flask_health                   PASSED
tests/test_model.py::test_flask_accepts_binary_integers  PASSED

10 passed in 1.04s
```

---

## 🚀 Deploy Free on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Main file: `app/streamlit_app.py`
4. Click **Deploy** — get a live public link!

---

## 🔧 Tech Stack

`scikit-learn` · `pandas` · `numpy` · `Flask` · `Streamlit` · `matplotlib` · `seaborn` · `pytest` · `joblib`

---

## 👤 Author

**Vivek Kumar** — B.Tech CSE · Gayatri Vidya Parishad College of Engineering  
[GitHub](https://github.com/vivekkumar726) · [LeetCode](https://leetcode.com/vivek_kumar_7)
