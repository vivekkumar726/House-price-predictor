import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor 🏠",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv(
    "API_URL",
    "https://house-price-predictor-qskr.onrender.com/"
)


# ── Helper ───────────────────────────────────────────────────────────────────
def format_inr(amount: float) -> str:
    """Format number as Indian currency: ₹12,34,567"""
    amount = int(round(amount, -3))
    if amount >= 10_000_000:
        return f"₹{amount/10_000_000:.2f} Cr"
    elif amount >= 100_000:
        return f"₹{amount/100_000:.2f} L"
    else:
        return f"₹{amount:,}"
    
@st.cache_data
def load_model_info():
    try:
        response = requests.get(f"{API_URL}/info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Cannot connect to FastAPI server.\n\n{e}")
        st.stop()


model_info = load_model_info()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/home.png", width=72)
st.sidebar.title("House Price Predictor")
st.sidebar.markdown("Enter house details to get an **instant price estimate in ₹**.")
st.sidebar.divider()
best = model_info['best_model']
st.sidebar.caption(f"🤖 Model: **{best}**")
st.sidebar.caption(f"📊 R² Score: **{model_info['results'][best]['R2']}**")
st.sidebar.caption(f"📉 Avg Error: **{format_inr(model_info['results'][best]['MAE'])}**")
st.sidebar.divider()
st.sidebar.info("Prices are in **Indian Rupees (₹)**. Dataset: 545 real housing records.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏠 House Price Prediction System")
st.markdown("Predict the **market value of a house in ₹** using ML trained on real housing data.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📐 Size & Structure")
    area     = st.slider("Area (sq ft)", 1650, 16200, 5000, step=50)
    stories  = st.selectbox("Number of Stories", [1, 2, 3, 4], index=1)
    parking  = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)

with col2:
    st.subheader("🛏️ Rooms")
    bedrooms  = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
    furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

with col3:
    st.subheader("✅ Amenities")
    mainroad       = st.checkbox("Main Road Access 🛣️",      value=True)
    guestroom      = st.checkbox("Guest Room 🛋️",            value=False)
    basement       = st.checkbox("Basement 🏚️",              value=False)
    hotwaterheating = st.checkbox("Hot Water Heating 🚿",    value=False)
    airconditioning = st.checkbox("Air Conditioning ❄️",     value=False)
    prefarea       = st.checkbox("Preferred Area 📍",         value=False)

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict Price", type="primary", use_container_width=True):

    payload = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishing
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload
        )

        response.raise_for_status()

        result = response.json()

        pred = result["predicted_price"]
        low = result["price_low"]
        high = result["price_high"]

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "💰 Estimated Price",
            format_inr(pred)
        )

        r2.metric(
            "📉 Low Estimate",
            format_inr(low)
        )

        r3.metric(
            "📈 High Estimate",
            format_inr(high)
        )

        st.success(
            f"🏡 Estimated market value: **{result['formatted']}** "
            f"(±10% range: {format_inr(low)} – {format_inr(high)})"
        )

        with st.expander("📋 Input Summary"):

            display = pd.DataFrame([payload])

            bool_cols = [
                "mainroad",
                "guestroom",
                "basement",
                "hotwaterheating",
                "airconditioning",
                "prefarea"
            ]

            for c in bool_cols:
                display[c] = display[c].map({
                    True: "Yes",
                    False: "No"
                })

            st.dataframe(
                display.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI server.\n\n"
            "Start it using:\n\n"
            "python -m uvicorn app.api.main:app --reload"
        )

    except Exception as e:

        st.error(str(e))

# ── Model comparison ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Model Comparison")

rows = []
for name, metrics in model_info['results'].items():
    rows.append({
        'Model':    name,
        'R² Score': metrics['R2'],
        'RMSE':     format_inr(metrics['RMSE']),
        'MAE':      format_inr(metrics['MAE']),
        'Best ✅':  '✅' if name == model_info['best_model'] else '',
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Scikit-learn · Streamlit · Python  |  ML Portfolio Project by Vivek Kumar")
