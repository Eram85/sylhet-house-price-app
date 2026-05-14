import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sylhet House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (YOUR DESIGN KEPT)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 50%, #0f1923 100%);
    color: #e8dcc8;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(200, 160, 80, 0.2);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #c8a050;
    margin: 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: #8a9bb0;
    text-transform: uppercase;
}

/* Input cards */
.input-panel {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 1.2rem;
}

/* Result card */
.result-card {
    background: rgba(200,160,80,0.1);
    border: 1px solid rgba(200,160,80,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-price {
    font-size: 2.5rem;
    color: #c8a050;
    font-weight: bold;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #c8a050, #a07030);
    color: black;
    width: 100%;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.7rem;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🏠 Sylhet Real Estate Estimator</div>
    <div class="hero-subtitle">AI Property Price Prediction</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INPUT UI
# -----------------------------
left, right = st.columns(2)

with left:
    st.markdown("### Property Details")

    size = st.number_input("Size (sqft)", 300, 15000, 1500)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 8, 2)
    floor = st.number_input("Floor Number", 0, 30, 5)

with right:
    st.markdown("### Features")

    balcony = st.number_input("Balcony", 0, 5, 1)
    parking = st.number_input("Parking", 0, 5, 1)
    lift = st.number_input("Lift (0 or 1)", 0, 1, 1)
    cctv = st.number_input("CCTV (0 or 1)", 0, 1, 0)
    generator = st.number_input("Generator (0 or 1)", 0, 1, 0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Price"):

    # IMPORTANT: MUST MATCH TRAINING FEATURES
    input_data = pd.DataFrame([{
        "Size (sqft)": size,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floor Number": floor,
        "Balcony": balcony,
        "Parking": parking,
        "Lift": lift,
        "CCTV": cctv,
        "Generator": generator
    }])

    prediction = model.predict(input_data)[0]

    # display result
    st.markdown("""
    <div class="result-card">
        <h3>Estimated Price</h3>
        <div class="result-price">
    """ + f"৳ {int(prediction):,}" + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("Prediction completed successfully 🚀")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Sylhet House Price Predictor | ML Project")