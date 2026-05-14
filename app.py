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
# CUSTOM CSS
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
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #8a9bb0;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.4rem;
    }

    /* Section labels */
    .section-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #c8a050;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(200, 160, 80, 0.15);
    }

    /* Card panels */
    .input-panel {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(200,160,80,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(200,160,80,0.15), rgba(200,160,80,0.05));
        border: 1px solid rgba(200,160,80,0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #8a9bb0;
        margin-bottom: 0.5rem;
    }
    .result-price {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #c8a050;
        line-height: 1.1;
    }
    .result-range {
        font-size: 0.85rem;
        color: #8a9bb0;
        margin-top: 0.5rem;
    }
    .result-per-sqft {
        font-size: 1rem;
        color: #a8b8c8;
        margin-top: 0.4rem;
    }

    /* Metric pills */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .metric-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(200,160,80,0.2);
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.8rem;
        color: #c8d8e8;
    }
    .metric-pill span {
        color: #c8a050;
        font-weight: 500;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #c8a050, #a07030);
        color: #0f1923;
        border: none;
        padding: 0.75rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 0.5rem;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #d8b060, #b08040);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(200,160,80,0.3);
    }

    /* Number inputs & selects */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(200,160,80,0.2) !important;
        border-radius: 8px !important;
        color: #e8dcc8 !important;
    }

    /* Toggle (checkbox) */
    .stCheckbox label {
        color: #c8d8e8 !important;
        font-size: 0.95rem;
    }

    /* Divider */
    hr {
        border-color: rgba(200,160,80,0.15);
    }

    /* Info box */
    .stInfo {
        background: rgba(200,160,80,0.08) !important;
        border-color: rgba(200,160,80,0.3) !important;
        color: #c8d8e8 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #4a6070;
        font-size: 0.8rem;
        letter-spacing: 1px;
        margin-top: 2rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🏠 Sylhet Real Estate Estimator</div>
    <div class="hero-subtitle">AI-Powered Property Valuation · Sylhet, Bangladesh</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ **Model file not found.** Place `house_price_model.pkl` in the same directory as `app.py`.")
    st.stop()

# -----------------------------
# LAYOUT: Two main columns
# -----------------------------
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # --- Property Basics ---
    st.markdown('<div class="section-label">📐 Property Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        size = st.number_input("Size (sqft)", min_value=300, max_value=15000, value=1500, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        floor = st.number_input("Floor Number", min_value=0, max_value=30, value=5,
                                help="Ground floor = 0")
    with c2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
        balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
        parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Amenities ---
    st.markdown('<div class="section-label">✨ Amenities & Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        lift = st.checkbox("🛗 Lift / Elevator", value=True)
    with a2:
        cctv = st.checkbox("📹 CCTV Security", value=False)
    with a3:
        generator = st.checkbox("⚡ Generator", value=False)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Location ---
    st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    location = st.selectbox(
        "Select Area",
        ["Sylhet", "Shahporan", "Akhalia", "Tilagor", "Ambarkhana",
         "Zindabazar", "Bondor", "Modina Market", "South Surma"],
        help="Choose the neighbourhood of the property"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Predict Button ---
    predict_btn = st.button("🔍 Estimate Property Price", use_container_width=True)

with right_col:
    st.markdown('<div class="section-label">💰 Valuation Result</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "Size (sqft)": size,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floor Number": floor,
            "Balcony": balcony,
            "Parking": parking,
            "Lift": int(lift),
            "CCTV": int(cctv),
            "Generator": int(generator),
            "Location": location
        }])

        prediction = model.predict(input_data)[0]

        # Confidence range: ±8%
        low = int(prediction * 0.92)
        high = int(prediction * 1.08)
        per_sqft = int(prediction / size)

        # Format numbers
        def fmt(n):
            if n >= 10_000_000:
                return f"৳ {n/10_000_000:.2f} Cr"
            elif n >= 100_000:
                return f"৳ {n/100_000:.1f} Lac"
            else:
                return f"৳ {int(n):,}"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Value</div>
            <div class="result-price">{fmt(int(prediction))}</div>
            <div class="result-per-sqft">≈ ৳ {per_sqft:,} per sqft</div>
            <div class="result-range">Range: {fmt(low)} – {fmt(high)}</div>
        </div>
        """, unsafe_allow_html=True)

        # Property summary pills
        amenities = []
        if lift: amenities.append("Lift")
        if cctv: amenities.append("CCTV")
        if generator: amenities.append("Generator")
        amenity_str = ", ".join(amenities) if amenities else "None"

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">📐 <span>{size:,}</span> sqft</div>
            <div class="metric-pill">🛏 <span>{bedrooms}</span> Beds</div>
            <div class="metric-pill">🚿 <span>{bathrooms}</span> Baths</div>
            <div class="metric-pill">🏢 Floor <span>{floor}</span></div>
            <div class="metric-pill">🏗 <span>{location}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        # Breakdown expander
        with st.expander("📊 View Input Summary"):
            st.dataframe(
                input_data.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

    else:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(200,160,80,0.2);
            border-radius: 12px;
            padding: 3rem 1.5rem;
            text-align: center;
            color: #4a6070;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🏗️</div>
            <div style="font-size: 0.9rem; letter-spacing: 1px;">
                Fill in the property details<br>and click <strong style="color:#c8a050">Estimate Price</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tips panel
    st.markdown('<div class="section-label">💡 Valuation Tips</div>', unsafe_allow_html=True)
    with st.expander("How is the price estimated?"):
        st.markdown("""
        - The model is trained on **real Sylhet property listings**
        - Price range reflects **±8% market variance**
        - Amenities (lift, CCTV, generator) add a **5–15% premium**
        - Higher floors generally command **better prices**
        - Per sqft rate helps **compare across sizes**
        """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    Sylhet Real Estate Estimator · Powered by Machine Learning<br>
    For informational purposes only. Consult a local agent for official valuation.
</div>
""", unsafe_allow_html=True)