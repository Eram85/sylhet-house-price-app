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
# CUSTOM CSS — Refined Dark Editorial
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #e2d9cc;
}

/* ── Background ── */
.stApp {
    background-color: #0d1117;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(180,120,40,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(90,60,20,0.15) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c8a050' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1100px; }

/* ════════════════════════════
   HERO SECTION
════════════════════════════ */
.hero {
    position: relative;
    padding: 3.5rem 2rem 2.5rem;
    text-align: center;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 1px;
    height: 3rem;
    background: linear-gradient(to bottom, transparent, #c8a050);
}

.hero-eyebrow {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c8a050;
    border: 1px solid rgba(200,160,80,0.35);
    padding: 0.3rem 1rem;
    border-radius: 2rem;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 300;
    line-height: 1.1;
    color: #f0e6d3;
    margin: 0 0 0.6rem;
    letter-spacing: -0.01em;
}

.hero-title span {
    color: #c8a050;
    font-weight: 700;
    font-style: italic;
}

.hero-desc {
    font-size: 0.92rem;
    color: #6b7c91;
    max-width: 420px;
    margin: 0 auto 2rem;
    line-height: 1.65;
}

.hero-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    max-width: 600px;
    margin: 0 auto 2.5rem;
}
.hero-divider::before, .hero-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(200,160,80,0.3));
}
.hero-divider::after {
    background: linear-gradient(to left, transparent, rgba(200,160,80,0.3));
}
.hero-divider-icon {
    font-size: 1rem;
    opacity: 0.5;
}

/* ════════════════════════════
   STAT BADGES
════════════════════════════ */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
    padding: 0 1rem;
}
.stat-badge {
    text-align: center;
}
.stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #c8a050;
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    margin-top: 0.2rem;
}

/* ════════════════════════════
   SECTION TITLES
════════════════════════════ */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #c8a050;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(200,160,80,0.2);
    margin-bottom: 1.4rem;
}

/* ════════════════════════════
   INPUT PANELS
════════════════════════════ */
.panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(200,160,80,0.12);
    border-radius: 16px;
    padding: 1.8rem 1.6rem;
    backdrop-filter: blur(8px);
    height: 100%;
}

/* Streamlit number inputs */
div[data-testid="stNumberInput"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: #8a99b0 !important;
    margin-bottom: 0.2rem !important;
}

div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(200,160,80,0.2) !important;
    border-radius: 8px !important;
    color: #f0e6d3 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(200,160,80,0.6) !important;
    box-shadow: 0 0 0 3px rgba(200,160,80,0.08) !important;
}

/* Stepper buttons */
div[data-testid="stNumberInput"] button {
    background: rgba(200,160,80,0.1) !important;
    border-color: rgba(200,160,80,0.2) !important;
    color: #c8a050 !important;
}

/* Toggle inputs for binary */
.toggle-label {
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #8a99b0;
    display: block;
    margin-bottom: 0.5rem;
}
.stRadio > div {
    gap: 0.5rem !important;
}
.stRadio label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(200,160,80,0.15) !important;
    border-radius: 8px !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.85rem !important;
    color: #8a99b0 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio label:has(input:checked) {
    background: rgba(200,160,80,0.15) !important;
    border-color: rgba(200,160,80,0.5) !important;
    color: #c8a050 !important;
}

/* ════════════════════════════
   PREDICT BUTTON
════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #c8a050 0%, #a07830 50%, #c8a050 100%) !important;
    background-size: 200% 100% !important;
    color: #0d1117 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(200,160,80,0.25) !important;
}
.stButton > button:hover {
    background-position: 100% 0 !important;
    box-shadow: 0 6px 32px rgba(200,160,80,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ════════════════════════════
   RESULT CARD
════════════════════════════ */
.result-outer {
    margin-top: 2rem;
    background: linear-gradient(135deg, rgba(200,160,80,0.08) 0%, rgba(160,120,48,0.04) 100%);
    border: 1px solid rgba(200,160,80,0.3);
    border-radius: 20px;
    padding: 0.15rem;
}
.result-inner {
    background: rgba(13,17,23,0.85);
    border-radius: 18px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.result-label {
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.8rem;
}
.result-currency {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    color: #8a7050;
    vertical-align: top;
    line-height: 1.8;
}
.result-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.6rem;
    font-weight: 700;
    color: #c8a050;
    line-height: 1;
    letter-spacing: -0.02em;
}
.result-unit {
    font-size: 0.8rem;
    color: #4a5568;
    margin-top: 0.6rem;
    letter-spacing: 0.1em;
}
.result-bar {
    width: 60px;
    height: 2px;
    background: linear-gradient(to right, #c8a050, transparent);
    margin: 1.2rem auto;
}
.result-note {
    font-size: 0.78rem;
    color: #4a5568;
    font-style: italic;
}

/* ════════════════════════════
   FOOTER
════════════════════════════ */
.footer {
    text-align: center;
    padding: 2rem 1rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(200,160,80,0.08);
}
.footer-text {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2d3748;
}
.footer-dot {
    margin: 0 0.6rem;
    color: #c8a050;
    opacity: 0.4;
}

/* Success message */
div[data-testid="stAlert"] {
    background: rgba(200,160,80,0.08) !important;
    border: 1px solid rgba(200,160,80,0.25) !important;
    color: #c8a050 !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("gbm_house_price_model.pkl")

model = load_model()

# -----------------------------
# HERO
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-Powered Valuation</div>
    <h1 class="hero-title">Sylhet <span>Real Estate</span><br>Price Estimator</h1>
    <p class="hero-desc">
        Enter your property details below for an instant machine-learning estimate
        based on Sylhet's current real estate market.
    </p>
    <div class="hero-divider"><span class="hero-divider-icon">◆</span></div>
</div>

<div class="stat-row">
    <div class="stat-badge">
        <div class="stat-num">GBM</div>
        <div class="stat-label">Model</div>
    </div>
    <div class="stat-badge">
        <div class="stat-num">9</div>
        <div class="stat-label">Features</div>
    </div>
    <div class="stat-badge">
        <div class="stat-num">BDT</div>
        <div class="stat-label">Currency</div>
    </div>
    <div class="stat-badge">
        <div class="stat-num">Sylhet</div>
        <div class="stat-label">Market</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INPUTS
# -----------------------------
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏗 Property Details</div>', unsafe_allow_html=True)
    size      = st.number_input("Size (sqft)",     300,  15000, 1500, step=50)
    bedrooms  = st.number_input("Bedrooms",          1,     10,    3)
    bathrooms = st.number_input("Bathrooms",         1,      8,    2)
    floor     = st.number_input("Floor Number",      0,     30,    5)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙ Amenities & Features</div>', unsafe_allow_html=True)
    balcony   = st.number_input("Balcony",  0, 5, 1)
    parking   = st.number_input("Parking",  0, 5, 1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lift_opt = st.radio("Lift", ["No", "Yes"], index=1, horizontal=False)
        lift = 1 if lift_opt == "Yes" else 0
    with col_b:
        cctv_opt = st.radio("CCTV", ["No", "Yes"], index=0, horizontal=False)
        cctv = 1 if cctv_opt == "Yes" else 0
    with col_c:
        gen_opt = st.radio("Generator", ["No", "Yes"], index=0, horizontal=False)
        generator = 1 if gen_opt == "Yes" else 0

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict = st.button("✦ Estimate Property Value")

# -----------------------------
# PREDICTION OUTPUT
# -----------------------------
if predict:
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
    price_formatted = f"{int(prediction):,}"

    # Format as crore/lakh for readability
    if prediction >= 1_00_00_000:
        crore = prediction / 1_00_00_000
        readable = f"≈ {crore:.2f} Crore BDT"
    elif prediction >= 1_00_000:
        lakh = prediction / 1_00_000
        readable = f"≈ {lakh:.1f} Lakh BDT"
    else:
        readable = ""

    st.markdown(f"""
    <div class="result-outer">
      <div class="result-inner">
        <div class="result-label">Estimated Market Value</div>
        <div>
          <span class="result-currency">৳</span>
          <span class="result-price">{price_formatted}</span>
        </div>
        {"<div class='result-unit'>" + readable + "</div>" if readable else ""}
        <div class="result-bar"></div>
        <div class="result-note">Based on {bedrooms}BR / {bathrooms}BA · {size:,} sqft · Floor {floor}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("✔ Prediction completed — powered by Gradient Boosting Machine")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    <div class="footer-text">
        Sylhet Real Estate Estimator
        <span class="footer-dot">◆</span>
        ML Price Prediction
        <span class="footer-dot">◆</span>
        For Reference Only
    </div>
</div>
""", unsafe_allow_html=True)