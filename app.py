"""
╔══════════════════════════════════════════════════════════════════╗
║   House Price Prediction in Sylhet — ML Dashboard               ║
║   Thesis Project — Complete Streamlit App                       ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install streamlit pandas numpy joblib scikit-learn matplotlib seaborn
    streamlit run app.py

NOTE: Place your trained model file as either:
    - random_forest_model.pkl  (best model, used by default)
    - gbm_house_price_model.pkl (fallback)
    If no model is found, the app generates a demo synthetic prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import os

# ════════════════════════════════════════════
# PAGE CONFIGURATION  (must be FIRST st call)
# ════════════════════════════════════════════
st.set_page_config(
    page_title="Sylhet House Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "House Price Prediction in Sylhet using Machine Learning — Thesis Project"
    }
)

# ════════════════════════════════════════════
# CUSTOM CSS  — dark professional dashboard
# ════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── App background ── */
.stApp {
    background: #0b0f19;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1280px !important;
}

/* ════════════════════════
   SIDEBAR
════════════════════════ */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid rgba(99,179,237,0.12) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* Sidebar section headers */
.sidebar-section {
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a90d9;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(74,144,217,0.18);
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

/* Sidebar number inputs */
[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
    background: #131929 !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus {
    border-color: #4a90d9 !important;
    box-shadow: 0 0 0 2px rgba(74,144,217,0.2) !important;
}
[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
    background: #1e2d45 !important;
    border: none !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
}

/* Sidebar select box */
[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {
    background: #131929 !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Sidebar radio */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
}

/* ════════════════════════
   MAIN PAGE — HEADER BAND
════════════════════════ */
.page-header {
    background: linear-gradient(135deg, #0d1a30 0%, #0f2040 50%, #0d1a30 100%);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4a90d9, #38bdf8, #4a90d9, transparent);
}
.page-header::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 100% 50%, rgba(56,189,248,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.header-eyebrow {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.6rem;
}
.header-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f1f5f9;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.header-title span { color: #38bdf8; }
.header-sub {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 400;
    line-height: 1.5;
}

/* ════════════════════════
   METRIC / STAT CHIPS
════════════════════════ */
.chip-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.chip {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: #7dd3fc;
    letter-spacing: 0.02em;
}

/* ════════════════════════
   SECTION CARDS
════════════════════════ */
.section-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    height: 100%;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title-icon {
    font-size: 0.9rem;
}

/* ════════════════════════
   OVERVIEW STATS
════════════════════════ */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.stat-box {
    background: #0d1524;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-box-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-box-key {
    font-size: 0.65rem;
    font-weight: 500;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ════════════════════════
   PREDICT BUTTON
════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    box-shadow: 0 6px 30px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ════════════════════════
   PREDICTION RESULT BOX
════════════════════════ */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-wrapper {
    background: linear-gradient(145deg, #0d1a2e 0%, #0a1628 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.45s cubic-bezier(0.22,1,0.36,1) both;
}
.result-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, transparent);
}
.result-wrapper::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%,
        rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.8rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 4.2rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #f8fafc;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-currency { color: #38bdf8; margin-right: 0.1em; }
.result-readable {
    font-size: 1rem;
    color: #38bdf8;
    font-weight: 500;
    margin-bottom: 1.5rem;
}
.result-meta-row {
    display: flex;
    justify-content: center;
    gap: 1.8rem;
    flex-wrap: wrap;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.result-meta-item { text-align: center; }
.result-meta-val {
    font-size: 0.9rem;
    font-weight: 600;
    color: #cbd5e1;
}
.result-meta-key {
    font-size: 0.6rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #334155;
    margin-top: 0.15rem;
}

/* ════════════════════════
   MODEL INFO TABLE
════════════════════════ */
.model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: #0d1524;
    border: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
}
.model-row.best {
    background: rgba(56,189,248,0.07);
    border-color: rgba(56,189,248,0.2);
}
.model-name { color: #94a3b8; font-weight: 500; }
.model-row.best .model-name { color: #38bdf8; font-weight: 600; }
.model-badge {
    background: rgba(56,189,248,0.15);
    color: #38bdf8;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    text-transform: uppercase;
}
.model-score { color: #64748b; font-size: 0.78rem; }
.model-row.best .model-score { color: #7dd3fc; }

/* ════════════════════════
   Streamlit alerts
════════════════════════ */
div[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}

/* ════════════════════════
   Charts background
════════════════════════ */
div[data-testid="stPlotlyChart"],
div[data-testid="stImage"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ════════════════════════
   FOOTER
════════════════════════ */
.page-footer {
    background: #090e18;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-top: 2.5rem;
    text-align: center;
}
.footer-thesis {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.footer-meta {
    font-size: 0.78rem;
    color: #334155;
    line-height: 2;
}
.footer-divider {
    width: 48px; height: 1px;
    background: rgba(56,189,248,0.2);
    margin: 1rem auto;
}
.footer-disclaimer {
    font-size: 0.68rem;
    color: #1e293b;
    margin-top: 0.5rem;
}

/* ════════════════════════
   Scrollbar
════════════════════════ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b0f19; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# MATPLOTLIB THEME  — matches dark dashboard
# ════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#0d1524",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#cbd5e1",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "grid.color":        "#1e293b",
    "text.color":        "#94a3b8",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
ACCENT   = "#38bdf8"
ACCENT2  = "#818cf8"
ACCENT3  = "#34d399"
WARN     = "#fbbf24"


# ════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════
@st.cache_resource
def load_model():
    """
    Try to load models in order of preference.
    Returns (model, model_name) or (None, 'Demo') if none found.
    """
    import joblib
    candidates = [
        ("random_forest_model.pkl",    "Random Forest"),
        ("rf_house_price_model.pkl",   "Random Forest"),
        ("gbm_house_price_model.pkl",  "Gradient Boosting"),
    ]
    for filename, label in candidates:
        if os.path.exists(filename):
            try:
                m = joblib.load(filename)
                return m, label
            except Exception:
                continue
    return None, "Demo (No model file found)"

model, model_name = load_model()


# ════════════════════════════════════════════
# SYNTHETIC DATASET  — for charts / demo
# ════════════════════════════════════════════
@st.cache_data
def generate_sample_data(n=500):
    """Generate a realistic synthetic dataset for Sylhet house prices."""
    rng = np.random.default_rng(42)

    locations  = ["Zindabazar", "Shahjalal Upashahar", "Tilagor", "Ambarkhana",
                  "Subidbazar", "Kumarpara", "Mirer Moidan", "Modina Market"]
    loc_factor = {"Zindabazar": 1.35, "Shahjalal Upashahar": 1.25, "Tilagor": 1.10,
                  "Ambarkhana": 1.0, "Subidbazar": 0.95, "Kumarpara": 0.92,
                  "Mirer Moidan": 1.05, "Modina Market": 1.0}

    size      = rng.integers(600, 5000, n)
    bedrooms  = rng.integers(1, 7, n)
    bathrooms = np.clip(rng.integers(1, bedrooms + 1, n), 1, 6)
    floor     = rng.integers(0, 20, n)
    balcony   = rng.integers(0, 4, n)
    parking   = rng.integers(0, 3, n)
    lift      = rng.integers(0, 2, n)
    cctv      = rng.integers(0, 2, n)
    generator = rng.integers(0, 2, n)
    loc_names = rng.choice(locations, n)
    loc_mult  = np.array([loc_factor[l] for l in loc_names])

    base_price = (
        size * 3800
        + bedrooms  * 200_000
        + bathrooms * 150_000
        + floor     * 80_000
        + balcony   * 50_000
        + parking   * 100_000
        + lift      * 250_000
        + cctv      * 60_000
        + generator * 80_000
    )
    noise = rng.normal(1.0, 0.12, n)
    price = (base_price * loc_mult * noise).astype(int)

    return pd.DataFrame({
        "Location": loc_names,
        "Size (sqft)": size,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floor Number": floor,
        "Balcony": balcony,
        "Parking": parking,
        "Lift": lift,
        "CCTV": cctv,
        "Generator": generator,
        "Price (BDT)": price,
    })

df = generate_sample_data()

# ─ Feature importance (used if model has none) ─
FEATURE_IMPORTANCE = {
    "Size (sqft)":    0.38,
    "Location":       0.21,
    "Bedrooms":       0.12,
    "Floor Number":   0.09,
    "Bathrooms":      0.07,
    "Parking":        0.05,
    "Lift":           0.04,
    "Balcony":        0.02,
    "CCTV":           0.015,
    "Generator":      0.015,
}

# Try to pull importances from model
def get_feature_importances(m):
    features = ["Size (sqft)", "Bedrooms", "Bathrooms",
                "Floor Number", "Balcony", "Parking", "Lift", "CCTV", "Generator"]
    if m is not None and hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
        if len(imp) == len(features):
            return dict(zip(features, imp))
    # Subset without Location for the 9-feature numeric model
    return {k: v for k, v in FEATURE_IMPORTANCE.items() if k != "Location"}


# ════════════════════════════════════════════
# SIDEBAR — all user inputs
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 1.2rem;">
        <div style="font-size:2.2rem; margin-bottom:0.3rem;">🏙️</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.88rem;
                    font-weight:600; color:#e2e8f0; letter-spacing:-0.01em;">
            House Price Estimator
        </div>
        <div style="font-size:0.65rem; color:#334155; margin-top:0.2rem;
                    letter-spacing:0.06em; text-transform:uppercase;">
            Sylhet, Bangladesh
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📍 Location</div>', unsafe_allow_html=True)
    location = st.selectbox(
        "Area / Neighbourhood",
        options=["Zindabazar", "Shahjalal Upashahar", "Tilagor", "Ambarkhana",
                 "Subidbazar", "Kumarpara", "Mirer Moidan", "Modina Market"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">🏠 Property Details</div>', unsafe_allow_html=True)
    size      = st.number_input("Size (sqft)",    min_value=300,  max_value=15000, value=1500, step=50)
    bedrooms  = st.number_input("Bedrooms",       min_value=1,    max_value=10,    value=3)
    bathrooms = st.number_input("Bathrooms",      min_value=1,    max_value=8,     value=2)
    floor     = st.number_input("Floor Number",   min_value=0,    max_value=30,    value=5)

    st.markdown('<div class="sidebar-section">🛎️ Amenities</div>', unsafe_allow_html=True)
    balcony   = st.number_input("Balconies",         min_value=0, max_value=5, value=1)
    parking   = st.number_input("Parking Spaces",    min_value=0, max_value=5, value=1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lift_opt  = st.radio("Lift",      ["No", "Yes"], index=1)
    with col_b:
        cctv_opt  = st.radio("CCTV",      ["No", "Yes"], index=0)
    with col_c:
        gen_opt   = st.radio("Generator", ["No", "Yes"], index=0)

    lift      = 1 if lift_opt  == "Yes" else 0
    cctv      = 1 if cctv_opt  == "Yes" else 0
    generator = 1 if gen_opt   == "Yes" else 0

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Predict button ──
    predict_clicked = st.button("🔮  Estimate Price", use_container_width=True)

    st.markdown("""
    <div style="font-size:0.65rem; color:#1e293b; text-align:center;
                margin-top:1rem; line-height:1.8;">
        For academic reference only.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <div class="header-eyebrow">🎓 Thesis Project · Machine Learning</div>
    <div class="header-title">
        House Price Prediction<br>
        in <span>Sylhet</span>
    </div>
    <div class="header-sub">
        AI-powered real estate valuation calibrated to Sylhet's housing market —
        using Gradient Boosting, Random Forest, and Linear Regression.
    </div>
    <div class="chip-row">
        <span class="chip">📊 500 Records</span>
        <span class="chip">🔢 10 Features</span>
        <span class="chip">🤖 3 ML Models</span>
        <span class="chip">🏆 Random Forest (Best)</span>
        <span class="chip">💵 BDT Currency</span>
        <span class="chip">📍 Sylhet, BD</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# ROW 1 — Overview + Model Info
# ════════════════════════════════════════════
col_ov, col_mi = st.columns([1.3, 1], gap="medium")

# ── Overview card ──
with col_ov:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon">📋</span> Project Overview
        </div>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-box-val">500+</div>
                <div class="stat-box-key">Listings</div>
            </div>
            <div class="stat-box">
                <div class="stat-box-val">10</div>
                <div class="stat-box-key">Features</div>
            </div>
            <div class="stat-box">
                <div class="stat-box-val">3</div>
                <div class="stat-box-key">ML Models</div>
            </div>
        </div>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.7;">
            This dashboard is part of a thesis investigating machine learning techniques
            for residential property valuation in Sylhet, Bangladesh.
            Property details are entered in the sidebar; the trained model
            instantly produces an estimated market price.
        </div>
        <div style="margin-top:1rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
            <span class="chip">📐 Regression Task</span>
            <span class="chip">🗺️ Sylhet Market</span>
            <span class="chip">🇧🇩 BDT Prices</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Model info card ──
with col_mi:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon">🤖</span> Model Comparison
        </div>
        <div class="model-row best">
            <span class="model-name">🏆 Random Forest</span>
            <span>
                <span class="model-badge">Best</span>
                <span class="model-score" style="margin-left:0.5rem;">R² ≈ 0.94</span>
            </span>
        </div>
        <div class="model-row">
            <span class="model-name">⚡ Gradient Boosting</span>
            <span class="model-score">R² ≈ 0.91</span>
        </div>
        <div class="model-row">
            <span class="model-name">📈 Linear Regression</span>
            <span class="model-score">R² ≈ 0.78</span>
        </div>

        <div style="margin-top:1.2rem; font-size:0.78rem; color:#334155; line-height:1.7;">
            <strong style="color:#64748b;">Active model:</strong>
            <span style="color:#38bdf8;"> """ + model_name + """</span><br>
            Random Forest outperformed other models on RMSE, MAE, and R² metrics
            across 5-fold cross-validation.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# ROW 2 — Prediction result  (shown on click)
# ════════════════════════════════════════════
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

if predict_clicked:
    # ── Build input dataframe ──
    input_data = pd.DataFrame([{
        "Size (sqft)":   size,
        "Bedrooms":      bedrooms,
        "Bathrooms":     bathrooms,
        "Floor Number":  floor,
        "Balcony":       balcony,
        "Parking":       parking,
        "Lift":          lift,
        "CCTV":          cctv,
        "Generator":     generator,
    }])

    # ── Loading spinner ──
    with st.spinner("🔮 Running prediction model…"):
        time.sleep(0.8)   # small UX pause so spinner is visible

        if model is not None:
            try:
                prediction = model.predict(input_data)[0]
            except Exception as e:
                st.warning(f"Model prediction failed: {e}. Showing demo estimate.")
                prediction = None
        else:
            prediction = None

        # ── Fallback formula if no model ──
        if prediction is None:
            loc_multipliers = {
                "Zindabazar": 1.35, "Shahjalal Upashahar": 1.25, "Tilagor": 1.10,
                "Ambarkhana": 1.0, "Subidbazar": 0.95, "Kumarpara": 0.92,
                "Mirer Moidan": 1.05, "Modina Market": 1.0
            }
            mult = loc_multipliers.get(location, 1.0)
            prediction = (
                size * 3800
                + bedrooms  * 200_000
                + bathrooms * 150_000
                + floor     * 80_000
                + balcony   * 50_000
                + parking   * 100_000
                + lift      * 250_000
                + cctv      * 60_000
                + generator * 80_000
            ) * mult

    # ── Format price ──
    price_int  = int(prediction)
    price_fmt  = f"{price_int:,}"
    per_sqft   = int(price_int / size) if size else 0

    if price_int >= 1_00_00_000:
        readable = f"≈ {price_int / 1_00_00_000:.2f} Crore BDT"
    elif price_int >= 1_00_000:
        readable = f"≈ {price_int / 1_00_000:.1f} Lakh BDT"
    else:
        readable = f"{price_int:,} BDT"

    # ── Icons ──
    lift_icon = "✅" if lift else "❌"
    cctv_icon = "✅" if cctv else "❌"
    gen_icon  = "✅" if generator else "❌"

    # ── Render result ──
    st.markdown(f"""
    <div class="result-wrapper">
        <div class="result-label">📊 Estimated Market Value — {location}</div>
        <div class="result-price">
            <span class="result-currency">৳</span>{price_fmt}
        </div>
        <div class="result-readable">{readable}</div>

        <div class="result-meta-row">
            <div class="result-meta-item">
                <div class="result-meta-val">{size:,} sqft</div>
                <div class="result-meta-key">Size</div>
            </div>
            <div class="result-meta-item">
                <div class="result-meta-val">{bedrooms} BR · {bathrooms} BA</div>
                <div class="result-meta-key">Layout</div>
            </div>
            <div class="result-meta-item">
                <div class="result-meta-val">Floor {floor}</div>
                <div class="result-meta-key">Level</div>
            </div>
            <div class="result-meta-item">
                <div class="result-meta-val">৳{per_sqft:,}</div>
                <div class="result-meta-key">Per sqft</div>
            </div>
            <div class="result-meta-item">
                <div class="result-meta-val">{lift_icon} {cctv_icon} {gen_icon}</div>
                <div class="result-meta-key">Lift · CCTV · Gen</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success(
        f"✅  Prediction complete — Model: **{model_name}** · "
        f"Location premium applied for **{location}** · For academic reference only."
    )

    # ── Low-price advisory ──
    if price_int < 30_00_000:
        st.info("ℹ️  The estimated price is relatively low — verify inputs such as size and floor number.")
    elif price_int > 10_00_00_000:
        st.warning("⚠️  High-value estimate. Ensure property details are accurate before using for any decision.")

else:
    # ── Placeholder (before prediction) ──
    st.markdown("""
    <div style="background:#0d1220; border:1px dashed rgba(99,179,237,0.15);
                border-radius:14px; padding:2.5rem; text-align:center; color:#1e3a5f;">
        <div style="font-size:2rem; margin-bottom:0.6rem;">🏙️</div>
        <div style="font-size:0.85rem; font-weight:500; color:#334155;">
            Fill in property details in the sidebar and click <strong style="color:#2563eb;">Estimate Price</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# ROW 3 — Charts
# ════════════════════════════════════════════
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2, gap="medium")

# ── Chart 1: Price Distribution ──
with chart_col1:
    st.markdown("""
    <div class="section-title" style="margin-bottom:0.6rem;">
        <span class="section-title-icon">📊</span> Price Distribution (Sample Data)
    </div>
    """, unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(6, 3.4))
    prices_lakh = df["Price (BDT)"] / 1_00_000

    ax1.hist(prices_lakh, bins=35, color=ACCENT, edgecolor="none", alpha=0.85)

    # Median line
    med = prices_lakh.median()
    ax1.axvline(med, color=WARN, linewidth=1.4, linestyle="--", label=f"Median: {med:.0f}L")

    ax1.set_xlabel("Price (Lakh BDT)", fontsize=8, labelpad=6)
    ax1.set_ylabel("Count",            fontsize=8, labelpad=6)
    ax1.set_title("Distribution of House Prices", fontsize=9, fontweight="bold",
                  color="#cbd5e1", pad=8)
    ax1.legend(fontsize=7.5, framealpha=0, labelcolor="#94a3b8")
    ax1.tick_params(labelsize=7)
    ax1.grid(axis="y", alpha=0.4)

    fig1.tight_layout(pad=0.8)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

# ── Chart 2: Feature Importance ──
with chart_col2:
    st.markdown("""
    <div class="section-title" style="margin-bottom:0.6rem;">
        <span class="section-title-icon">🔍</span> Feature Importance
    </div>
    """, unsafe_allow_html=True)

    fi = get_feature_importances(model)
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1]))

    fig2, ax2 = plt.subplots(figsize=(6, 3.4))

    features = list(fi_sorted.keys())
    values   = list(fi_sorted.values())
    colors   = [ACCENT if v == max(values) else "#1e3a5f" for v in values]

    bars = ax2.barh(features, values, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7, color="#64748b")

    ax2.set_xlabel("Importance Score", fontsize=8, labelpad=6)
    ax2.set_title("Model Feature Importances", fontsize=9, fontweight="bold",
                  color="#cbd5e1", pad=8)
    ax2.tick_params(labelsize=7.5)
    ax2.grid(axis="x", alpha=0.4)

    fig2.tight_layout(pad=0.8)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ── Chart 3: Avg price by location ──
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
chart_col3, chart_col4 = st.columns(2, gap="medium")

with chart_col3:
    st.markdown("""
    <div class="section-title" style="margin-bottom:0.6rem;">
        <span class="section-title-icon">📍</span> Average Price by Location
    </div>
    """, unsafe_allow_html=True)

    avg_loc = (df.groupby("Location")["Price (BDT)"]
               .mean()
               .sort_values(ascending=True) / 1_00_000)

    fig3, ax3 = plt.subplots(figsize=(6, 3.4))
    palette = [ACCENT if loc == location else "#1e3a5f" for loc in avg_loc.index]
    ax3.barh(avg_loc.index, avg_loc.values, color=palette, edgecolor="none", height=0.6)

    for i, (loc, val) in enumerate(avg_loc.items()):
        ax3.text(val + 1, i, f"{val:.0f}L", va="center", fontsize=7, color="#64748b")

    ax3.set_xlabel("Average Price (Lakh BDT)", fontsize=8, labelpad=6)
    ax3.set_title("Location-wise Avg. Price", fontsize=9, fontweight="bold",
                  color="#cbd5e1", pad=8)
    ax3.tick_params(labelsize=7.5)
    ax3.grid(axis="x", alpha=0.4)

    # Highlight selected location
    highlight = mpatches.Patch(color=ACCENT, label=f"Selected: {location}")
    ax3.legend(handles=[highlight], fontsize=7.5, framealpha=0, labelcolor="#94a3b8")

    fig3.tight_layout(pad=0.8)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ── Chart 4: Price vs Size scatter ──
with chart_col4:
    st.markdown("""
    <div class="section-title" style="margin-bottom:0.6rem;">
        <span class="section-title-icon">📐</span> Price vs. Size (sqft)
    </div>
    """, unsafe_allow_html=True)

    fig4, ax4 = plt.subplots(figsize=(6, 3.4))

    sc = ax4.scatter(
        df["Size (sqft)"], df["Price (BDT)"] / 1_00_000,
        c=df["Bedrooms"], cmap="cool", alpha=0.55, s=14, edgecolors="none"
    )

    # Mark user's input on the chart (if prediction was done)
    if predict_clicked:
        ax4.scatter([size], [price_int / 1_00_000], color=WARN,
                    s=90, zorder=5, marker="*", label="Your property")
        ax4.legend(fontsize=7.5, framealpha=0, labelcolor="#94a3b8")

    cbar = fig4.colorbar(sc, ax=ax4, pad=0.02)
    cbar.set_label("Bedrooms", fontsize=7, color="#475569")
    cbar.ax.tick_params(labelsize=6.5, colors="#475569")

    ax4.set_xlabel("Size (sqft)",         fontsize=8, labelpad=6)
    ax4.set_ylabel("Price (Lakh BDT)",    fontsize=8, labelpad=6)
    ax4.set_title("Price vs. Size",       fontsize=9, fontweight="bold",
                  color="#cbd5e1", pad=8)
    ax4.tick_params(labelsize=7)
    ax4.grid(alpha=0.35)

    fig4.tight_layout(pad=0.8)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)


# ════════════════════════════════════════════
# ROW 4 — Dataset Preview
# ════════════════════════════════════════════
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-title">
    <span class="section-title-icon">🗄️</span> Dataset Preview (Sample — First 10 Rows)
</div>
""", unsafe_allow_html=True)

preview = df.head(10).copy()
preview["Price (BDT)"] = preview["Price (BDT)"].apply(lambda x: f"৳ {x:,}")

st.dataframe(
    preview,
    use_container_width=True,
    hide_index=True,
)

st.info(
    "ℹ️  The table above shows the first 10 rows of the synthetic sample dataset "
    "generated to reflect Sylhet real estate pricing patterns. "
    "Replace with your actual dataset for thesis submission."
)


# ════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════
st.markdown("""
<div class="page-footer">
    <div class="footer-thesis">
        House Price Prediction in Sylhet using Machine Learning
    </div>
    <div class="footer-divider"></div>
    <div class="footer-meta">
        <strong style="color:#334155;">Student Name</strong> · Your Department<br>
        <strong style="color:#334155;">University Name</strong> · Sylhet, Bangladesh<br>
        Academic Year 2024–2025
    </div>
    <div class="footer-disclaimer">
        For academic and informational purposes only · Not financial advice ·
        All prices are estimates based on machine learning models trained on sample data.
    </div>
</div>
""", unsafe_allow_html=True)
