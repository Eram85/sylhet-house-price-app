import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Sylhet House Price",
    page_icon="🏠",
    layout="wide"
)

# ─────────────────────────────
# APPLE-THEMED CSS
# ─────────────────────────────
st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: #000000;
    color: #f5f5f7;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 980px !important;
    margin: 0 auto !important;
}

.apple-nav {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(0,0,0,0.72);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 0 2rem;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.apple-nav-logo { font-size: 1rem; font-weight: 600; color: #f5f5f7; letter-spacing: -0.01em; }
.apple-nav-tag  { font-size: 0.72rem; color: #6e6e73; letter-spacing: 0.02em; }

.hero { text-align: center; padding: 5rem 1.5rem 3rem; }
.hero-eyebrow {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #0a84ff; margin-bottom: 1rem; display: block;
}
.hero-title {
    font-size: clamp(2.6rem, 6vw, 4.5rem); font-weight: 700;
    line-height: 1.05; letter-spacing: -0.03em; color: #f5f5f7; margin-bottom: 1.2rem;
}
.hero-title .accent {
    background: linear-gradient(135deg, #0a84ff 0%, #30d158 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 1.05rem; font-weight: 400; color: #86868b;
    max-width: 480px; margin: 0 auto 3rem auto; line-height: 1.6; text-align: center;
}
.stat-strip { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 3.5rem; }
.stat-chip {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 980px; padding: 0.3rem 0.9rem; font-size: 0.75rem;
    font-weight: 500; color: #98989d;
}
.stat-chip b { color: #f5f5f7; font-weight: 600; }

.card {
    background: #1c1c1e; border-radius: 18px; padding: 2rem 1.8rem;
    border: 1px solid rgba(255,255,255,0.07); margin-bottom: 1rem; height: 100%;
}
.card-title {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #636366; margin-bottom: 1.5rem;
    padding-bottom: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.06);
}

div[data-testid="stNumberInput"] label {
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #98989d !important; margin-bottom: 0.25rem !important;
}
div[data-testid="stNumberInput"] input {
    background: #2c2c2e !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #f5f5f7 !important;
    font-size: 1rem !important; font-weight: 500 !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #0a84ff !important; box-shadow: 0 0 0 3px rgba(10,132,255,0.18) !important;
}
div[data-testid="stNumberInput"] button {
    background: #3a3a3c !important; border: none !important;
    color: #f5f5f7 !important; border-radius: 8px !important;
}

div[data-testid="stRadio"] label { font-size: 0.8rem !important; font-weight: 500 !important; color: #98989d !important; }
.stRadio > div { gap: 0.35rem !important; flex-direction: row !important; flex-wrap: wrap !important; }
.stRadio label {
    background: #2c2c2e !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; padding: 0.3rem 0.75rem !important;
    font-size: 0.8rem !important; color: #98989d !important; cursor: pointer !important;
}
.stRadio label:has(input:checked) {
    background: rgba(10,132,255,0.12) !important;
    border-color: rgba(10,132,255,0.45) !important; color: #0a84ff !important;
}

div[data-testid="stSelectbox"] label {
    font-size: 0.82rem !important; font-weight: 500 !important; color: #98989d !important;
}

.stButton > button {
    background: #0a84ff !important; color: #ffffff !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    border: none !important; border-radius: 980px !important;
    padding: 0.75rem 2.5rem !important; width: 100% !important;
    transition: all 0.2s cubic-bezier(0.34,1.56,0.64,1) !important;
    box-shadow: 0 4px 20px rgba(10,132,255,0.4) !important;
}
.stButton > button:hover {
    background: #409cff !important; box-shadow: 0 6px 28px rgba(10,132,255,0.55) !important;
    transform: scale(1.018) !important;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(28px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-card {
    background: #1c1c1e; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px; padding: 3rem 2.5rem; text-align: center;
    margin-top: 2rem; position: relative; overflow: hidden;
    animation: slideUp 0.5s cubic-bezier(0.34,1.2,0.64,1) both;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(10,132,255,0.6) 50%, transparent 100%);
}
.result-eyebrow { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #48484a; margin-bottom: 1.2rem; }
.result-price-row { display: flex; align-items: flex-start; justify-content: center; gap: 0.25rem; line-height: 1; margin-bottom: 0.5rem; }
.result-sym  { font-size: 1.8rem; font-weight: 300; color: #636366; padding-top: 0.5rem; }
.result-price { font-size: clamp(3rem, 9vw, 5.2rem); font-weight: 700; letter-spacing: -0.045em; color: #f5f5f7; }
.result-lakh  { font-size: 0.88rem; font-weight: 500; color: #0a84ff; margin-bottom: 1rem; }
.result-range { font-size: 0.80rem; font-weight: 400; color: #636366; margin-bottom: 2rem; }
.result-sep   { width: 48px; height: 1px; background: rgba(255,255,255,0.08); margin: 1.6rem auto; }
.result-specs { display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; }
.spec-item    { text-align: center; }
.spec-val     { font-size: 1rem; font-weight: 600; color: #f5f5f7; }
.spec-key     { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #3a3a3c; margin-top: 0.2rem; }

div[data-testid="stAlert"] {
    background: rgba(48,209,88,0.07) !important; border: 1px solid rgba(48,209,88,0.2) !important;
    color: #30d158 !important; border-radius: 12px !important; font-size: 0.84rem !important;
}

.apple-footer {
    text-align: center; padding: 3rem 1.5rem 2.5rem;
    border-top: 1px solid rgba(255,255,255,0.06); margin-top: 5rem;
}
.apple-footer p { font-size: 0.72rem; color: #3a3a3c; line-height: 2; }

@media (max-width: 768px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .hero { padding: 3rem 1rem 2rem; }
    div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
    .card { padding: 1.4rem 1.1rem; }
    .result-card { padding: 2rem 1.4rem; }
}

html { scroll-behavior: smooth; }
#result-anchor { scroll-margin-top: 5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# LOAD MODEL & ARTEFACTS
# ─────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("gbm_bundle.pkl")
    model   = bundle["model"]
    columns = list(bundle["columns"])          # force plain Python list
    rmse    = float(bundle.get("rmse", 2_797_745))  # force plain float
    return model, columns, rmse


_loaded       = load_model()
model         = _loaded[0]
model_columns = _loaded[1]
gbm_rmse      = _loaded[2]

LOCATIONS = [
    "Akhalia, Sylhet", "Ambarkhana, Sylhet", "Bagbari, Sylhet",
    "Chowhatta, Sylhet", "Dargah Gate, Sylhet", "Kazirbazar, Sylhet",
    "Kumarpara, Sylhet", "Majortila, Sylhet", "Mendibagh, Sylhet",
    "Mira Housing Estate, Sylhet", "Mirabazar, Sylhet", "Moulikergaon, Sylhet",
    "Nayasarak, Sylhet", "Pathantula, Sylhet", "Shahjalal Uposhahar, Sylhet",
    "Shahporan, Sylhet", "Shibgonj, Sylhet", "Subidbazar, Sylhet",
    "Taltola, Sylhet", "Tilagor, Sylhet", "Zindabazar, Sylhet",
]

PROPERTY_TYPES = ["Apartment", "House", "Duplex"]

# ─────────────────────────────
# NAV BAR
# ─────────────────────────────
st.markdown("""
<div class="apple-nav">
    <div class="apple-nav-logo">🏠 Sylhet Real Estate</div>
    <div class="apple-nav-tag">AI Price Estimator</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# HERO
# ─────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">Powered by Machine Learning</span>
    <h1 class="hero-title">Know your home's<br><span class="accent">true value.</span></h1>
    <p class="hero-sub">
        Enter your property details for an instant AI-powered price estimate
        calibrated to Sylhet's real estate market.
    </p>
    <div class="stat-strip">
        <div class="stat-chip"><b>Gradient Boosting</b> Model</div>
        <div class="stat-chip"><b>12</b> Features</div>
        <div class="stat-chip"><b>BDT</b> Currency</div>
        <div class="stat-chip"><b>Sylhet</b> Market</div>
        <div class="stat-chip">Confidence <b>Range</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# INPUT CARDS
# ─────────────────────────────
left, right = st.columns(2, gap="medium")

with left:
    st.markdown('<div class="card"><div class="card-title">Property Details</div>', unsafe_allow_html=True)
    property_type   = st.selectbox("Property Type", PROPERTY_TYPES, index=0)
    location        = st.selectbox("Location", LOCATIONS, index=0)
    size            = st.number_input("Size (sqft)",     300, 15000, 1500, step=50)
    bedrooms        = st.number_input("Bedrooms",          1,    10,    3)
    bathrooms       = st.number_input("Bathrooms",         1,     8,    2)
    floor           = st.number_input("Floor Number (0 = ground / whole house)", 0, 30, 5)
    is_ground_whole = 1 if floor == 0 else 0
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">Amenities</div>', unsafe_allow_html=True)
    balcony = st.number_input("Balconies",       0, 5, 1)
    parking = st.number_input("Parking Spaces",  0, 5, 1)
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lift_opt  = st.radio("Lift",      ["No", "Yes"], index=1)
        lift      = 1 if lift_opt == "Yes" else 0
    with col_b:
        cctv_opt  = st.radio("CCTV",      ["No", "Yes"], index=0)
        cctv      = 1 if cctv_opt == "Yes" else 0
    with col_c:
        gen_opt   = st.radio("Generator", ["No", "Yes"], index=0)
        generator = 1 if gen_opt == "Yes" else 0
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("Get Price Estimate  →")

# ─────────────────────────────
# RESULT
# ─────────────────────────────
if predict:
    raw = pd.DataFrame([{
        "Size (sqft)":        size,
        "Bedrooms":           bedrooms,
        "Bathrooms":          bathrooms,
        "Floor Number":       floor,
        "Is_Ground_Or_Whole": is_ground_whole,
        "Balcony":            balcony,
        "Parking":            parking,
        "Lift":               lift,
        "CCTV":               cctv,
        "Generator":          generator,
        "Location":           location,
        "Property Type":      property_type,
    }])

    # One-hot encode then align to training columns
    input_data = pd.get_dummies(raw, columns=["Location", "Property Type"])
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Model was trained on log(price), so reverse with exp()
    log_pred   = model.predict(input_data)[0]
    prediction = np.exp(log_pred)

    # Confidence interval: point estimate ± 1 RMSE
    price_low  = max(0, prediction - gbm_rmse)
    price_high = prediction + gbm_rmse

    def fmt_readable(p):
        if p >= 1_00_00_000:
            return f"{p/1_00_00_000:.2f} Crore BDT"
        elif p >= 1_00_000:
            return f"{p/1_00_000:.1f} Lakh BDT"
        return f"{int(p):,} BDT"

    price_fmt    = f"{int(prediction):,}"
    readable     = f"≈ {fmt_readable(prediction)}"
    range_str    = f"Estimated range: ৳ {int(price_low):,} – ৳ {int(price_high):,}"

    lift_icon = "✓" if lift else "–"
    cctv_icon = "✓" if cctv else "–"
    gen_icon  = "✓" if generator else "–"

    st.markdown(f"""
    <div id="result-anchor"></div>
    <div class="result-card">
        <div class="result-eyebrow">Estimated Market Value</div>
        <div class="result-price-row">
            <span class="result-sym">৳</span>
            <span class="result-price">{price_fmt}</span>
        </div>
        <div class="result-lakh">{readable}</div>
        <div class="result-range">{range_str}</div>
        <div class="result-sep"></div>
        <div class="result-specs">
            <div class="spec-item">
                <div class="spec-val">{property_type}</div>
                <div class="spec-key">Type</div>
            </div>
            <div class="spec-item">
                <div class="spec-val">{size:,}</div>
                <div class="spec-key">sqft</div>
            </div>
            <div class="spec-item">
                <div class="spec-val">{bedrooms}BR · {bathrooms}BA</div>
                <div class="spec-key">Layout</div>
            </div>
            <div class="spec-item">
                <div class="spec-val">Floor {floor}</div>
                <div class="spec-key">Level</div>
            </div>
            <div class="spec-item">
                <div class="spec-val">{lift_icon} {cctv_icon} {gen_icon}</div>
                <div class="spec-key">Lift · CCTV · Gen</div>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        function doScroll() {{
            var el = document.getElementById('result-anchor');
            if (el) {{ el.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); return; }}
            try {{
                var containers = [
                    window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                    window.parent.document.querySelector('section.main'),
                    window.parent.document.body
                ];
                for (var i = 0; i < containers.length; i++) {{
                    if (containers[i]) {{
                        containers[i].scrollTo({{ top: containers[i].scrollHeight, behavior: 'smooth' }});
                        break;
                    }}
                }}
            }} catch(e) {{}}
        }}
        setTimeout(doScroll, 150);
    }})();
    </script>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.success("✓ Estimate ready  ·  Gradient Boosting Model  ·  For reference only")

# ─────────────────────────────
# MODEL PERFORMANCE METRICS
# ─────────────────────────────
import json, os

st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; margin-bottom:1.5rem;'>
    <span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
                text-transform:uppercase; color:#636366;'>
        Model Performance — Test Set Results
    </span>
</div>
""", unsafe_allow_html=True)

metrics_path = "model_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Sort by R² descending
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]["R2"], reverse=True)
    best_name = sorted_models[0][0]

    rows_html = ""
    for name, m in sorted_models:
        highlight = "background:rgba(10,132,255,0.08); border:1px solid rgba(10,132,255,0.25);" if name == best_name else "background:#1c1c1e; border:1px solid rgba(255,255,255,0.07);"
        badge = " <span style='font-size:0.6rem;background:#0a84ff;color:#fff;border-radius:4px;padding:1px 6px;margin-left:6px;vertical-align:middle;'>BEST</span>" if name == best_name else ""
        rows_html += f"""
        <tr style='{highlight} border-radius:10px;'>
            <td style='padding:0.6rem 1rem; font-size:0.82rem; color:#f5f5f7; font-weight:{"600" if name==best_name else "400"};'>{name}{badge}</td>
            <td style='padding:0.6rem 1rem; text-align:center; font-size:0.82rem; color:{"#30d158" if name==best_name else "#98989d"}; font-weight:600;'>{m["R2"]:.4f}</td>
            <td style='padding:0.6rem 1rem; text-align:center; font-size:0.82rem; color:#98989d;'>৳ {m["MAE"]:,}</td>
            <td style='padding:0.6rem 1rem; text-align:center; font-size:0.82rem; color:#98989d;'>৳ {m["RMSE"]:,}</td>
        </tr>"""

    st.markdown(f"""
    <div style='background:#1c1c1e; border-radius:18px; padding:1.5rem;
                border:1px solid rgba(255,255,255,0.07); overflow-x:auto;'>
        <table style='width:100%; border-collapse:separate; border-spacing:0 4px;'>
            <thead>
                <tr>
                    <th style='padding:0.4rem 1rem; text-align:left; font-size:0.65rem;
                            font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
                                color:#48484a;'>Model</th>
                    <th style='padding:0.4rem 1rem; text-align:center; font-size:0.65rem;
                            font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
                                color:#48484a;'>R²</th>
                    <th style='padding:0.4rem 1rem; text-align:center; font-size:0.65rem;
                            font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
                                color:#48484a;'>MAE (BDT)</th>
                    <th style='padding:0.4rem 1rem; text-align:center; font-size:0.65rem;
                            font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
                                color:#48484a;'>RMSE (BDT)</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    <p style='text-align:center; font-size:0.68rem; color:#3a3a3c; margin-top:0.75rem;'>
        Evaluated on held-out test set (20% of data) · No data leakage · 60/20/20 split
    </p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align:center; color:#48484a; font-size:0.8rem; padding:1rem;'>
        Run <code>model.py</code> to generate model_metrics.json
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("""
<div class="apple-footer">
    <p>Copyright © 2026 Sylhet Real Estate Estimator. All rights reserved.</p>
    <p>For informational purposes only · Not financial advice · Sylhet, Bangladesh</p>
</div>
""", unsafe_allow_html=True)
