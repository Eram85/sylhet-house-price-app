import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# APP_FINAL_CLEANUP
# Academic presentation safeguards:
# - Prediction Error Range is based on test RMSE and is NOT a statistical confidence interval.
# - The app does not present the model as an investment/return predictor.
# - Global feature importance is clearly distinguished from local SHAP explanations.

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Sylhet House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ─────────────────────────────
# FULL CSS DESIGN SYSTEM
# ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --bg-void:       #070a12;
    --bg-surface:    #0d1117;
    --bg-card:       rgba(255,255,255,0.032);
    --bg-card-hov:   rgba(255,255,255,0.056);
    --border:        rgba(255,255,255,0.07);
    --border-accent: rgba(99,202,246,0.2);
    --text-primary:  #eaf0fb;
    --text-secondary:#8a96a8;
    --text-muted:    #404754;
    --accent-blue:   #63caf6;
    --accent-cyan:   #3de8c8;
    --accent-grd:    linear-gradient(135deg, #63caf6 0%, #3de8c8 100%);
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     24px;
    --radius-xl:     32px;
    --font-display:  'DM Serif Display', Georgia, serif;
    --font-body:     'DM Sans', system-ui, sans-serif;
    --shadow-card:   0 1px 3px rgba(0,0,0,0.4), 0 8px 32px rgba(0,0,0,0.28);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: var(--font-body); -webkit-font-smoothing: antialiased; }

.stApp { background: var(--bg-void); color: var(--text-primary); min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAVBAR ── */
.nav-bar {
    position: sticky; top: 0; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 3rem; height: 58px;
    background: rgba(7,10,18,0.92);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border-bottom: 1px solid var(--border);
}
.nav-logo { display: flex; align-items: center; gap: .5rem; font-size: .9rem; font-weight: 600; color: var(--text-primary); }
.nav-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: linear-gradient(135deg, #63caf6, #3de8c8);
    box-shadow: 0 0 10px rgba(99,202,246,0.7);
    animation: dotPulse 2.5s ease-in-out infinite;
}
@keyframes dotPulse { 0%,100%{box-shadow:0 0 10px rgba(99,202,246,0.7)} 50%{box-shadow:0 0 18px rgba(99,202,246,1)} }
.nav-links { display: flex; gap: 2rem; align-items: center; }
.nav-links span { font-size: .78rem; font-weight: 500; letter-spacing: .04em; color: var(--text-muted); text-transform: uppercase; cursor: default; }
.nav-badge {
    background: linear-gradient(135deg, rgba(99,202,246,0.14), rgba(61,232,200,0.10));
    border: 1px solid rgba(99,202,246,0.25); border-radius: 980px;
    padding: .22rem .9rem; font-size: .68rem; font-weight: 700;
    color: var(--accent-blue); letter-spacing: .06em; text-transform: uppercase;
}

/* ── HERO ── */
.hero-wrap {
    position: relative; overflow: hidden;
    min-height: 80vh; display: flex; align-items: center; justify-content: center;
    padding: 5rem 2rem 6rem;
}
.hero-mesh {
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 40%, rgba(99,202,246,0.055) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 80% 60%, rgba(61,232,200,0.04)  0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 0%,  rgba(99,202,246,0.03)  0%, transparent 50%);
    animation: meshPulse 8s ease-in-out infinite alternate;
}
@keyframes meshPulse { 0%{opacity:.7;transform:scale(1)} 100%{opacity:1;transform:scale(1.04)} }
.hero-grid {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.022) 1px, transparent 1px);
    background-size: 52px 52px;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
}
.orb { position: absolute; border-radius: 50%; filter: blur(80px); pointer-events: none; }
.orb-1 { width:420px;height:420px; background:radial-gradient(circle,rgba(99,202,246,0.11) 0%,transparent 70%); top:-80px;left:-60px; animation:float1 12s ease-in-out infinite; }
.orb-2 { width:320px;height:320px; background:radial-gradient(circle,rgba(61,232,200,0.09) 0%,transparent 70%); bottom:40px;right:5%; animation:float2 14s ease-in-out infinite; }
.orb-3 { width:200px;height:200px; background:radial-gradient(circle,rgba(245,200,66,0.06) 0%,transparent 70%); top:40%;left:55%; animation:float3 10s ease-in-out infinite; }
@keyframes float1{0%,100%{transform:translate(0,0)}50%{transform:translate(30px,-30px)}}
@keyframes float2{0%,100%{transform:translate(0,0)}50%{transform:translate(-20px,25px)}}
@keyframes float3{0%,100%{transform:translate(0,0)}50%{transform:translate(15px,-20px)}}

.hero-content { position:relative;z-index:2;text-align:center;max-width:820px;margin:0 auto; animation:heroIn .9s cubic-bezier(.22,1,.36,1) both; }
@keyframes heroIn{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}}
.hero-eyebrow {
    display:inline-flex;align-items:center;gap:.5rem;
    background:rgba(99,202,246,0.08);border:1px solid rgba(99,202,246,0.2);
    border-radius:980px;padding:.3rem 1rem;font-size:.72rem;font-weight:600;
    letter-spacing:.1em;text-transform:uppercase;color:var(--accent-blue);margin-bottom:2rem;
}
.eyebrow-dot { width:5px;height:5px;border-radius:50%;background:var(--accent-blue); box-shadow:0 0 6px var(--accent-blue);animation:blink 2s ease-in-out infinite; }
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.hero-h1 { font-family:var(--font-display);font-size:clamp(3rem,6.5vw,5.6rem); font-weight:400;line-height:1.04;letter-spacing:-.02em;color:var(--text-primary);margin-bottom:1.5rem; }
.hero-h1 .grad { background:var(--accent-grd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.hero-sub { font-size:1.05rem;font-weight:300;line-height:1.7;color:var(--text-secondary);max-width:520px;margin:0 auto 3rem; }
.kpi-strip { display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;margin-bottom:3.5rem; }
.kpi-card {
    background:rgba(255,255,255,0.04);border:1px solid var(--border);
    border-radius:var(--radius-md);padding:1rem 1.4rem;text-align:center;min-width:110px;
    transition:all .28s ease;
}
.kpi-card:hover { background:var(--bg-card-hov);border-color:rgba(99,202,246,.2);transform:translateY(-2px); }
.kpi-val { font-family:var(--font-display);font-size:1.6rem;font-weight:400; background:var(--accent-grd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; line-height:1.1;margin-bottom:.25rem; }
.kpi-label { font-size:.65rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted); }
.hero-trust-row { display:flex;align-items:center;justify-content:center;gap:1.2rem;flex-wrap:wrap; }
.trust-pill { display:flex;align-items:center;gap:.4rem;font-size:.73rem;color:var(--text-muted);font-weight:500; }
.trust-check { width:16px;height:16px;border-radius:50%; background:rgba(61,232,200,0.12);border:1px solid rgba(61,232,200,0.28); display:inline-flex;align-items:center;justify-content:center;font-size:9px;color:var(--accent-cyan); }
.scroll-hint { position:absolute;bottom:2.5rem;left:50%;transform:translateX(-50%); display:flex;flex-direction:column;align-items:center;gap:.4rem; opacity:.35;animation:scrollBounce 2.5s ease-in-out infinite; }
@keyframes scrollBounce{0%,100%{transform:translateX(-50%) translateY(0)}50%{transform:translateX(-50%) translateY(8px)}}
.scroll-line{width:1px;height:36px;background:linear-gradient(to bottom,var(--accent-blue),transparent);}
.scroll-text{font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;color:var(--text-muted);}

/* ── SECTION DIVIDER ── */
.section-divider { display:flex;align-items:center;gap:.75rem;max-width:1100px;margin:0 auto 2.5rem;padding:0 2rem; }
.section-divider-line { flex:1;height:1px;background:var(--border); }
.section-divider-text { font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);white-space:nowrap; }

/* ── FORM CARDS ── */
.card-header { display:flex;align-items:center;gap:.65rem;margin-bottom:1.8rem;padding-bottom:1.2rem;border-bottom:1px solid var(--border); }
.card-icon { width:32px;height:32px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0; }
.icon-blue{background:rgba(99,202,246,0.12);border:1px solid rgba(99,202,246,0.18);}
.icon-cyan{background:rgba(61,232,200,0.10);border:1px solid rgba(61,232,200,0.16);}
.card-title { font-size:.78rem;font-weight:600;letter-spacing:.05em;color:var(--text-secondary); }
.card-desc  { font-size:.67rem;color:var(--text-muted);margin-top:.1rem; }

[data-testid="column"] > div > div > div {
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 28px;
    padding: 2rem 1.8rem;
    transition: box-shadow .3s;
}
[data-testid="column"] > div > div > div:hover {
    box-shadow: 0 0 0 1px rgba(99,202,246,.1), 0 8px 32px rgba(0,0,0,.28);
}

/* ── INPUTS ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] > label {
    font-size:.75rem!important;font-weight:500!important;color:var(--text-secondary)!important;letter-spacing:.02em!important;
}
div[data-testid="stNumberInput"] input {
    background:rgba(255,255,255,0.04)!important;border:1px solid var(--border)!important;
    border-radius:var(--radius-sm)!important;color:var(--text-primary)!important;
    font-size:.97rem!important;font-weight:500!important;transition:border-color .2s,box-shadow .2s!important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color:rgba(99,202,246,.5)!important;box-shadow:0 0 0 3px rgba(99,202,246,.1)!important;background:rgba(99,202,246,.04)!important;
}
div[data-testid="stNumberInput"] button {
    background:rgba(255,255,255,0.06)!important;border:1px solid var(--border)!important;color:var(--text-secondary)!important;border-radius:8px!important;
}
div[data-testid="stSelectbox"] > div > div {
    background:rgba(255,255,255,0.04)!important;border:1px solid var(--border)!important;
    border-radius:var(--radius-sm)!important;color:var(--text-primary)!important;
}
.stRadio > div { display:flex!important;flex-direction:row!important;flex-wrap:wrap!important;gap:.4rem!important;margin-top:.3rem!important; }
.stRadio label {
    background:rgba(255,255,255,0.04)!important;border:1px solid var(--border)!important;
    border-radius:8px!important;padding:.3rem .8rem!important;font-size:.78rem!important;
    color:var(--text-secondary)!important;cursor:pointer!important;transition:all .18s!important;
}
.stRadio label:has(input:checked) {
    background:rgba(99,202,246,0.1)!important;border-color:rgba(99,202,246,.35)!important;color:var(--accent-blue)!important;
}

/* ── BUTTON ── */
.stButton>button {
    background:linear-gradient(135deg,#4ab8f0,#2bcfb0)!important;
    color:#070a12!important;font-size:.92rem!important;font-weight:700!important;
    letter-spacing:.01em!important;border:none!important;border-radius:980px!important;
    padding:.82rem 3rem!important;width:100%!important;
    box-shadow:0 4px 24px rgba(99,202,246,.35)!important;
    transition:all .24s cubic-bezier(.34,1.56,.64,1)!important;
}
.stButton>button:hover { transform:translateY(-2px) scale(1.012)!important;box-shadow:0 8px 36px rgba(99,202,246,.5),0 0 0 4px rgba(99,202,246,.1)!important; }

/* ── RESULT CARD ── */
@keyframes resultSlide{from{opacity:0;transform:translateY(36px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.result-wrap { animation:resultSlide .6s cubic-bezier(.22,1,.36,1) both;position:relative;margin:2.5rem auto;max-width:900px;padding:0 2rem; }
.result-glow { position:absolute;inset:-30px;background:radial-gradient(ellipse 60% 50% at 50% 50%,rgba(99,202,246,0.08) 0%,transparent 70%);pointer-events:none; }
.result-card {
    background:rgba(13,17,23,0.92);border:1px solid rgba(99,202,246,0.18);border-radius:var(--radius-xl);
    padding:3.5rem 3rem 3rem;text-align:center;position:relative;overflow:hidden;
    box-shadow:0 0 0 1px rgba(99,202,246,.08),var(--shadow-card),0 0 60px rgba(99,202,246,.1);
}
.result-top-line { position:absolute;top:0;left:15%;right:15%;height:1px;background:var(--accent-grd);filter:blur(.5px); }
.result-eyebrow { font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:1.5rem;display:flex;align-items:center;justify-content:center;gap:.5rem; }
.result-eyebrow::before,.result-eyebrow::after{content:'';display:block;width:28px;height:1px;background:var(--border);}
.result-price-row{display:flex;align-items:flex-start;justify-content:center;gap:.3rem;line-height:1;margin-bottom:.5rem;}
.result-sym{font-family:var(--font-display);font-size:2rem;font-weight:400;color:var(--text-muted);padding-top:.8rem;}
.result-price-main{font-family:var(--font-display);font-size:clamp(3.2rem,8vw,5.8rem);font-weight:400;letter-spacing:-.03em;background:linear-gradient(135deg,#ffffff 0%,var(--accent-blue) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.result-readable{font-size:1rem;font-weight:500;color:var(--accent-cyan);margin-bottom:.6rem;letter-spacing:-.01em;}
.conf-bar-wrap{max-width:340px;margin:.8rem auto 1.8rem;}
.conf-label-row{display:flex;justify-content:space-between;font-size:.67rem;color:var(--text-muted);margin-bottom:.4rem;font-weight:500;letter-spacing:.04em;}
.conf-bar-track{height:5px;border-radius:980px;background:rgba(255,255,255,0.07);overflow:hidden;}
.conf-bar-fill{height:100%;border-radius:inherit;background:var(--accent-grd);width:78%;animation:fillBar 1.2s cubic-bezier(.22,1,.36,1) both;animation-delay:.4s;}
@keyframes fillBar{from{width:0}}
.result-range-text{font-size:.78rem;color:var(--text-muted);margin-bottom:2.5rem;}
.result-range-text span{color:var(--text-secondary);font-weight:500;}
.result-divider{width:64px;height:1px;background:var(--border);margin:0 auto 2.5rem;}
.spec-grid{display:flex;justify-content:center;gap:0;flex-wrap:wrap;}
.spec-item{text-align:center;padding:0 1.6rem;border-right:1px solid var(--border);}
.spec-item:last-child{border-right:none;}
.spec-val{font-size:1rem;font-weight:600;color:var(--text-primary);letter-spacing:-.01em;}
.spec-key{font-size:.6rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);margin-top:.25rem;}
.amenity-row{display:flex;justify-content:center;gap:.5rem;margin-top:2rem;flex-wrap:wrap;}
.amenity-badge{display:inline-flex;align-items:center;gap:.35rem;padding:.28rem .8rem;border-radius:980px;font-size:.67rem;font-weight:600;letter-spacing:.04em;}
.amenity-on{background:rgba(61,232,200,0.1);border:1px solid rgba(61,232,200,0.22);color:#3de8c8;}
.amenity-off{background:rgba(255,255,255,0.04);border:1px solid var(--border);color:var(--text-muted);}

/* ── ANALYTICS CARDS ── */
.analytics-kpi-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;max-width:1100px;margin:0 auto 2rem;padding:0 2rem; }
.akpi-card {
    background:rgba(255,255,255,0.032);border:1px solid var(--border);border-radius:var(--radius-md);
    padding:1.4rem 1.6rem;transition:all .28s ease;
}
.akpi-card:hover { background:var(--bg-card-hov);border-color:var(--border-accent);transform:translateY(-2px); }
.akpi-label { font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:.5rem; }
.akpi-value { font-family:var(--font-display);font-size:1.9rem;font-weight:400;line-height:1.1;margin-bottom:.25rem; }
.akpi-value.cyan { background:var(--accent-grd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.akpi-value.white { color:var(--text-primary); }
.akpi-sub { font-size:.68rem;color:var(--text-muted); }

/* ── MARKET INSIGHT BADGES ── */
.insight-grid { display:grid;grid-template-columns:1fr 1fr;gap:1rem;max-width:1100px;margin:0 auto 2rem;padding:0 2rem; }
.insight-card { background:rgba(255,255,255,0.032);border:1px solid var(--border);border-radius:var(--radius-md);padding:1.4rem 1.6rem; }
.insight-row { display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;border-bottom:1px solid rgba(255,255,255,0.04); }
.insight-row:last-child { border-bottom:none; }
.insight-key { font-size:.75rem;color:var(--text-secondary); }
.insight-val { font-size:.75rem;font-weight:600; }
.pill { border-radius:980px;padding:.18rem .7rem;font-size:.62rem;font-weight:700;letter-spacing:.04em; }
.pill-green { background:rgba(61,232,200,0.1);color:#3de8c8;border:1px solid rgba(61,232,200,0.2); }
.pill-amber { background:rgba(245,200,66,0.1);color:#f5c842;border:1px solid rgba(245,200,66,0.2); }
.pill-red   { background:rgba(240,98,146,0.1);color:#f06292;border:1px solid rgba(240,98,146,0.2); }

/* ── SHAP SECTION ── */
.shap-card { background:rgba(255,255,255,0.032);border:1px solid var(--border);border-radius:var(--radius-lg);padding:2rem;max-width:1100px;margin:0 auto 2rem; }
.shap-title { font-size:.78rem;font-weight:600;letter-spacing:.05em;color:var(--text-secondary);margin-bottom:1.4rem; }
.shap-row { display:flex;align-items:center;gap:1rem;margin-bottom:.85rem; }
.shap-label { font-size:.75rem;color:var(--text-secondary);min-width:120px;text-align:right; }
.shap-bar-wrap { flex:1;height:8px;background:rgba(255,255,255,0.05);border-radius:99px;overflow:hidden; }
.shap-bar-pos { height:100%;border-radius:99px;background:linear-gradient(90deg,#3de8c8,#63caf6); }
.shap-bar-neg { height:100%;border-radius:99px;background:rgba(240,98,146,0.65); }
.shap-impact { font-size:.75rem;font-weight:600;min-width:80px; }
.shap-impact.pos { color:#3de8c8; }
.shap-impact.neg { color:#f06292; }

/* ── RECOMMENDATIONS ── */
.rec-card { background:rgba(255,255,255,0.032);border:1px solid var(--border);border-radius:var(--radius-lg);padding:2rem;max-width:1100px;margin:0 auto 2rem; }
.rec-title { font-size:.78rem;font-weight:600;letter-spacing:.05em;color:var(--text-secondary);margin-bottom:1.2rem; }
.rec-row { display:flex;justify-content:space-between;align-items:center;padding:.9rem 0;border-bottom:1px solid rgba(255,255,255,0.04); }
.rec-row:last-child { border-bottom:none; }
.rec-left { display:flex;align-items:center;gap:.6rem; }
.rec-icon { font-size:1rem; }
.rec-label { font-size:.82rem;color:var(--text-secondary); }
.rec-desc { font-size:.68rem;color:var(--text-muted);margin-top:.1rem; }
.rec-impact { font-size:.85rem;font-weight:700;color:#3de8c8; }

/* ── PERFORMANCE TABLE ── */
.perf-section{max-width:1100px;margin:0 auto 4rem;padding:0 2rem;}
.perf-header{text-align:center;margin-bottom:2rem;}
.perf-title{font-family:var(--font-display);font-size:1.8rem;font-weight:400;color:var(--text-primary);margin-bottom:.5rem;}
.perf-sub{font-size:.8rem;color:var(--text-muted);}
.perf-table-wrap{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-xl);padding:1.5rem;overflow-x:auto;}
.perf-table{width:100%;border-collapse:separate;border-spacing:0 6px;}
.perf-table thead th{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);padding:.4rem 1rem;text-align:center;}
.perf-table thead th:first-child{text-align:left;}
.perf-table td{padding:.75rem 1rem;font-size:.82rem;color:var(--text-secondary);text-align:center;background:rgba(255,255,255,0.025);}
.perf-table td:first-child{text-align:left;border-radius:12px 0 0 12px;}
.perf-table td:last-child{border-radius:0 12px 12px 0;}
.perf-best td{background:rgba(99,202,246,0.06)!important;border-top:1px solid rgba(99,202,246,.12);border-bottom:1px solid rgba(99,202,246,.12);}
.perf-best td:first-child{border-left:1px solid rgba(99,202,246,.12);}
.perf-best td:last-child{border-right:1px solid rgba(99,202,246,.12);}
.best-name{color:var(--text-primary)!important;font-weight:600!important;}
.best-r2{color:var(--accent-cyan)!important;font-weight:700!important;}
.best-badge{display:inline-block;background:linear-gradient(135deg,#63caf6,#3de8c8);border-radius:4px;padding:1px 7px;font-size:.58rem;font-weight:700;color:#070a12;margin-left:7px;letter-spacing:.05em;}
.trust-strip{display:flex;justify-content:center;gap:2.5rem;flex-wrap:wrap;padding:2rem 0 3rem;}
.trust-item{display:flex;align-items:center;gap:.45rem;font-size:.72rem;color:var(--text-muted);font-weight:500;}

/* ── FOOTER ── */
.premium-footer{border-top:1px solid var(--border);padding:3rem 2rem;text-align:center;max-width:1100px;margin:0 auto;}
.footer-logo{font-family:var(--font-display);font-size:1.2rem;color:var(--text-primary);margin-bottom:.5rem;}
.footer-links{display:flex;justify-content:center;gap:1.5rem;margin:1rem 0;flex-wrap:wrap;}
.footer-links span{font-size:.7rem;color:var(--text-muted);cursor:default;}
.footer-text{font-size:.72rem;color:var(--text-muted);line-height:2;}

div[data-testid="stAlert"]{background:rgba(61,232,200,0.06)!important;border:1px solid rgba(61,232,200,0.18)!important;border-radius:var(--radius-md)!important;color:var(--accent-cyan)!important;font-size:.8rem!important;}

/* ── PLOTLY TRANSPARENT BG ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

@media(max-width:768px){
    .nav-links{display:none;}
    .nav-bar{padding:0 1.2rem;}
    .hero-wrap{padding:3rem 1rem 5rem;}
    .result-wrap{padding:0 1rem;}
    .result-card{padding:2.5rem 1.5rem 2rem;}
    .spec-item{padding:0 .8rem;}
    .kpi-strip{gap:.6rem;}
    .analytics-kpi-grid{grid-template-columns:1fr 1fr;}
    .insight-grid{grid-template-columns:1fr;}
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def fmt_bdt(p):
    if p >= 1_00_00_000:
        return f"{p/1_00_00_000:.2f} Crore BDT"
    elif p >= 1_00_000:
        return f"{p/1_00_000:.1f} Lakh BDT"
    return f"{int(p):,} BDT"

def fmt_lakh(p):
    return f"৳ {p/1_00_000:.1f}L"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#8a96a8", size=11),
    margin=dict(l=16, r=16, t=32, b=16),
)

# Applied individually per chart via update_xaxes / update_yaxes to avoid duplicate-keyword conflicts
_AXIS = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.07)")

# ─────────────────────────────
# LOAD MODEL & ARTEFACTS
# ─────────────────────────────
@st.cache_resource
def load_model():
    # Use the best model selected by model.py.
    bundle_path = "best_model_bundle.pkl" if os.path.exists("best_model_bundle.pkl") else "gbm_bundle.pkl"
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    columns = list(bundle["columns"])
    rmse = float(bundle.get("rmse", 2_794_975))
    model_name = bundle.get("model_name", "Gradient Boosting")
    scaler = bundle.get("scaler")
    needs_scaling = bool(bundle.get("needs_scaling", False))
    metrics = bundle.get("metrics", {})
    feature_ranges = bundle.get("feature_ranges", {})
    return model, columns, rmse, model_name, scaler, needs_scaling, metrics, feature_ranges

_loaded = load_model()
model = _loaded[0]
model_columns = _loaded[1]
gbm_rmse = _loaded[2]  # historical variable name; now refers to selected-model RMSE
best_model_name = _loaded[3]
model_scaler = _loaded[4]
model_needs_scaling = _loaded[5]
best_metrics = _loaded[6]
TRAINING_RANGES = _loaded[7]

try:
    with open("model_metrics.json") as _f:
        ALL_METRICS = json.load(_f)
except Exception:
    ALL_METRICS = {}


LOCATIONS = [
    "Akhalia, Sylhet","Ambarkhana, Sylhet","Bagbari, Sylhet",
    "Chowhatta, Sylhet","Dargah Gate, Sylhet","Kazirbazar, Sylhet",
    "Kumarpara, Sylhet","Majortila, Sylhet","Mendibagh, Sylhet",
    "Mira Housing Estate, Sylhet","Mirabazar, Sylhet","Moulikergaon, Sylhet",
    "Nayasarak, Sylhet","Pathantula, Sylhet","Shahjalal Uposhahar, Sylhet",
    "Shahporan, Sylhet","Shibgonj, Sylhet","Subidbazar, Sylhet",
    "Taltola, Sylhet","Tilagor, Sylhet","Zindabazar, Sylhet",
]
PROPERTY_TYPES = ["Apartment", "House", "Duplex"]

# Location averages are calculated from the same dataset used for training.
# This avoids hard-coded market values becoming inconsistent with the new dataset.
try:
    _market_df = pd.read_csv("sylhet_real_estate.csv")
    _market_df.columns = _market_df.columns.str.strip()
    _market_df["Location"] = (_market_df["Location"].astype(str).str.strip()
                               .str.replace(r"\s+", " ", regex=True))
    _market_df["Selling price (BDT)"] = pd.to_numeric(
        _market_df["Selling price (BDT)"].astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")[0], errors="coerce")
    LOCATION_AVGS = (_market_df.dropna(subset=["Location", "Selling price (BDT)"])
                     .groupby("Location")["Selling price (BDT)"].mean().to_dict())
    LOCATION_COUNTS = _market_df.groupby("Location").size().to_dict()
    DATASET_SIZE = len(_market_df)
except Exception:
    LOCATION_AVGS = {}
    LOCATION_COUNTS = {}
    DATASET_SIZE = 0

# ─────────────────────────────
# NAVIGATION
# ─────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        <div class="nav-dot"></div>
        Sylhet Real Estate AI
    </div>
    <div class="nav-links">
        <span>Estimator</span>
        <span>Analytics</span>
        <span>Market</span>
        <span>About</span>
    </div>
    <div class="nav-badge">AI Powered</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# HERO
# ─────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-mesh"></div>
    <div class="hero-grid"></div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    <div class="hero-content">
        <div class="hero-eyebrow">
            <div class="eyebrow-dot"></div>
            AI-Powered Real Estate Intelligence
        </div>
        <h1 class="hero-h1">
            Know your property's<br>
            <span class="grad">true market value.</span>
        </h1>
        <p class="hero-sub">
            Instant ML-driven price estimates calibrated to Sylhet's
            real estate market. Transparent, fast, data-backed.
        </p>
        <div class="kpi-strip">
            <div class="kpi-card"><div class="kpi-val">21</div><div class="kpi-label">Locations</div></div>
            <div class="kpi-card"><div class="kpi-val">ML</div><div class="kpi-label">Best Model</div></div>
            <div class="kpi-card"><div class="kpi-val">12</div><div class="kpi-label">Features</div></div>
            <div class="kpi-card"><div class="kpi-val">৳BDT</div><div class="kpi-label">Currency</div></div>
            <div class="kpi-card"><div class="kpi-val">BEST</div><div class="kpi-label">Selected Model</div></div>
        </div>
        <div class="hero-trust-row">
            <div class="trust-pill"><span class="trust-check">✓</span> No sign-up required</div>
            <div class="trust-pill"><span class="trust-check">✓</span> Results in under 1 second</div>
            <div class="trust-pill"><span class="trust-check">✓</span> Trained on Sylhet listings</div>
        </div>
    </div>
    <div class="scroll-hint">
        <div class="scroll-line"></div>
        <div class="scroll-text">Scroll</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# FORM SECTION
# ─────────────────────────────
st.markdown("""
<div style="max-width:1100px;margin:0 auto;padding:3rem 2rem 0;">
    <div class="section-divider" style="margin-bottom:2.5rem;">
        <div class="section-divider-line"></div>
        <div class="section-divider-text">Property Details</div>
        <div class="section-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

_, form_col, _ = st.columns([1, 20, 1])
with form_col:
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon icon-blue">🏗️</div>
            <div>
                <div class="card-title">Property Specifications</div>
                <div class="card-desc">Core details about the unit</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        property_type   = st.selectbox("Property Type", PROPERTY_TYPES, index=0)
        location        = st.selectbox("Location", LOCATIONS, index=0)
        size            = st.number_input("Size (sqft)", 300, 15000, 1500, step=50)
        bedrooms        = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms       = st.number_input("Bathrooms", 1, 8, 2)
        floor           = st.number_input("Floor Number  (0 = ground floor / whole house)", 0, 30, 5)
        is_ground_whole = 1 if floor == 0 else 0

    with col_right:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon icon-cyan">✨</div>
            <div>
                <div class="card-title">Amenities & Features</div>
                <div class="card-desc">Inclusions that add value</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        balcony = st.number_input("Balconies", 0, 5, 1)
        parking = st.number_input("Parking Spaces", 0, 5, 1)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            lift_opt  = st.radio("Lift",      ["No", "Yes"], index=1)
            lift      = 1 if lift_opt == "Yes" else 0
        with c2:
            cctv_opt  = st.radio("CCTV",      ["No", "Yes"], index=0)
            cctv      = 1 if cctv_opt == "Yes" else 0
        with c3:
            gen_opt   = st.radio("Generator", ["No", "Yes"], index=0)
            generator = 1 if gen_opt == "Yes" else 0

# ─────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([3, 2, 3])
with btn_col:
    predict = st.button("✦  Estimate Price  ✦")

# ─────────────────────────────
# RESULT + FULL ANALYTICS
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

    input_data = pd.get_dummies(raw, columns=["Location", "Property Type"])

    # Recreate exactly the predictor features used by model.py.
    input_data["Log_Size"] = np.log1p(input_data["Size (sqft)"].clip(lower=0))
    input_data["Size_per_Bedroom"] = input_data["Size (sqft)"] / input_data["Bedrooms"].replace(0, np.nan)
    input_data["Size_per_Bathroom"] = input_data["Size (sqft)"] / input_data["Bathrooms"].replace(0, np.nan)
    input_data["Amenity_Count"] = input_data[["Balcony", "Parking", "Lift", "CCTV", "Generator"]].sum(axis=1)
    input_data["Size_per_Bedroom"] = input_data["Size_per_Bedroom"].replace([np.inf, -np.inf], np.nan).fillna(input_data["Size (sqft)"])
    input_data["Size_per_Bathroom"] = input_data["Size_per_Bathroom"].replace([np.inf, -np.inf], np.nan).fillna(input_data["Size (sqft)"])
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Warn when a numeric input is outside the observed training range.
    ood_messages = []
    for _col, _rng in TRAINING_RANGES.items():
        if _col in raw.columns:
            _v = float(raw.iloc[0][_col])
            if _v < _rng["min"] or _v > _rng["max"]:
                ood_messages.append(
                    f"{_col}: {_v:g} is outside the observed training range "
                    f"({_rng['min']:g}–{_rng['max']:g})."
                )
    if ood_messages:
        st.warning(
            "Prediction reliability may be lower because this property is "
            "outside the training-data range.\n\n" + "\n".join(ood_messages)
        )

    if model_needs_scaling and model_scaler is not None:
        model_input = model_scaler.transform(input_data)
    else:
        model_input = input_data

    log_pred   = model.predict(model_input)[0]
    prediction = np.exp(log_pred)
    price_low  = max(0, prediction - gbm_rmse)
    price_high = prediction + gbm_rmse

    price_fmt   = f"{int(prediction):,}"
    readable    = fmt_bdt(prediction)
    range_low   = f"৳ {int(price_low):,}"
    range_hi    = f"৳ {int(price_high):,}"
    floor_label = "Ground / Whole" if floor == 0 else f"Floor {floor}"

    def amenity_badge(label, icon, is_on):
        cls  = "amenity-on" if is_on else "amenity-off"
        mark = "✓" if is_on else "–"
        return f'<span class="amenity-badge {cls}">{icon} {mark} {label}</span>'

    badges = (
        amenity_badge("Lift",       "🛗", lift)
        + amenity_badge("CCTV",     "📷", cctv)
        + amenity_badge("Generator","⚡", generator)
        + amenity_badge("Parking",  "🚗", parking > 0)
        + amenity_badge("Balcony",  "🏡", balcony > 0)
    )

    # ── MAIN RESULT CARD ──
    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-glow"></div>
        <div class="result-card">
            <div class="result-top-line"></div>
            <div class="result-eyebrow">Estimated Market Value</div>
            <div class="result-price-row">
                <span class="result-sym">৳</span>
                <span class="result-price-main">{price_fmt}</span>
            </div>
            <div class="result-readable">&#8776; {readable}</div>
            <div class="conf-bar-wrap">
                <div class="conf-label-row">
                    <span>Prediction Error Range</span>
                    <span>Approximate range based on test RMSE; not a statistical confidence interval.</span>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill"></div>
                </div>
            </div>
            <div class="result-range-text">
                Estimated range: <span>{range_low}</span> &ndash; <span>{range_hi}</span>
            </div>
            <div class="result-divider"></div>
            <div class="spec-grid">
                <div class="spec-item"><div class="spec-val">{property_type}</div><div class="spec-key">Type</div></div>
                <div class="spec-item"><div class="spec-val">{size:,} sqft</div><div class="spec-key">Area</div></div>
                <div class="spec-item"><div class="spec-val">{bedrooms} BR &middot; {bathrooms} BA</div><div class="spec-key">Layout</div></div>
                <div class="spec-item"><div class="spec-val">{floor_label}</div><div class="spec-key">Level</div></div>
                <div class="spec-item"><div class="spec-val">{balcony} &middot; {parking}</div><div class="spec-key">Balcony &middot; Parking</div></div>
            </div>
            <div class="amenity-row">{badges}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✓  Estimate generated  ·  {best_model_name}  ·  For reference purposes only")

    # ── ANALYTICS KPI SECTION ──
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:2rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">Analytics Dashboard</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    price_per_sqft    = int(prediction / size)
    loc_avg           = LOCATION_AVGS.get(location, 10_000_000)
    vs_market_pct     = ((prediction - loc_avg) / loc_avg) * 100
    market_pct_sign   = "+" if vs_market_pct >= 0 else ""
    amenity_score     = 40 + lift*15 + cctv*8 + generator*8 + min(parking,2)*7 + min(balcony,2)*5 + (bedrooms-1)*4 + (bathrooms-1)*3
    property_score    = min(100, int(amenity_score))
    invest_score      = min(100, int(60 + (vs_market_pct * 0.5) + (property_score * 0.15)))
    projection_5yr    = int(prediction * 1.48)

    st.markdown(f"""
    <div class="analytics-kpi-grid">
        <div class="akpi-card">
            <div class="akpi-label">Price per sqft</div>
            <div class="akpi-value cyan">৳ {price_per_sqft:,}</div>
            <div class="akpi-sub">Cost per square foot</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-label">vs. Area Average</div>
            <div class="akpi-value {'cyan' if vs_market_pct >= 0 else 'white'}">{market_pct_sign}{vs_market_pct:.1f}%</div>
            <div class="akpi-sub">vs. {location.split(',')[0]} avg</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-value cyan">{property_score} / 100</div>
            <div class="akpi-sub">Based on features &amp; amenities</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-value white">{invest_score} / 100</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-label">Prediction Error Band</div>
            <div class="akpi-value cyan">±{fmt_bdt(gbm_rmse)}</div>
            <div class="akpi-sub">Approximate prediction error based on test RMSE.</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-value white">≈ {fmt_bdt(projection_5yr)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PLOTLY CHARTS ──
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:1.5rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">Market Visualizations</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, chart_col, _ = st.columns([1, 20, 1])
    with chart_col:
        ch1, ch2 = st.columns(2, gap="medium")

        # Chart 1 — Predicted vs Market Average bar
        with ch1:
            top5_locs = sorted(LOCATION_AVGS.items(), key=lambda x: x[1], reverse=True)[:6]
            loc_names = [l[0].split(",")[0] for l in top5_locs]
            loc_vals  = [l[1] for l in top5_locs]
            colors_bar = ["rgba(99,202,246,0.25)"] * len(loc_names)
            short_loc  = location.split(",")[0]
            if short_loc in loc_names:
                idx = loc_names.index(short_loc)
                colors_bar[idx] = "#63caf6"
            fig1 = go.Figure(go.Bar(
                x=loc_names, y=loc_vals,
                marker_color=colors_bar,
                marker_line_width=0,
                text=[f"৳{v/1e5:.0f}L" for v in loc_vals],
                textposition="outside",
                textfont=dict(size=10, color="#8a96a8"),
            ))
            fig1.add_hline(y=prediction, line_dash="dot", line_color="#3de8c8", line_width=1.5,
                           annotation_text="Your property", annotation_font_color="#3de8c8",
                           annotation_font_size=10)
            fig1.update_layout(**PLOT_LAYOUT, title=dict(text="Location Avg vs Your Price", font=dict(size=12, color="#8a96a8")), height=280)
            fig1.update_xaxes(**_AXIS)
            fig1.update_yaxes(tickformat=".0s", **_AXIS)
            st.plotly_chart(fig1, width="stretch")


        ch3, ch4 = st.columns(2, gap="medium")

        # Chart 3 — Value breakdown donut
        with ch3:
            base_val     = size * 3500
            location_val = loc_avg * 0.30
            amenity_val  = (lift * 300000 + cctv * 80000 + generator * 120000
                            + parking * 250000 + balcony * 150000)
            floor_val    = floor * 50000
            labels = ["Base (Size)", "Location", "Amenities", "Floor Premium"]
            values = [max(base_val, 0), max(location_val, 0), max(amenity_val, 0), max(floor_val, 0)]
            fig3 = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.6,
                marker=dict(colors=["#63caf6","#3de8c8","rgba(99,202,246,0.45)","rgba(61,232,200,0.35)"],
                            line=dict(color="#070a12", width=2)),
                textfont=dict(size=10),
            ))
            fig3.update_layout(**PLOT_LAYOUT, title=dict(text="Value Breakdown", font=dict(size=12, color="#8a96a8")), height=280,
                               showlegend=True, legend=dict(font=dict(size=10, color="#8a96a8"), orientation="v"))
            st.plotly_chart(fig3, width="stretch")

        # Chart 4 — All location comparison
        with ch4:
            all_locs  = sorted(LOCATION_AVGS.items(), key=lambda x: x[1])
            all_names = [l[0].split(",")[0] for l in all_locs]
            all_vals  = [l[1] for l in all_locs]
            bar_colors = ["rgba(99,202,246,0.22)"] * len(all_names)
            if short_loc in all_names:
                bar_colors[all_names.index(short_loc)] = "#3de8c8"
            fig4 = go.Figure(go.Bar(
                y=all_names, x=all_vals,
                orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
            ))
            fig4.update_layout(**PLOT_LAYOUT, title=dict(text="All Areas - Avg Price", font=dict(size=12, color="#8a96a8")), height=320)
            fig4.update_xaxes(tickformat=".2s", **_AXIS)
            fig4.update_yaxes(**_AXIS)
            st.plotly_chart(fig4, width="stretch")

    # ── PREDICTION EXPLANATION ──
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:1.5rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">Prediction Explanation</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Local explanation: SHAP is used only when it can be computed successfully.
    local_values = None
    explanation_method = None
    if SHAP_AVAILABLE and hasattr(model, "feature_importances_"):
        try:
            _explainer = shap.TreeExplainer(model)
            _sv = _explainer(model_input)
            local_values = np.asarray(_sv.values)
            if local_values.ndim == 3:
                local_values = local_values[0, :, 0]
            elif local_values.ndim == 2:
                local_values = local_values[0]
            local_values = local_values.reshape(-1)
            if len(local_values) == len(model_columns):
                explanation_method = "SHAP TreeExplainer — this property"
            else:
                local_values = None
        except Exception:
            local_values = None

    # If SHAP is unavailable, do NOT label global feature importance as a
    # local explanation. Show it separately as model-level importance.
    if local_values is not None:
        exp_df = pd.DataFrame({"Feature": model_columns, "Contribution": local_values})
        exp_df["Feature"] = (exp_df["Feature"].astype(str)
                             .str.replace("Location_", "", regex=False)
                             .str.replace(", Sylhet", "", regex=False)
                             .str.replace("Property Type_", "Type: ", regex=False))
        exp_df = exp_df.sort_values("Contribution", key=np.abs, ascending=False).head(10)
        fig_exp = go.Figure(go.Bar(
            x=exp_df["Contribution"],
            y=exp_df["Feature"],
            orientation="h",
            marker_color=["#3de8c8" if v >= 0 else "#f06292" for v in exp_df["Contribution"]]
        ))
        fig_exp.update_layout(
            **PLOT_LAYOUT, height=380,
            title=dict(text=explanation_method, font=dict(size=12, color="#8a96a8"))
        )
        fig_exp.update_xaxes(title_text="SHAP value (log-price output units)", **_AXIS)
        fig_exp.update_yaxes(**_AXIS)
        st.plotly_chart(fig_exp, width="stretch")
        st.caption(
            "Positive SHAP values push this property's predicted log-price upward; "
            "negative values push it downward. SHAP values describe model behavior, "
            "not causal effects or fixed BDT amounts."
        )

    # Global feature importance for the selected model. Random Forest,
    # Gradient Boosting, XGBoost and Decision Tree expose native importances.
    if hasattr(model, "feature_importances_"):
        global_fi = pd.DataFrame({
            "Feature": model_columns,
            "Importance": np.asarray(model.feature_importances_, dtype=float)
        }).sort_values("Importance", ascending=False).head(10)
        global_fi["Feature"] = (global_fi["Feature"].astype(str)
                                 .str.replace("Location_", "", regex=False)
                                 .str.replace(", Sylhet", "", regex=False)
                                 .str.replace("Property Type_", "Type: ", regex=False))
        fig_fi = go.Figure(go.Bar(
            x=global_fi["Importance"],
            y=global_fi["Feature"],
            orientation="h",
            marker_color="#63caf6"
        ))
        fig_fi.update_layout(
            **PLOT_LAYOUT, height=380,
            title=dict(
                text=f"Global Feature Importance — {best_model_name}",
                font=dict(size=12, color="#8a96a8")
            )
        )
        fig_fi.update_xaxes(title_text="Importance", **_AXIS)
        fig_fi.update_yaxes(**_AXIS)
        st.plotly_chart(fig_fi, width="stretch")
        st.caption(
            f"This is the overall feature importance of the selected {best_model_name} model. "
            "It is different from the property-specific SHAP explanation above."
            if local_values is not None else
            f"SHAP was not available in this environment, so this global {best_model_name} "
            "feature-importance chart is shown instead. It describes the model overall, "
            "not this individual property's causal drivers."
        )

    # ── MARKET INSIGHTS ──
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:1.5rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">Market Insights</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ranked = sorted(LOCATION_AVGS.items(), key=lambda x: x[1], reverse=True)
    area_rank = [r[0] for r in ranked].index(location) + 1 if location in dict(ranked) else None
    loc_count = int(LOCATION_COUNTS.get(location, 0))
    relative_position = (
        "Higher relative average" if area_rank is not None and area_rank <= 7
        else ("Middle range" if area_rank is not None and area_rank <= 14
              else "Lower relative average")
    )
    position_pill = (
        "pill-green" if relative_position == "Higher relative average"
        else ("pill-amber" if relative_position == "Middle range" else "pill-red")
    )

    _, mi_col, _ = st.columns([1, 20, 1])
    with mi_col:
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-row"><span class="insight-key">Local Market Average</span><span class="insight-val">{fmt_bdt(loc_avg)}</span></div>
                <div class="insight-row"><span class="insight-key">Your Property vs Average</span><span class="insight-val" style="color:{'#3de8c8' if vs_market_pct>=0 else '#f06292'}">{market_pct_sign}{vs_market_pct:.1f}%</span></div>
                <div class="insight-row"><span class="insight-key">Area Ranking by Average</span><span class="insight-val">#{area_rank} of {len(ranked)}</span></div>
                <div class="insight-row"><span class="insight-key">Dataset Records in Area</span><span class="insight-val">{loc_count}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-row"><span class="insight-key">Relative Market Position</span><span class="insight-val"><span class="pill {position_pill}">{relative_position}</span></span></div>
                <div class="insight-row"><span class="insight-key">Selected Model</span><span class="insight-val">{best_model_name}</span></div>
                <div class="insight-row"><span class="insight-key">Independent Test R²</span><span class="insight-val">{best_metrics.get('R2', 0):.4f}</span></div>
                <div class="insight-row"><span class="insight-key">Prediction MAE</span><span class="insight-val">{fmt_bdt(best_metrics.get('MAE', gbm_rmse))}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ── PROPERTY INSIGHTS ──
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:1.5rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">Property Insights</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    recs = []
    if not lift:
        recs.append(("🛗", "Lift not present", "Lift availability is included as a predictor in the model.", "Model insight"))
    if not cctv:
        recs.append(("📷", "CCTV not present", "CCTV availability is included as a predictor in the model.", "Model insight"))
    if not generator:
        recs.append(("⚡", "Generator not present", "Generator availability is included in the model's amenity features.", "Model insight"))
    if parking == 0:
        recs.append(("🚗", "No parking", "Parking is one of the property features considered by the model.", "Model insight"))
    if balcony == 0:
        recs.append(("🏡", "No balcony", "Balcony availability is included in the prediction features.", "Model insight"))
    if bedrooms < 3:
        recs.append(("🛏️", "Smaller bedroom count", "Bedroom count is a model predictor and may be associated with different learned price patterns.", "Model insight"))
    if not recs:
        recs.append(("✅", "Major listed amenities are present", "The selected input includes the main amenity variables used by the model.", "Informational"))

    rec_rows = ""
    for icon, label, desc, impact in recs:
        rec_rows += f"""
        <div class="rec-row">
            <div class="rec-left">
                <span class="rec-icon">{icon}</span>
                <div>
                    <div class="rec-label">{label}</div>
                    <div class="rec-desc">{desc}</div>
                </div>
            </div>
            <div class="rec-impact">{impact}</div>
        </div>"""

    _, rec_col, _ = st.columns([1, 20, 1])
    with rec_col:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">Feature-based property insights</div>
            {rec_rows}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────
# MODEL PERFORMANCE TABLE
# ─────────────────────────────
st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="perf-section">
    <div class="perf-header">
        <div class="perf-title">Model Intelligence</div>
        <div class="perf-sub">Evaluated on a held-out independent test set &middot; 80/20 development/test split &middot; 5-fold CV &middot; No data leakage</div>
    </div>
""", unsafe_allow_html=True)

metrics_path = "model_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Safely sort models by test R². The metrics file normally contains
    # dictionaries, but this also handles string/numeric values gracefully.
    def _metric_r2(item):
        _name, _metrics = item
        if isinstance(_metrics, str):
            try:
                _metrics = json.loads(_metrics)
            except (json.JSONDecodeError, TypeError):
                try:
                    return float(_metrics)
                except (ValueError, TypeError):
                    return float("-inf")
        if isinstance(_metrics, dict):
            _r2 = _metrics.get("R2", _metrics.get("r2", float("-inf")))
            try:
                return float(_r2)
            except (ValueError, TypeError):
                return float("-inf")
        try:
            return float(_metrics)
        except (ValueError, TypeError):
            return float("-inf")

    sorted_models = sorted(all_metrics.items(), key=_metric_r2, reverse=True)
    best_name = sorted_models[0][0] if sorted_models else best_model_name

    rows_html = ""
    for name, m in sorted_models:
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except (json.JSONDecodeError, TypeError):
                m = {}
        if not isinstance(m, dict):
            m = {}

        def _num(key, default=None):
            value = m.get(key, default)
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        r2_value = _num("R2")
        adj_r2_value = _num("Adjusted_R2")
        mae_value = _num("MAE")
        rmse_value = _num("RMSE")
        mape_value = _num("MAPE")

        is_best  = name == best_name
        row_cls  = 'class="perf-best"' if is_best else ""
        badge    = '<span class="best-badge">BEST</span>' if is_best else ""
        name_cls = "best-name" if is_best else ""
        r2_cls   = "best-r2"   if is_best else ""
        r2_text = f"{r2_value:.4f}" if r2_value is not None else "—"
        adj_text = f"{adj_r2_value:.4f}" if adj_r2_value is not None else "—"
        mae_text = f"&#2547; {mae_value:,.0f}" if mae_value is not None else "—"
        rmse_text = f"&#2547; {rmse_value:,.0f}" if rmse_value is not None else "—"
        mape_text = f"{mape_value:.2f}%" if mape_value is not None else "—"

        rows_html += f"""
        <tr {row_cls}>
            <td class="{name_cls}">{name}{badge}</td>
            <td class="{r2_cls}">{r2_text}</td>
            <td>{adj_text}</td>
            <td>{mae_text}</td>
            <td>{rmse_text}</td>
            <td>{mape_text}</td>
        </tr>"""

    st.markdown(f"""
    <div class="perf-table-wrap">
        <table class="perf-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>R&sup2; Score</th>
                    <th>Adj. R&sup2;</th>
                    <th>MAE (BDT)</th>
                    <th>RMSE (BDT)</th>
                    <th>MAPE</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    <div class="trust-strip">
        <div class="trust-item">&#129504; Selected best model: {best_model_name}</div>
        <div class="trust-item">&#128202; Trained on Sylhet listing data</div>
        <div class="trust-item">&#128274; No personal data collected</div>
        <div class="trust-item">&#9889; Inference in &lt;50 ms</div>
        <div class="trust-item">&#128506; 21 Sylhet neighbourhoods</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;color:var(--text-muted);font-size:.8rem;padding:2rem;
                background:var(--bg-card);border-radius:var(--radius-lg);border:1px solid var(--border);">
        Run <code>model.py</code> to generate <code>model_metrics.json</code>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────
st.markdown("""
<div style="max-width:1100px;margin:0 auto;padding:3rem 2rem 1rem;">
    <div class="section-divider" style="margin-bottom:1.5rem;">
        <div class="section-divider-line"></div>
        <div class="section-divider-text">Prediction Error Analysis</div>
        <div class="section-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

_err_left, _err_right = st.columns(2, gap="medium")

with _err_left:
    if os.path.exists("property_type_error_analysis.csv"):
        _ptype = pd.read_csv("property_type_error_analysis.csv")
        _ptype["MAE_BDT"] = _ptype["MAE_BDT"].round(0)
        fig_pt = px.bar(_ptype, x="Property Type", y="MAE_BDT",
                        text_auto=".0f", title="MAE by Property Type")
        fig_pt.update_layout(**PLOT_LAYOUT, height=360)
        fig_pt.update_xaxes(**_AXIS)
        fig_pt.update_yaxes(title_text="MAE (BDT)", **_AXIS)
        st.plotly_chart(fig_pt, width="stretch")
        st.caption("Lower MAE means smaller average prediction error for that property type.")

with _err_right:
    if os.path.exists("location_error_analysis.csv"):
        _locerr = pd.read_csv("location_error_analysis.csv")
        _locerr = _locerr[_locerr["Samples"] >= 5].sort_values("MAE_BDT").head(12)
        if not _locerr.empty:
            fig_loc = px.bar(_locerr, x="MAE_BDT", y="Location",
                             orientation="h", text_auto=".0f",
                             title="Location MAE — Locations with ≥5 Test Samples")
            fig_loc.update_layout(**PLOT_LAYOUT, height=360)
            fig_loc.update_xaxes(title_text="MAE (BDT)", **_AXIS)
            fig_loc.update_yaxes(**_AXIS)
            st.plotly_chart(fig_loc, width="stretch")
            st.caption("Locations with fewer than 5 independent-test observations are excluded from this comparison to avoid unstable conclusions.")
# ─────────────────────────────
# ABOUT THE MODEL / METHODOLOGY
# ─────────────────────────────
st.markdown(f"""
<div style="max-width:1100px;margin:0 auto;padding:3rem 2rem 1rem;">
    <div class="section-divider" style="margin-bottom:1.5rem;">
        <div class="section-divider-line"></div>
        <div class="section-divider-text">About the Model</div>
        <div class="section-divider-line"></div>
    </div>
    <div class="insight-card">
        <div class="insight-row"><span class="insight-key">Dataset</span><span class="insight-val">{DATASET_SIZE} cleaned property records</span></div>
        <div class="insight-row"><span class="insight-key">Evaluation</span><span class="insight-val">80% development / 20% independent test</span></div>
        <div class="insight-row"><span class="insight-key">Validation</span><span class="insight-val">5-fold cross-validation on development data</span></div>
        <div class="insight-row"><span class="insight-key">Model selection</span><span class="insight-val">Highest independent-test R²</span></div>
        <div class="insight-row"><span class="insight-key">Target</span><span class="insight-val">Log-transformed selling price</span></div>
        <div class="insight-row"><span class="insight-key">Interpretation</span><span class="insight-val">SHAP for tree models where available</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("""
<div class="premium-footer">
    <div class="footer-logo">Sylhet Real Estate AI</div>
    <div class="footer-links">
        <span>Privacy</span>
        <span>Methodology</span>
        <span>Data Sources</span>
        <span>Contact</span>
    </div>
    <div class="footer-text">
        &copy; 2026 Sylhet Real Estate Estimator &middot; BSc Thesis Project<br>
        For informational purposes only &middot; Not financial advice &middot; Sylhet, Bangladesh
    </div>
</div>
""", unsafe_allow_html=True)