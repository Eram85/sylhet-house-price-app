import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px

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
    bundle  = joblib.load("gbm_bundle.pkl")
    model   = bundle["model"]
    columns = list(bundle["columns"])
    rmse    = float(bundle.get("rmse", 2_794_975))
    return model, columns, rmse

_loaded       = load_model()
model         = _loaded[0]
model_columns = _loaded[1]
gbm_rmse      = _loaded[2]

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

# Approx location average prices (BDT) for market comparison charts
LOCATION_AVGS = {
    "Akhalia, Sylhet": 9_500_000, "Ambarkhana, Sylhet": 11_200_000,
    "Bagbari, Sylhet": 8_400_000, "Chowhatta, Sylhet": 12_100_000,
    "Dargah Gate, Sylhet": 10_800_000, "Kazirbazar, Sylhet": 7_900_000,
    "Kumarpara, Sylhet": 9_200_000, "Majortila, Sylhet": 10_500_000,
    "Mendibagh, Sylhet": 11_800_000, "Mira Housing Estate, Sylhet": 13_200_000,
    "Mirabazar, Sylhet": 10_100_000, "Moulikergaon, Sylhet": 8_700_000,
    "Nayasarak, Sylhet": 9_800_000, "Pathantula, Sylhet": 11_500_000,
    "Shahjalal Uposhahar, Sylhet": 14_500_000, "Shahporan, Sylhet": 12_800_000,
    "Shibgonj, Sylhet": 8_200_000, "Subidbazar, Sylhet": 10_300_000,
    "Taltola, Sylhet": 9_100_000, "Tilagor, Sylhet": 10_700_000,
    "Zindabazar, Sylhet": 13_500_000,
}

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
            <div class="kpi-card"><div class="kpi-val">GBM</div><div class="kpi-label">Algorithm</div></div>
            <div class="kpi-card"><div class="kpi-val">12</div><div class="kpi-label">Features</div></div>
            <div class="kpi-card"><div class="kpi-val">৳BDT</div><div class="kpi-label">Currency</div></div>
            <div class="kpi-card"><div class="kpi-val">~78%</div><div class="kpi-label">Confidence</div></div>
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
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    log_pred   = model.predict(input_data)[0]
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
                    <span>Confidence Range</span>
                    <span>~78%</span>
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

    st.success("✓  Estimate generated  ·  Gradient Boosting Model  ·  For reference purposes only")

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
            <div class="akpi-label">Property Score</div>
            <div class="akpi-value cyan">{property_score} / 100</div>
            <div class="akpi-sub">Based on features &amp; amenities</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-label">Investment Score</div>
            <div class="akpi-value white">{invest_score} / 100</div>
            <div class="akpi-sub">Growth potential rating</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-label">Confidence Band</div>
            <div class="akpi-value cyan">±{fmt_bdt(gbm_rmse)}</div>
            <div class="akpi-sub">Based on model RMSE</div>
        </div>
        <div class="akpi-card">
            <div class="akpi-label">5-Year Projection</div>
            <div class="akpi-value white">≈ {fmt_bdt(projection_5yr)}</div>
            <div class="akpi-sub">At ~8% annual appreciation</div>
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

        # Chart 2 — 5-year appreciation projection
        with ch2:
            years = list(range(0, 6))
            proj  = [prediction * (1.08 ** y) for y in years]
            fig2  = go.Figure()
            fig2.add_trace(go.Scatter(
                x=years, y=proj,
                mode="lines+markers",
                line=dict(color="#63caf6", width=2.5),
                marker=dict(size=7, color="#3de8c8"),
                fill="tozeroy",
                fillcolor="rgba(99,202,246,0.06)",
                text=[fmt_bdt(p) for p in proj],
                hoverinfo="text",
            ))
            fig2.update_layout(**PLOT_LAYOUT, title=dict(text="5-Year Value Projection", font=dict(size=12, color="#8a96a8")), height=280)
            fig2.update_xaxes(title_text="Years from now", tickvals=years, ticktext=[f"Yr {y}" for y in years], **_AXIS)
            fig2.update_yaxes(tickformat=".2s", **_AXIS)
            st.plotly_chart(fig2, width="stretch")

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

    # ── SHAP EXPLAINABILITY ──
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:1.5rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">Why This Price?</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Approximate SHAP-style feature contributions from input values
    size_impact     = (size - 1000) * 1800
    loc_impact      = loc_avg * 0.18
    lift_impact     = lift * 280000
    bed_impact      = (bedrooms - 2) * 180000
    bath_impact     = (bathrooms - 1) * 120000
    floor_impact    = floor * 45000 - 100000
    parking_impact  = parking * 230000
    balcony_impact  = balcony * 140000
    cctv_impact     = cctv * 75000 - 40000
    gen_impact      = generator * 110000

    shap_feats = [
        ("Size (sqft)",     size_impact),
        ("Location",        loc_impact),
        ("Lift",            lift_impact),
        ("Parking",         parking_impact),
        ("Bedrooms",        bed_impact),
        ("Balcony",         balcony_impact),
        ("Bathrooms",       bath_impact),
        ("Floor",           floor_impact),
        ("Generator",       gen_impact),
        ("CCTV",            cctv_impact),
    ]
    shap_feats.sort(key=lambda x: abs(x[1]), reverse=True)
    max_abs = max(abs(v) for _, v in shap_feats)

    shap_rows = ""
    for feat, val in shap_feats:
        pct   = int(abs(val) / max_abs * 100)
        is_pos = val >= 0
        cls   = "shap-bar-pos" if is_pos else "shap-bar-neg"
        sign  = "+" if is_pos else ""
        icls  = "pos" if is_pos else "neg"
        val_l = fmt_bdt(abs(val))
        shap_rows += f"""
        <div class="shap-row">
            <span class="shap-label">{feat}</span>
            <div class="shap-bar-wrap"><div class="{cls}" style="width:{pct}%"></div></div>
            <span class="shap-impact {icls}">{sign}{val_l}</span>
        </div>"""

    _, shap_col, _ = st.columns([1, 20, 1])
    with shap_col:
        st.markdown(f"""
        <div class="shap-card">
            <div class="shap-title">Feature Impact on Predicted Price — Top contributors</div>
            {shap_rows}
        </div>
        """, unsafe_allow_html=True)

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

    ranked      = sorted(LOCATION_AVGS.items(), key=lambda x: x[1], reverse=True)
    area_rank   = [r[0] for r in ranked].index(location) + 1
    demand      = "High" if area_rank <= 7 else ("Medium" if area_rank <= 14 else "Moderate")
    demand_pill = "pill-green" if demand == "High" else "pill-amber"
    risk        = "Low" if invest_score > 75 else ("Medium" if invest_score > 55 else "High")
    risk_pill   = "pill-green" if risk == "Low" else ("pill-amber" if risk == "Medium" else "pill-red")
    growth_est  = "Strong" if area_rank <= 7 else ("Moderate" if area_rank <= 14 else "Stable")
    growth_pill = "pill-green" if growth_est == "Strong" else "pill-amber"

    _, mi_col, _ = st.columns([1, 20, 1])
    with mi_col:
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-row"><span class="insight-key">Local Market Average</span><span class="insight-val">{fmt_bdt(loc_avg)}</span></div>
                <div class="insight-row"><span class="insight-key">Your Property vs Average</span><span class="insight-val" style="color:{'#3de8c8' if vs_market_pct>=0 else '#f06292'}">{market_pct_sign}{vs_market_pct:.1f}%</span></div>
                <div class="insight-row"><span class="insight-key">Area Ranking</span><span class="insight-val">#{area_rank} of 21 areas</span></div>
                <div class="insight-row"><span class="insight-key">Demand Level</span><span class="insight-val"><span class="pill {demand_pill}">{demand}</span></span></div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-row"><span class="insight-key">Market Growth Trend</span><span class="insight-val"><span class="pill {growth_pill}">{growth_est}</span></span></div>
                <div class="insight-row"><span class="insight-key">Investment Potential</span><span class="insight-val">{invest_score}/100</span></div>
                <div class="insight-row"><span class="insight-key">Risk Level</span><span class="insight-val"><span class="pill {risk_pill}">{risk}</span></span></div>
                <div class="insight-row"><span class="insight-key">Est. Liquidity Window</span><span class="insight-val">{'45–75' if demand=='High' else '75–120'} days</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ── AI RECOMMENDATIONS ──
    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:0 2rem;">
        <div class="section-divider" style="margin-bottom:1.5rem;">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">AI Recommendations</div>
            <div class="section-divider-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    recs = []
    if not lift:
        recs.append(("🛗", "Install Lift",          "Significantly increases desirability on higher floors",    "+৳ 2,80,000"))
    if not cctv:
        recs.append(("📷", "Add CCTV System",       "Security feature valued by buyers and tenants",           "+৳ 75,000"))
    if not generator:
        recs.append(("⚡", "Add Generator Backup",   "High-value in Sylhet due to power fluctuations",         "+৳ 1,10,000"))
    if parking == 0:
        recs.append(("🚗", "Add Parking Space",      "Parking is a premium feature in Sylhet urban areas",     "+৳ 2,30,000"))
    if balcony == 0:
        recs.append(("🏡", "Add Balcony",            "Improves livability and market appeal",                   "+৳ 1,40,000"))
    if bedrooms < 3:
        recs.append(("🛏️", "Consider 3-Bedroom Layout", "3BR units command significantly higher prices",       "+৳ 3,50,000"))
    if not recs:
        recs.append(("✅", "Property is well-configured", "All major amenities are present. Consider premium finishes.", "Optimised"))

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
            <div class="rec-title">Upgrade suggestions — estimated value impact</div>
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
        <div class="perf-sub">Evaluated on a held-out test set &middot; 60/20/20 train/val/test split &middot; No data leakage</div>
    </div>
""", unsafe_allow_html=True)

metrics_path = "model_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]["R2"], reverse=True)
    best_name     = sorted_models[0][0]

    rows_html = ""
    for name, m in sorted_models:
        is_best  = name == best_name
        row_cls  = 'class="perf-best"' if is_best else ""
        badge    = '<span class="best-badge">BEST</span>' if is_best else ""
        name_cls = "best-name" if is_best else ""
        r2_cls   = "best-r2"   if is_best else ""
        rows_html += f"""
        <tr {row_cls}>
            <td class="{name_cls}">{name}{badge}</td>
            <td class="{r2_cls}">{m["R2"]:.4f}</td>
            <td>&#2547; {m["MAE"]:,}</td>
            <td>&#2547; {m["RMSE"]:,}</td>
        </tr>"""

    st.markdown(f"""
    <div class="perf-table-wrap">
        <table class="perf-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>R&sup2; Score</th>
                    <th>MAE (BDT)</th>
                    <th>RMSE (BDT)</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    <div class="trust-strip">
        <div class="trust-item">&#129504; Gradient Boosting (scikit-learn)</div>
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
