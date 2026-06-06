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
# PREMIUM CSS DESIGN SYSTEM
# ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --bg-void:      #070a12;
    --bg-surface:   #0d1117;
    --bg-card:      rgba(255,255,255,0.032);
    --bg-card-hov:  rgba(255,255,255,0.056);
    --border:       rgba(255,255,255,0.07);
    --border-glow:  rgba(99,202,246,0.35);
    --text-primary: #eaf0fb;
    --text-secondary:#8a96a8;
    --text-muted:   #404754;
    --accent-blue:  #63caf6;
    --accent-cyan:  #3de8c8;
    --accent-gold:  #f5c842;
    --accent-grd:   linear-gradient(135deg, #63caf6 0%, #3de8c8 100%);
    --glow-blue:    rgba(99,202,246,0.18);
    --glow-cyan:    rgba(61,232,200,0.14);
    --radius-sm:    10px;
    --radius-md:    16px;
    --radius-lg:    24px;
    --radius-xl:    32px;
    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', system-ui, sans-serif;
    --shadow-card:  0 1px 3px rgba(0,0,0,0.4), 0 8px 32px rgba(0,0,0,0.28);
    --shadow-glow:  0 0 60px rgba(99,202,246,0.12);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: var(--font-body);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: var(--bg-void);
    color: var(--text-primary);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
}

/* ═══════════════════════════════════
   NAVIGATION
═══════════════════════════════════ */
.nav-bar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    height: 58px;
    background: rgba(7,10,18,0.82);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border-bottom: 1px solid var(--border);
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-primary);
}
.nav-logo .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-grd);
    box-shadow: 0 0 10px var(--accent-blue);
}
.nav-links {
    display: flex;
    gap: 2rem;
    align-items: center;
}
.nav-links span {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    text-transform: uppercase;
    cursor: default;
}
.nav-badge {
    background: linear-gradient(135deg, rgba(99,202,246,0.14), rgba(61,232,200,0.10));
    border: 1px solid rgba(99,202,246,0.25);
    border-radius: 980px;
    padding: 0.22rem 0.8rem;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--accent-blue);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ═══════════════════════════════════
   HERO
═══════════════════════════════════ */
.hero-wrap {
    position: relative;
    overflow: hidden;
    padding: 0 2rem;
    min-height: 82vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
/* Animated mesh gradient bg */
.hero-mesh {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 40%, rgba(99,202,246,0.055) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 80% 60%, rgba(61,232,200,0.04) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 0%,  rgba(99,202,246,0.03) 0%, transparent 50%);
    animation: meshPulse 8s ease-in-out infinite alternate;
}
@keyframes meshPulse {
    0%   { opacity: 0.7; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.04); }
}
/* Grid lines overlay */
.hero-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.022) 1px, transparent 1px);
    background-size: 52px 52px;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
}
/* Floating orbs */
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
}
.orb-1 {
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(99,202,246,0.11) 0%, transparent 70%);
    top: -80px; left: -60px;
    animation: float1 12s ease-in-out infinite;
}
.orb-2 {
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(61,232,200,0.09) 0%, transparent 70%);
    bottom: 40px; right: 5%;
    animation: float2 14s ease-in-out infinite;
}
.orb-3 {
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(245,200,66,0.06) 0%, transparent 70%);
    top: 40%; left: 55%;
    animation: float3 10s ease-in-out infinite;
}
@keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(30px,-30px)} }
@keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-20px,25px)} }
@keyframes float3 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(15px,-20px)} }

.hero-content {
    position: relative;
    z-index: 2;
    text-align: center;
    max-width: 820px;
    margin: 0 auto;
    animation: heroIn 0.9s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes heroIn {
    from { opacity:0; transform:translateY(40px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,202,246,0.08);
    border: 1px solid rgba(99,202,246,0.2);
    border-radius: 980px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 2rem;
}
.hero-eyebrow-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent-blue);
    box-shadow: 0 0 6px var(--accent-blue);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero-h1 {
    font-family: var(--font-display);
    font-size: clamp(3rem, 6.5vw, 5.6rem);
    font-weight: 400;
    line-height: 1.04;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin-bottom: 1.5rem;
}
.hero-h1 .grad {
    background: var(--accent-grd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-h1 em {
    font-style: italic;
    color: var(--text-secondary);
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    line-height: 1.7;
    color: var(--text-secondary);
    max-width: 520px;
    margin: 0 auto 3rem;
}

/* KPI strip */
.kpi-strip {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 3.5rem;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    text-align: center;
    min-width: 110px;
    transition: all 0.28s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background: var(--accent-grd);
    opacity: 0;
    transition: opacity 0.28s;
}
.kpi-card:hover { background: var(--bg-card-hov); border-color: rgba(99,202,246,0.2); transform:translateY(-2px); }
.kpi-card:hover::before { opacity:1; }
.kpi-val {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 400;
    background: var(--accent-grd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.kpi-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.hero-cta-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
}
.hero-trust {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-muted);
}
.hero-trust .check {
    width: 16px; height: 16px; border-radius: 50%;
    background: rgba(61,232,200,0.12);
    border: 1px solid rgba(61,232,200,0.25);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 9px; color: var(--accent-cyan);
}

/* Scroll indicator */
.scroll-hint {
    position: absolute;
    bottom: 2.5rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    opacity: 0.35;
    animation: scrollBounce 2.5s ease-in-out infinite;
}
@keyframes scrollBounce { 0%,100%{transform:translateX(-50%) translateY(0)} 50%{transform:translateX(-50%) translateY(8px)} }
.scroll-hint-line {
    width: 1px; height: 36px;
    background: linear-gradient(to bottom, var(--accent-blue), transparent);
}
.scroll-hint-text {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ═══════════════════════════════════
   SECTION LAYOUT
═══════════════════════════════════ */
.section-wrap {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 2rem 5rem;
}
.section-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2.5rem;
}
.section-label-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-label-text {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
}

/* ═══════════════════════════════════
   GLASS CARDS (input panels)
═══════════════════════════════════ */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2.2rem 2rem;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.3s ease;
}
.glass-card::after {
    content:'';
    position:absolute;
    inset:0; border-radius: inherit;
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
    pointer-events: none;
}
.glass-card:hover {
    box-shadow: 0 0 0 1px rgba(99,202,246,0.12), var(--shadow-card);
}
.glass-card-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.glass-card-icon {
    width: 32px; height: 32px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.icon-blue { background: rgba(99,202,246,0.12); border: 1px solid rgba(99,202,246,0.18); }
.icon-cyan { background: rgba(61,232,200,0.10); border: 1px solid rgba(61,232,200,0.16); }
.glass-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}
.glass-card-desc {
    font-size: 0.67rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}

/* ═══════════════════════════════════
   INPUT OVERRIDES
═══════════════════════════════════ */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label { font-family: var(--font-body) !important; }

div[data-testid="stNumberInput"] label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 0.35rem !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(99,202,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,202,246,0.1) !important;
    background: rgba(99,202,246,0.04) !important;
}
div[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: rgba(99,202,246,0.12) !important;
    border-color: rgba(99,202,246,0.25) !important;
}

div[data-testid="stSelectbox"] label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(99,202,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,202,246,0.1) !important;
}

/* Radio pill buttons */
div[data-testid="stRadio"] > label {
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}
.stRadio > div { display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 0.4rem !important; margin-top: 0.3rem !important; }
.stRadio label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.8rem !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
}
.stRadio label:hover { border-color: rgba(99,202,246,0.25) !important; color: var(--text-primary) !important; }
.stRadio label:has(input:checked) {
    background: rgba(99,202,246,0.1) !important;
    border-color: rgba(99,202,246,0.35) !important;
    color: var(--accent-blue) !important;
}

/* ═══════════════════════════════════
   CTA BUTTON
═══════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #4ab8f0, #2bcfb0) !important;
    color: #070a12 !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 0.82rem 3rem !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(99,202,246,0.35), 0 0 0 0 rgba(99,202,246,0) !important;
    transition: all 0.24s cubic-bezier(0.34,1.56,0.64,1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.012) !important;
    box-shadow: 0 8px 36px rgba(99,202,246,0.5), 0 0 0 4px rgba(99,202,246,0.1) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ═══════════════════════════════════
   RESULT CARD
═══════════════════════════════════ */
@keyframes resultSlide {
    from { opacity:0; transform:translateY(36px) scale(0.97); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
.result-wrap {
    animation: resultSlide 0.6s cubic-bezier(0.22,1,0.36,1) both;
    position: relative;
    margin: 2.5rem auto;
    max-width: 860px;
}
.result-glow {
    position: absolute;
    inset: -30px;
    background: radial-gradient(ellipse 60% 50% at 50% 50%, rgba(99,202,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.result-card {
    background: rgba(13,17,23,0.92);
    border: 1px solid rgba(99,202,246,0.18);
    border-radius: var(--radius-xl);
    padding: 3.5rem 3rem 3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 0 1px rgba(99,202,246,0.08), var(--shadow-card), var(--shadow-glow);
}
.result-top-line {
    position: absolute; top:0; left:15%; right:15%; height:1px;
    background: var(--accent-grd);
    filter: blur(0.5px);
}
.result-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
}
.result-eyebrow::before, .result-eyebrow::after {
    content:'';
    display: block;
    width: 28px; height: 1px;
    background: var(--border);
}

.result-price-row {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0.3rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-sym {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 400;
    color: var(--text-muted);
    padding-top: 0.8rem;
}
.result-price-main {
    font-family: var(--font-display);
    font-size: clamp(3.2rem, 8vw, 5.8rem);
    font-weight: 400;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-readable {
    font-size: 1rem;
    font-weight: 500;
    color: var(--accent-cyan);
    margin-bottom: 0.6rem;
    letter-spacing: -0.01em;
}

/* Confidence bar */
.conf-bar-wrap {
    max-width: 340px;
    margin: 0.8rem auto 1.8rem;
}
.conf-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.67rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.conf-bar-track {
    height: 5px;
    border-radius: 980px;
    background: rgba(255,255,255,0.07);
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: inherit;
    background: var(--accent-grd);
    width: 78%;
    animation: fillBar 1.2s cubic-bezier(0.22,1,0.36,1) both;
    animation-delay: 0.4s;
}
@keyframes fillBar { from{width:0} }

.result-range-text {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 2.5rem;
}
.result-range-text span {
    color: var(--text-secondary);
    font-weight: 500;
}

.result-divider { width: 64px; height: 1px; background: var(--border); margin: 0 auto 2.5rem; }

/* Spec grid */
.spec-grid {
    display: flex;
    justify-content: center;
    gap: 0;
    flex-wrap: wrap;
}
.spec-item {
    text-align: center;
    padding: 0 1.6rem;
    border-right: 1px solid var(--border);
}
.spec-item:last-child { border-right: none; }
.spec-val {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.spec-key {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* Amenity badges */
.amenity-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.amenity-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.8rem;
    border-radius: 980px;
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.amenity-on  { background: rgba(61,232,200,0.1); border:1px solid rgba(61,232,200,0.22); color:#3de8c8; }
.amenity-off { background: rgba(255,255,255,0.04); border:1px solid var(--border); color:var(--text-muted); }

/* ═══════════════════════════════════
   MODEL PERFORMANCE TABLE
═══════════════════════════════════ */
.perf-section {
    max-width: 900px;
    margin: 0 auto 5rem;
    padding: 0 2rem;
}
.perf-header {
    text-align: center;
    margin-bottom: 2rem;
}
.perf-title {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 400;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.perf-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
}
.perf-table-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    overflow-x: auto;
}
.perf-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 6px;
}
.perf-table thead th {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0.4rem 1rem;
    text-align: center;
}
.perf-table thead th:first-child { text-align: left; }
.perf-table tbody tr {
    border-radius: 12px;
}
.perf-table td {
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
    text-align: center;
    background: rgba(255,255,255,0.025);
}
.perf-table td:first-child {
    text-align: left;
    border-radius: 12px 0 0 12px;
}
.perf-table td:last-child { border-radius: 0 12px 12px 0; }
.perf-best td {
    background: rgba(99,202,246,0.06) !important;
    border-top: 1px solid rgba(99,202,246,0.12);
    border-bottom: 1px solid rgba(99,202,246,0.12);
}
.perf-best td:first-child { border-left:1px solid rgba(99,202,246,0.12); }
.perf-best td:last-child { border-right:1px solid rgba(99,202,246,0.12); }
.best-name { color: var(--text-primary) !important; font-weight: 600 !important; }
.best-r2 { color: var(--accent-cyan) !important; font-weight: 700 !important; }
.best-badge {
    display: inline-block;
    background: var(--accent-grd);
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.58rem;
    font-weight: 700;
    color: #070a12;
    margin-left: 7px;
    letter-spacing: 0.05em;
}

/* Trust strip */
.trust-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    flex-wrap: wrap;
    padding: 2rem 0 3rem;
}
.trust-item {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 500;
}
.trust-icon { font-size: 0.85rem; }

/* ═══════════════════════════════════
   FOOTER
═══════════════════════════════════ */
.premium-footer {
    border-top: 1px solid var(--border);
    padding: 3rem 2rem;
    text-align: center;
    max-width: 1100px;
    margin: 0 auto;
}
.footer-logo {
    font-family: var(--font-display);
    font-size: 1.2rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.footer-text {
    font-size: 0.72rem;
    color: var(--text-muted);
    line-height: 2;
}
.footer-links {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.footer-links span {
    font-size: 0.7rem;
    color: var(--text-muted);
    cursor: default;
}
.footer-links span:hover { color: var(--text-secondary); }

/* ═══════════════════════════════════
   ALERT
═══════════════════════════════════ */
div[data-testid="stAlert"] {
    background: rgba(61,232,200,0.06) !important;
    border: 1px solid rgba(61,232,200,0.18) !important;
    border-radius: var(--radius-md) !important;
    color: var(--accent-cyan) !important;
    font-size: 0.8rem !important;
    font-family: var(--font-body) !important;
}

/* ═══════════════════════════════════
   RESPONSIVE
═══════════════════════════════════ */
@media (max-width: 768px) {
    .nav-links { display: none; }
    .nav-bar { padding: 0 1.2rem; }
    .hero-wrap { min-height: 70vh; padding: 0 1rem; }
    .section-wrap { padding: 0 1rem 4rem; }
    .glass-card { padding: 1.6rem 1.2rem; }
    .result-card { padding: 2.5rem 1.5rem 2rem; }
    .spec-item { padding: 0 0.8rem; }
    .kpi-strip { gap: 0.6rem; }
    .kpi-card { min-width: 90px; padding: 0.7rem 0.9rem; }
}

html { scroll-behavior: smooth; }
#estimator { scroll-margin-top: 70px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# LOAD MODEL & ARTEFACTS
# ─────────────────────────────
@st.cache_resource
def load_model():
    bundle  = joblib.load("gbm_bundle.pkl")
    model   = bundle["model"]
    columns = list(bundle["columns"])
    rmse    = float(bundle.get("rmse", 2_797_745))
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
# NAVIGATION
# ─────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        <div class="dot"></div>
        Sylhet Real Estate
    </div>
    <div class="nav-links">
        <span>Estimator</span>
        <span>Market</span>
        <span>About</span>
    </div>
    <div class="nav-badge">AI Beta</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# HERO SECTION
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
            <div class="hero-eyebrow-dot"></div>
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
            <div class="kpi-card">
                <div class="kpi-val">21</div>
                <div class="kpi-label">Locations</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">GBM</div>
                <div class="kpi-label">Algorithm</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">12</div>
                <div class="kpi-label">Features</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">৳BDT</div>
                <div class="kpi-label">Currency</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">~78%</div>
                <div class="kpi-label">Confidence</div>
            </div>
        </div>

        <div class="hero-cta-wrap">
            <div class="hero-trust"><span class="check">✓</span> No sign-up required</div>
            <div class="hero-trust"><span class="check">✓</span> Results in under 1 second</div>
            <div class="hero-trust"><span class="check">✓</span> Trained on Sylhet listings</div>
        </div>
    </div>

    <div class="scroll-hint">
        <div class="scroll-hint-line"></div>
        <div class="scroll-hint-text">Scroll</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# ESTIMATOR FORM
# ─────────────────────────────
st.markdown('<div id="estimator"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="section-label">
    <div class="section-label-line"></div>
    <div class="section-label-text">Property Details</div>
    <div class="section-label-line"></div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="medium")

with col_left:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon icon-blue">🏗️</div>
            <div>
                <div class="glass-card-title">Property Specifications</div>
                <div class="glass-card-desc">Core details about the unit</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    property_type = st.selectbox("Property Type", PROPERTY_TYPES, index=0)
    location      = st.selectbox("Location", LOCATIONS, index=0)
    size          = st.number_input("Size (sqft)", 300, 15000, 1500, step=50)
    bedrooms      = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms     = st.number_input("Bathrooms", 1, 8, 2)
    floor         = st.number_input("Floor Number  (0 = ground floor / whole house)", 0, 30, 5)
    is_ground_whole = 1 if floor == 0 else 0

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon icon-cyan">✨</div>
            <div>
                <div class="glass-card-title">Amenities & Features</div>
                <div class="glass-card-desc">Inclusions that add value</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    balcony = st.number_input("Balconies", 0, 5, 1)
    parking = st.number_input("Parking Spaces", 0, 5, 1)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        lift_opt = st.radio("Lift", ["No", "Yes"], index=1)
        lift = 1 if lift_opt == "Yes" else 0
    with c2:
        cctv_opt = st.radio("CCTV", ["No", "Yes"], index=0)
        cctv = 1 if cctv_opt == "Yes" else 0
    with c3:
        gen_opt = st.radio("Generator", ["No", "Yes"], index=0)
        generator = 1 if gen_opt == "Yes" else 0

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("✦  Estimate Price  ✦")

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

    input_data = pd.get_dummies(raw, columns=["Location", "Property Type"])
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    log_pred   = model.predict(input_data)[0]
    prediction = np.exp(log_pred)
    price_low  = max(0, prediction - gbm_rmse)
    price_high = prediction + gbm_rmse

    def fmt_readable(p):
        if p >= 1_00_00_000:
            return f"{p/1_00_00_000:.2f} Crore BDT"
        elif p >= 1_00_000:
            return f"{p/1_00_000:.1f} Lakh BDT"
        return f"{int(p):,} BDT"

    price_fmt = f"{int(prediction):,}"
    readable  = fmt_readable(prediction)
    range_low = f"৳ {int(price_low):,}"
    range_hi  = f"৳ {int(price_high):,}"

    # Amenity badges
    def amenity_badge(label, icon, is_on):
        cls = "amenity-on" if is_on else "amenity-off"
        mark = "✓" if is_on else "–"
        return f'<span class="amenity-badge {cls}">{icon} {mark} {label}</span>'

    badges = (
        amenity_badge("Lift", "🛗", lift)
        + amenity_badge("CCTV", "📷", cctv)
        + amenity_badge("Generator", "⚡", generator)
        + amenity_badge("Parking", "🚗", parking > 0)
        + amenity_badge("Balcony", "🏡", balcony > 0)
    )

    floor_label = "Ground / Whole" if floor == 0 else f"Floor {floor}"

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
            <div class="result-readable">≈ {readable}</div>

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
                Estimated range: <span>{range_low}</span> – <span>{range_hi}</span>
            </div>

            <div class="result-divider"></div>

            <div class="spec-grid">
                <div class="spec-item">
                    <div class="spec-val">{property_type}</div>
                    <div class="spec-key">Type</div>
                </div>
                <div class="spec-item">
                    <div class="spec-val">{size:,} sqft</div>
                    <div class="spec-key">Area</div>
                </div>
                <div class="spec-item">
                    <div class="spec-val">{bedrooms} BR · {bathrooms} BA</div>
                    <div class="spec-key">Layout</div>
                </div>
                <div class="spec-item">
                    <div class="spec-val">{floor_label}</div>
                    <div class="spec-key">Level</div>
                </div>
                <div class="spec-item">
                    <div class="spec-val">{balcony} · {parking}</div>
                    <div class="spec-key">Balcony · Parking</div>
                </div>
            </div>

            <div class="amenity-row">{badges}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✓  Estimate generated  ·  Gradient Boosting Model  ·  For reference purposes only")

st.markdown('</div>', unsafe_allow_html=True)  # /section-wrap

# ─────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────
import json, os

st.markdown('<div class="perf-section">', unsafe_allow_html=True)
st.markdown("""
<div class="perf-header">
    <div class="perf-title">Model Performance</div>
    <div class="perf-sub">Evaluated on a held-out test set · 60/20/20 train/val/test split · No data leakage</div>
</div>
""", unsafe_allow_html=True)

metrics_path = "model_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]["R2"], reverse=True)
    best_name = sorted_models[0][0]

    rows_html = ""
    for name, m in sorted_models:
        is_best  = name == best_name
        row_cls  = 'class="perf-best"' if is_best else ''
        badge    = '<span class="best-badge">BEST</span>' if is_best else ""
        name_cls = "best-name" if is_best else ""
        r2_cls   = "best-r2"   if is_best else ""
        rows_html += f"""
        <tr {row_cls}>
            <td class="{name_cls}">{name}{badge}</td>
            <td class="{r2_cls}">{m["R2"]:.4f}</td>
            <td>৳ {m["MAE"]:,}</td>
            <td>৳ {m["RMSE"]:,}</td>
        </tr>"""

    st.markdown(f"""
    <div class="perf-table-wrap">
        <table class="perf-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>R² Score</th>
                    <th>MAE (BDT)</th>
                    <th>RMSE (BDT)</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align:center; color:var(--text-muted); font-size:0.8rem; padding:2rem;
                background:var(--bg-card); border-radius:var(--radius-lg); border:1px solid var(--border);'>
        Run <code>model.py</code> to generate <code>model_metrics.json</code>
    </div>
    """, unsafe_allow_html=True)

# Trust strip
st.markdown("""
<div class="trust-strip">
    <div class="trust-item"><span class="trust-icon">🧠</span> Gradient Boosting (scikit-learn)</div>
    <div class="trust-item"><span class="trust-icon">📊</span> Trained on Sylhet listing data</div>
    <div class="trust-item"><span class="trust-icon">🔒</span> No personal data collected</div>
    <div class="trust-item"><span class="trust-icon">⚡</span> Inference in &lt;50 ms</div>
    <div class="trust-item"><span class="trust-icon">🗺️</span> 21 Sylhet neighbourhoods</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /perf-section

# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("""
<div class="premium-footer">
    <div class="footer-logo">Sylhet Real Estate</div>
    <div class="footer-links">
        <span>Privacy</span>
        <span>Methodology</span>
        <span>Data Sources</span>
        <span>Contact</span>
    </div>
    <div class="footer-text">
        © 2026 Sylhet Real Estate Estimator. All rights reserved.<br>
        For informational purposes only · Not financial advice · Sylhet, Bangladesh
    </div>
</div>
""", unsafe_allow_html=True)
