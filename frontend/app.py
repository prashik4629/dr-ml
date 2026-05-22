import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

try:
    HEART_API_URL = st.secrets["HEART_API_URL"]
    HEART_API_KEY = st.secrets["HEART_API_KEY"]
    DIABETES_API_URL = st.secrets["DIABETES_API_URL"]
    DIABETES_API_KEY = st.secrets["DIABETES_API_KEY"]
except:
    HEART_API_URL = os.getenv("HEART_API_URL", "http://127.0.0.1:8000/api/v1/predict")
    HEART_API_KEY = os.getenv("HEART_API_KEY", "medbuddy-dev-key-change-in-production")
    DIABETES_API_URL = os.getenv("DIABETES_API_URL", "http://127.0.0.1:8001/api/v1/predict")
    DIABETES_API_KEY = os.getenv("DIABETES_API_KEY", "drml-dev-key-change-in-production")

st.set_page_config(
    page_title="Dr.ML — Multi Disease Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════
# MEGA CSS — BIOPUNK / MEDICAL SCI-FI THEME
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Orbitron:wght@400;600;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:     #020810;
    --bg-card:     rgba(0, 255, 200, 0.03);
    --bg-card2:    rgba(0, 180, 255, 0.04);
    --cyan:        #00ffe7;
    --cyan-dim:    #00b4a0;
    --blue:        #0af;
    --blue-dim:    #0080b3;
    --red:         #ff4d6d;
    --orange:      #ff9f43;
    --green:       #00f5a0;
    --text-main:   #d0f0ea;
    --text-sub:    #3d7a70;
    --text-muted:  #1e4a44;
    --border:      rgba(0,255,200,0.12);
    --border-glow: rgba(0,255,200,0.35);
    --font-display: 'Orbitron', monospace;
    --font-body:    'Rajdhani', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-main) !important;
}

.stApp {
    background: var(--bg-deep) !important;
    min-height: 100vh;
}

.main .block-container {
    background: transparent !important;
    padding: 0 2.5rem 4rem !important;
    max-width: 1180px !important;
}

/* ── Animated Grid Background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,200,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 8s ease-in-out infinite;
}

@keyframes gridPulse {
    0%, 100% { opacity: 0.5; }
    50%       { opacity: 1; }
}

/* ── Radial glow orbs ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 20% 10%, rgba(0,255,200,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 50% 30% at 80% 80%, rgba(0,170,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(0,100,80,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── HERO ── */
.hero-wrap {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 3rem 0 0.5rem;
    animation: heroFade 1.2s ease forwards;
}

@keyframes heroFade {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: var(--cyan-dim);
    border: 1px solid rgba(0,255,200,0.2);
    padding: 0.3rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
    background: rgba(0,255,200,0.04);
    animation: badgePulse 3s ease-in-out infinite;
}

@keyframes badgePulse {
    0%, 100% { border-color: rgba(0,255,200,0.2); box-shadow: none; }
    50%       { border-color: rgba(0,255,200,0.5); box-shadow: 0 0 15px rgba(0,255,200,0.15); }
}

.hero-title {
    font-family: var(--font-display) !important;
    font-size: 3.8rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #00ffe7 0%, #00aaff 50%, #00ffe7 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite, heroFade 1s ease forwards;
    text-shadow: none;
    margin: 0;
    line-height: 1.1;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-sub {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-sub) !important;
    letter-spacing: 0.2em;
    margin-top: 0.6rem;
    text-transform: uppercase;
}

/* ── Scan line divider ── */
.scan-divider {
    position: relative;
    height: 2px;
    margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent 0%, var(--cyan) 50%, transparent 100%);
    overflow: visible;
}
.scan-divider::after {
    content: '';
    position: absolute;
    left: -100%;
    top: 0;
    width: 40%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,200,0.8), transparent);
    animation: scanMove 3s linear infinite;
}
@keyframes scanMove {
    0%   { left: -40%; }
    100% { left: 140%; }
}

/* ── Section Labels ── */
.section-label {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.25em !important;
    color: var(--cyan-dim) !important;
    text-transform: uppercase !important;
    margin-bottom: 0.8rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '▸';
    color: var(--cyan);
    font-size: 0.7rem;
}

/* ── Glass Panel ── */
.glass-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
    z-index: 1;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
    opacity: 0.6;
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 25px rgba(0,255,200,0.08), inset 0 0 30px rgba(0,255,200,0.02);
}

/* ── Corner brackets ── */
.bracket-card {
    position: relative;
    padding: 1.8rem;
    background: rgba(0,255,200,0.02);
    border-radius: 4px;
    margin-bottom: 1rem;
    animation: cardIn 0.5s ease forwards;
}
@keyframes cardIn {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0); }
}
.bracket-card::before,
.bracket-card::after {
    content: '';
    position: absolute;
    width: 18px; height: 18px;
    border-color: var(--cyan);
    border-style: solid;
}
.bracket-card::before {
    top: 0; left: 0;
    border-width: 2px 0 0 2px;
    border-radius: 2px 0 0 0;
}
.bracket-card::after {
    bottom: 0; right: 0;
    border-width: 0 2px 2px 0;
    border-radius: 0 0 2px 0;
}

/* ── Column headers ── */
.col-header {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,255,200,0.1);
}

/* ── Streamlit Inputs ── */
.stNumberInput input,
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input {
    background: rgba(0,255,200,0.04) !important;
    border: 1px solid rgba(0,255,200,0.15) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,255,200,0.15) !important;
    outline: none !important;
}
.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(0,255,200,0.4) !important;
}

/* Label text */
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    color: var(--text-sub) !important;
    text-transform: uppercase !important;
}

/* ── Radio Buttons — pill/tab style ── */
div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
}
div[role="radiogroup"] label {
    background: rgba(0,255,200,0.04) !important;
    border: 1px solid rgba(0,255,200,0.18) !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.2rem !important;
    font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
    color: #3d7a70 !important;
    letter-spacing: 0.08em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
div[role="radiogroup"] label:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 14px rgba(0,255,200,0.15) !important;
    background: rgba(0,255,200,0.07) !important;
}
div[role="radiogroup"] label:has(input:checked) {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: rgba(0,255,200,0.1) !important;
    box-shadow: 0 0 18px rgba(0,255,200,0.18) !important;
}
/* Hide the actual radio circle dot */
div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}
div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
    margin: 0 !important;
}

/* ── Submit Button ── */
.stForm [data-testid="stFormSubmitButton"] button,
.stButton > button {
    position: relative;
    background: transparent !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-family: var(--font-display) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    overflow: hidden;
    transition: all 0.3s ease !important;
    cursor: pointer;
}
.stForm [data-testid="stFormSubmitButton"] button::before,
.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,255,200,0.12) 0%, rgba(0,170,255,0.12) 100%);
    opacity: 0;
    transition: opacity 0.3s;
}
.stForm [data-testid="stFormSubmitButton"] button:hover::before,
.stButton > button:hover::before { opacity: 1; }
.stForm [data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
    box-shadow: 0 0 25px rgba(0,255,200,0.25), 0 0 60px rgba(0,255,200,0.08) !important;
    transform: translateY(-1px) !important;
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    animation: badgeAppear 0.5s ease forwards;
}
@keyframes badgeAppear {
    from { opacity:0; transform: scale(0.85); }
    to   { opacity:1; transform: scale(1); }
}
.risk-high {
    background: rgba(255,77,109,0.1);
    border: 1px solid rgba(255,77,109,0.5);
    color: #ff4d6d;
    box-shadow: 0 0 20px rgba(255,77,109,0.15), inset 0 0 20px rgba(255,77,109,0.05);
    animation: redPulse 2s ease-in-out infinite, badgeAppear 0.5s ease forwards;
}
@keyframes redPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255,77,109,0.15); }
    50%       { box-shadow: 0 0 35px rgba(255,77,109,0.35); }
}
.risk-moderate {
    background: rgba(255,159,67,0.1);
    border: 1px solid rgba(255,159,67,0.5);
    color: #ff9f43;
    box-shadow: 0 0 20px rgba(255,159,67,0.15);
}
.risk-low {
    background: rgba(0,245,160,0.08);
    border: 1px solid rgba(0,245,160,0.4);
    color: #00f5a0;
    box-shadow: 0 0 20px rgba(0,245,160,0.12);
}

/* ── Result Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    position: relative;
    background: rgba(0,255,200,0.03);
    border: 1px solid rgba(0,255,200,0.12);
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    overflow: hidden;
    animation: cardIn 0.6s ease forwards;
    transition: border-color 0.3s;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.4;
}
.metric-card:hover {
    border-color: rgba(0,255,200,0.3);
}
.metric-label {
    font-family: var(--font-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-sub);
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    display: block;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-main);
    letter-spacing: 0.05em;
}
.metric-prob {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--cyan);
    letter-spacing: 0.05em;
    text-shadow: 0 0 30px rgba(0,255,200,0.4);
    animation: probGlow 2s ease-in-out infinite;
}
@keyframes probGlow {
    0%, 100% { text-shadow: 0 0 20px rgba(0,255,200,0.3); }
    50%       { text-shadow: 0 0 40px rgba(0,255,200,0.6), 0 0 80px rgba(0,255,200,0.2); }
}

/* ── History Card ── */
.history-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(0,255,200,0.02);
    border: 1px solid rgba(0,255,200,0.08);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: all 0.25s ease;
    animation: cardIn 0.4s ease forwards;
}
.history-item:hover {
    border-color: rgba(0,255,200,0.25);
    background: rgba(0,255,200,0.04);
    transform: translateX(4px);
}

/* ── About Tech Stack ── */
.tech-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(0,255,200,0.06);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-sub);
}
.tech-item:last-child { border-bottom: none; }
.tech-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
    flex-shrink: 0;
    animation: dotPulse 2s ease-in-out infinite;
}
@keyframes dotPulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.4); }
}

/* ── Spinner ── */
.stSpinner > div { border-color: var(--cyan) transparent transparent transparent !important; }

/* ── Divider ── */
hr { border-color: rgba(0,255,200,0.08) !important; margin: 1.8rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #020810; }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 4px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Stagger animation for form rows ── */
.stForm { animation: cardIn 0.5s ease forwards; }
[data-testid="column"]:nth-child(1) { animation: cardIn 0.4s ease forwards; }
[data-testid="column"]:nth-child(2) { animation: cardIn 0.55s 0.1s ease both; }
[data-testid="column"]:nth-child(3) { animation: cardIn 0.55s 0.2s ease both; }

</style>
""", unsafe_allow_html=True)

# ── Animated Particle Canvas — injected via components.html into parent DOM ──
components.html("""
<script>
(function(){
  // Inject a canvas into the PARENT window (Streamlit host page)
  var parent = window.parent || window;
  var doc = parent.document;

  // Remove old canvas if rerun
  var old = doc.getElementById('drml-particles');
  if(old) old.remove();

  var wrapper = doc.createElement('div');
  wrapper.id = 'drml-particles';
  wrapper.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;';

  var c = doc.createElement('canvas');
  c.style.cssText = 'width:100%;height:100%;display:block;';
  wrapper.appendChild(c);
  doc.body.appendChild(wrapper);

  var ctx = c.getContext('2d');
  var W, H, pts = [];

  function resize(){
    W = c.width  = parent.innerWidth;
    H = c.height = parent.innerHeight;
  }
  resize();
  parent.addEventListener('resize', resize);

  for(var i=0;i<60;i++){
    pts.push({
      x: Math.random()*W, y: Math.random()*H,
      vx: (Math.random()-.5)*0.3, vy: (Math.random()-.5)*0.3,
      r: Math.random()*1.5+0.4
    });
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<pts.length;i++){
      for(var j=i+1;j<pts.length;j++){
        var dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y;
        var d=Math.sqrt(dx*dx+dy*dy);
        if(d<170){
          ctx.beginPath();
          ctx.strokeStyle='rgba(0,255,200,'+(((1-d/170)*0.13))+')';
          ctx.lineWidth=0.7;
          ctx.moveTo(pts[i].x,pts[i].y);
          ctx.lineTo(pts[j].x,pts[j].y);
          ctx.stroke();
        }
      }
      pts[i].x+=pts[i].vx; pts[i].y+=pts[i].vy;
      if(pts[i].x<0||pts[i].x>W) pts[i].vx*=-1;
      if(pts[i].y<0||pts[i].y>H) pts[i].vy*=-1;
      ctx.beginPath();
      ctx.arc(pts[i].x,pts[i].y,pts[i].r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,255,200,0.55)';
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", height=0)

# ══════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════
# HERO
# ══════════════════════════════════════════
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-badge'>◈ MEDICAL AI DIAGNOSTICS PLATFORM v2.1 ◈</div>
    <div class='hero-title'>DR.ML</div>
    <div class='hero-sub'>[ MULTI-DISEASE PREDICTION SYSTEM — POWERED BY MACHINE LEARNING ]</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════
nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])
with nav_c2:
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.markdown('<div class="section-label" style="justify-content:center;margin-bottom:0.4rem;">Disease Module</div>', unsafe_allow_html=True)
        disease = st.radio(
            "disease",
            ["🫀  Heart Disease", "🩺  Diabetes"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with nav_col2:
        st.markdown('<div class="section-label" style="justify-content:center;margin-bottom:0.4rem;">Interface</div>', unsafe_allow_html=True)
        page = st.radio(
            "page",
            ["⚡  Prediction", "📡  History", "🔬  About"],
            horizontal=True,
            label_visibility="collapsed",
        )

st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# RESULTS RENDERER
# ══════════════════════════════════════════
def show_results(probability, diagnosis, risk_level, label):
    risk_class = {
        "High Risk": "risk-high",
        "Moderate Risk": "risk-moderate",
        "Low Risk": "risk-low"
    }.get(risk_level, "risk-low")

    risk_icon = {
        "High Risk": "🔴",
        "Moderate Risk": "🟠",
        "Low Risk": "🟢"
    }.get(risk_level, "⚪")

    st.markdown(f"""
    <div class='metric-grid'>
        <div class='metric-card'>
            <span class='metric-label'>Diagnosis</span>
            <div class='metric-value'>{diagnosis}</div>
        </div>
        <div class='metric-card'>
            <span class='metric-label'>Risk Level</span>
            <span class='risk-badge {risk_class}'>{risk_icon} {risk_level}</span>
        </div>
        <div class='metric-card'>
            <span class='metric-label'>Probability</span>
            <div class='metric-prob'>{probability:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plotly Gauge
    _, g_col, _ = st.columns([1, 2.2, 1])
    with g_col:
        gauge_color = (
            "#ff4d6d" if probability >= 0.75 else
            "#ff9f43" if probability >= 0.45 else
            "#00f5a0"
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={
                "suffix": "%",
                "font": {"size": 52, "color": gauge_color, "family": "Orbitron"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickfont": {"color": "#3d7a70", "size": 10, "family": "JetBrains Mono"},
                    "tickcolor": "rgba(0,255,200,0.2)",
                    "linecolor": "rgba(0,255,200,0.1)",
                },
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45],   "color": "rgba(0,245,160,0.06)"},
                    {"range": [45, 75],  "color": "rgba(255,159,67,0.06)"},
                    {"range": [75, 100], "color": "rgba(255,77,109,0.06)"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 3},
                    "thickness": 0.85,
                    "value": round(probability * 100, 1),
                },
            },
            title={
                "text": f"<span style='font-family:JetBrains Mono;font-size:11px;letter-spacing:3px;color:#3d7a70;text-transform:uppercase;'>{label} RISK INDEX</span>",
                "font": {"family": "JetBrains Mono"},
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Orbitron"},
            height=320,
            margin=dict(t=50, b=10, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Interpretation bar
    interp = (
        "⚠️  High probability detected. Immediate clinical consultation recommended."
        if probability >= 0.75 else
        "⚡  Moderate indicators present. Lifestyle modifications and monitoring advised."
        if probability >= 0.45 else
        "✅  Low risk profile. Maintain current health practices and routine checkups."
    )
    interp_color = (
        "rgba(255,77,109,0.08)" if probability >= 0.75 else
        "rgba(255,159,67,0.08)" if probability >= 0.45 else
        "rgba(0,245,160,0.06)"
    )
    border_color = (
        "rgba(255,77,109,0.3)" if probability >= 0.75 else
        "rgba(255,159,67,0.3)" if probability >= 0.45 else
        "rgba(0,245,160,0.25)"
    )
    st.markdown(f"""
    <div style="
        background: {interp_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #a0c0bc;
        letter-spacing: 0.03em;
        line-height: 1.6;
        margin-top: 0.5rem;
    ">
        <span style='color:#3d7a70;letter-spacing:0.2em;font-size:0.6rem;'>CLINICAL INTERPRETATION //</span><br>
        {interp}
        <br><br>
        <span style='color:#1e4a44;font-size:0.65rem;'>⚕ FOR EDUCATIONAL PURPOSES ONLY. NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE.</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════
if "Prediction" in page:

    if "Heart" in disease:
        st.markdown('<div class="section-label">Patient Data Input — Cardiac Module</div>', unsafe_allow_html=True)

        with st.form("heart_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="col-header">⬡ Demographics</div>', unsafe_allow_html=True)
                age = st.number_input("Age (yrs)", min_value=1, max_value=120, value=52)
                sex = st.selectbox("Biological Sex", options=[0, 1],
                                   format_func=lambda x: "Male" if x == 1 else "Female")
                cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                  format_func=lambda x: {
                                      0: "0 — Typical Angina",
                                      1: "1 — Atypical Angina",
                                      2: "2 — Non-Anginal Pain",
                                      3: "3 — Asymptomatic"}[x])
                fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")

            with col2:
                st.markdown('<div class="col-header">⬡ Vitals</div>', unsafe_allow_html=True)
                trestbps = st.number_input("Resting BP (mmHg)", min_value=80, max_value=250, value=125)
                chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=212)
                thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=250, value=168)
                oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            with col3:
                st.markdown('<div class="col-header">⬡ ECG & Tests</div>', unsafe_allow_html=True)
                restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                       format_func=lambda x: {
                                           0: "0 — Normal",
                                           1: "1 — ST-T Abnormality",
                                           2: "2 — LV Hypertrophy"}[x])
                exang = st.selectbox("Exercise Angina", options=[0, 1],
                                     format_func=lambda x: "Yes" if x == 1 else "No")
                slope = st.selectbox("ST Slope", options=[0, 1, 2],
                                     format_func=lambda x: {
                                         0: "0 — Upsloping",
                                         1: "1 — Flat",
                                         2: "2 — Downsloping"}[x])
                ca = st.number_input("Major Vessels (0–4)", min_value=0, max_value=4, value=0)
                thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                    format_func=lambda x: {
                                        0: "0 — Normal",
                                        1: "1 — Fixed Defect",
                                        2: "2 — Reversible Defect",
                                        3: "3 — Unknown"}[x])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡  INITIATE CARDIAC RISK ANALYSIS")

        if submitted:
            payload = {
                "age": age, "sex": sex, "cp": cp,
                "trestbps": trestbps, "chol": chol, "fbs": fbs,
                "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }
            with st.spinner("Processing cardiac biomarkers..."):
                try:
                    r = requests.post(
                        HEART_API_URL, json=payload,
                        headers={"X-API-Key": HEART_API_KEY},
                        timeout=60,
                    )
                    res = r.json()
                    if r.status_code != 200:
                        st.error(f"API Error [{r.status_code}]: {res.get('detail')}")
                        st.stop()
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "disease": "Heart Disease",
                        "probability": res["probability"],
                        "risk_level": res["risk_level"],
                        "diagnosis": res["diagnosis"],
                    })
                    st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Analysis Output — Cardiac Risk</div>', unsafe_allow_html=True)
                    show_results(res["probability"], res["diagnosis"], res["risk_level"], "Heart Disease")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️  Connection to inference API failed. Retry in 30 seconds.")
                except Exception as e:
                    st.error(f"Runtime Error: {e}")

    else:
        st.markdown('<div class="section-label">Patient Data Input — Metabolic Module</div>', unsafe_allow_html=True)

        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="col-header">⬡ Personal Details</div>', unsafe_allow_html=True)
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
                age = st.number_input("Age (yrs)", min_value=1, max_value=120, value=35)
                bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=28.5, step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.45, step=0.01)

            with col2:
                st.markdown('<div class="col-header">⬡ Clinical Measurements</div>', unsafe_allow_html=True)
                glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=400, value=120)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=250, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=25)
                insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡  INITIATE METABOLIC RISK ANALYSIS")

        if submitted:
            payload = {
                "Pregnancies": pregnancies, "Glucose": glucose,
                "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
                "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age,
            }
            with st.spinner("Processing metabolic biomarkers..."):
                try:
                    r = requests.post(
                        DIABETES_API_URL, json=payload,
                        headers={"X-API-Key": DIABETES_API_KEY},
                        timeout=60,
                    )
                    res = r.json()
                    if r.status_code != 200:
                        st.error(f"API Error [{r.status_code}]: {res.get('detail')}")
                        st.stop()
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "disease": "Diabetes",
                        "probability": res["probability"],
                        "risk_level": res["risk_level"],
                        "diagnosis": res["diagnosis"],
                    })
                    st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Analysis Output — Metabolic Risk</div>', unsafe_allow_html=True)
                    show_results(res["probability"], res["diagnosis"], res["risk_level"], "Diabetes")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️  Connection to inference API failed. Retry in 30 seconds.")
                except Exception as e:
                    st.error(f"Runtime Error: {e}")


# ══════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════
elif "History" in page:
    st.markdown("""
    <div style='margin-bottom:1.5rem;animation:heroFade 0.6s ease forwards;'>
        <div class='hero-title' style='font-size:2.2rem;'>PREDICTION LOG</div>
        <div class='hero-sub'>[ SESSION DIAGNOSTIC HISTORY ]</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3.5rem;'>
            <div style='font-size:2.5rem;margin-bottom:0.8rem;'>📡</div>
            <div style='font-family:"JetBrains Mono",monospace;color:#1e4a44;font-size:0.75rem;letter-spacing:0.2em;'>
                NO DIAGNOSTIC DATA IN BUFFER
            </div>
            <div style='font-family:"JetBrains Mono",monospace;color:#1e4a44;font-size:0.6rem;letter-spacing:0.1em;margin-top:0.3rem;'>
                RUN A PREDICTION TO POPULATE THIS LOG
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, row in enumerate(reversed(st.session_state.history)):
            risk_class = {
                "High Risk": "risk-high",
                "Moderate Risk": "risk-moderate",
                "Low Risk": "risk-low",
            }.get(row["risk_level"], "risk-low")
            icon = "🫀" if row["disease"] == "Heart Disease" else "🩺"
            idx = len(st.session_state.history) - i
            st.markdown(f"""
            <div class='history-item'>
                <div style='display:flex;gap:1.2rem;align-items:center;'>
                    <div style='font-size:1.3rem;'>{icon}</div>
                    <div>
                        <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;color:#1e4a44;letter-spacing:0.15em;'>
                            #{idx:02d} — {row["time"]}
                        </div>
                        <div style='font-family:"Rajdhani",sans-serif;font-size:0.95rem;color:#a0c0bc;margin-top:0.15rem;'>
                            {row["disease"]}
                        </div>
                    </div>
                </div>
                <div style='display:flex;align-items:center;gap:1.2rem;'>
                    <div style='font-family:"Orbitron",monospace;font-size:1.1rem;font-weight:700;color:#00ffe7;'>
                        {row["probability"]:.1%}
                    </div>
                    <span class='risk-badge {risk_class}'>{row["risk_level"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(st.session_state.history) > 1:
            st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)
            df_h = pd.DataFrame(st.session_state.history)
            risk_counts = df_h["risk_level"].value_counts()
            color_map = {
                "High Risk": "#ff4d6d",
                "Moderate Risk": "#ff9f43",
                "Low Risk": "#00f5a0",
            }
            fig_pie = go.Figure(go.Pie(
                labels=risk_counts.index.tolist(),
                values=risk_counts.values.tolist(),
                hole=0.65,
                marker_colors=[color_map.get(l, "#3d7a70") for l in risk_counts.index],
                textfont={"color": "#d0f0ea", "family": "JetBrains Mono", "size": 11},
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "JetBrains Mono", "color": "#3d7a70"},
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(
                    font={"color": "#3d7a70", "size": 10, "family": "JetBrains Mono"},
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            _, chart_col, _ = st.columns([1, 2, 1])
            with chart_col:
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  CLEAR SESSION LOG"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div style='margin-bottom:1.5rem;animation:heroFade 0.6s ease forwards;'>
        <div class='hero-title' style='font-size:2.2rem;'>SYSTEM MANIFEST</div>
        <div class='hero-sub'>[ PLATFORM OVERVIEW & TECHNICAL SPECIFICATIONS ]</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Disease Modules</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='bracket-card'>
            <div class='tech-item'><span class='tech-dot'></span>🫀 Heart Disease — Cleveland Heart Dataset</div>
            <div class='tech-item'><span class='tech-dot'></span>🩺 Diabetes — Pima Indians Dataset</div>
            <div class='tech-item'><span class='tech-dot' style='background:#3d7a70;box-shadow:none;opacity:0.4;'></span>🔬 Cancer Module — Coming Soon</div>
            <div class='tech-item'><span class='tech-dot' style='background:#3d7a70;box-shadow:none;opacity:0.4;'></span>🧠 Neurological Module — In Development</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='bracket-card'>
            <div class='tech-item'><span class='tech-dot'></span>Frontend &nbsp;→&nbsp; Streamlit + Plotly</div>
            <div class='tech-item'><span class='tech-dot'></span>Backend &nbsp;&nbsp;→&nbsp; FastAPI + Uvicorn</div>
            <div class='tech-item'><span class='tech-dot'></span>ML Engine →&nbsp; XGBoost + Scikit-learn</div>
            <div class='tech-item'><span class='tech-dot'></span>Deploy &nbsp;&nbsp;→&nbsp; Render + Streamlit Cloud</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Model Performance</div>', unsafe_allow_html=True)
    perf_c1, perf_c2, perf_c3, perf_c4 = st.columns(4)
    for col, label, val, color in [
        (perf_c1, "Heart Accuracy", "87.2%", "#00ffe7"),
        (perf_c2, "Heart AUC-ROC",  "0.924",  "#00ffe7"),
        (perf_c3, "Diab. Accuracy", "83.5%", "#00aaff"),
        (perf_c4, "Diab. AUC-ROC",  "0.891",  "#00aaff"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='height:90px;'>
                <span class='metric-label'>{label}</span>
                <div style='font-family:"Orbitron",monospace;font-size:1.4rem;font-weight:700;color:{color};
                    text-shadow:0 0 20px {color}66;'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Legal Disclaimer</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;color:#1e4a44;line-height:2;letter-spacing:0.03em;'>
            This platform is developed for <span style='color:#3d7a70;'>educational and research purposes only</span>.<br>
            Predictions generated by this system do not constitute medical advice, diagnosis, or treatment.<br>
            Always consult a <span style='color:#3d7a70;'>qualified healthcare professional</span> for medical decisions.<br>
            <br>
            <span style='color:#0d2e2a;font-size:0.6rem;'>© Dr.ML Platform — Built with ❤️ using FastAPI + Streamlit + XGBoost</span>
        </div>
    </div>
    """, unsafe_allow_html=True)