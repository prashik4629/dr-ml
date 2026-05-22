import requests
import streamlit as st
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
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #060d1f; }
    .main .block-container { background: #060d1f; padding: 2rem 3rem; max-width: 1200px; }

    .hero-wrap {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed 0%, #76e4f7 50%, #b794f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: #4a5568;
        margin-top: 0.4rem;
    }

    /* Disease toggle pills */
    .disease-toggle {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.5rem 0 2rem;
    }

    /* Section label */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        color: #63b3ed;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Risk badges */
    .risk-high {
        background: rgba(245,101,101,0.15);
        border: 1px solid rgba(245,101,101,0.4);
        color: #fc8181;
        padding: 0.35rem 1.1rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .risk-moderate {
        background: rgba(237,137,54,0.15);
        border: 1px solid rgba(237,137,54,0.4);
        color: #f6ad55;
        padding: 0.35rem 1.1rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .risk-low {
        background: rgba(72,187,120,0.15);
        border: 1px solid rgba(72,187,120,0.4);
        color: #68d391;
        padding: 0.35rem 1.1rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Result metric card */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 14px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .metric-label {
        font-size: 0.65rem;
        color: #718096;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .metric-value {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .metric-prob {
        font-size: 1.9rem;
        font-weight: 700;
        color: #63b3ed;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #44337a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 0.95rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Radio buttons styling */
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    hr { border-color: rgba(99,179,237,0.08); margin: 1.5rem 0; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #060d1f; }
    ::-webkit-scrollbar-thumb { background: #2b6cb0; border-radius: 4px; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ── Hero ──
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-title'>🏥 Dr.ML</div>
    <div class='hero-sub'>AI-powered Multi Disease Prediction Platform</div>
</div>
""", unsafe_allow_html=True)

# ── Disease Selection ──
st.markdown('<div class="section-label" style="text-align:center;">Select Disease</div>',
            unsafe_allow_html=True)

d1, d2, d3 = st.columns([1, 2, 1])
with d2:
    disease = st.radio(
        "disease",
        ["🫀  Heart Disease", "🩺  Diabetes"],
        horizontal=True,
        label_visibility="collapsed",
    )

# ── Page Navigation ──
st.markdown('<div class="section-label" style="text-align:center;margin-top:0.5rem;">Navigation</div>',
            unsafe_allow_html=True)

n1, n2, n3 = st.columns([1, 2, 1])
with n2:
    page = st.radio(
        "page",
        ["Prediction", "History", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("---")


def show_results(probability, diagnosis, risk_level, label):
    risk_class = {
        "High Risk": "risk-high",
        "Moderate Risk": "risk-moderate",
        "Low Risk": "risk-low"
    }[risk_level]

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Diagnosis</div>
            <div class='metric-value'>{diagnosis}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Risk Level</div>
            <span class='{risk_class}'>{risk_level}</span>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Probability</div>
            <div class='metric-prob'>{probability:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _, g_col, _ = st.columns([1, 2, 1])
    with g_col:
        gauge_color = (
            "#fc8181" if probability >= 0.75 else
            "#f6ad55" if probability >= 0.45 else
            "#68d391"
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 48, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#718096", "size": 11}},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 45],   "color": "rgba(72,187,120,0.08)"},
                    {"range": [45, 75],  "color": "rgba(237,137,54,0.08)"},
                    {"range": [75, 100], "color": "rgba(245,101,101,0.08)"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 3},
                    "thickness": 0.82,
                    "value": round(probability * 100, 1),
                },
            },
            title={"text": f"{label} Risk Score",
                   "font": {"color": "#718096", "size": 13}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
            height=300,
            margin=dict(t=40, b=0, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════
if page == "Prediction":

    if "Heart" in disease:
        st.markdown('<div class="section-label">Patient Information — Heart Disease</div>',
                    unsafe_allow_html=True)

        with st.form("heart_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Demographics**")
                age = st.number_input("Age", min_value=1, max_value=120, value=52)
                sex = st.selectbox("Sex", options=[0, 1],
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
                st.markdown("**Vitals**")
                trestbps = st.number_input("Resting Blood Pressure (mmHg)",
                                           min_value=80, max_value=250, value=125)
                chol = st.number_input("Cholesterol (mg/dl)",
                                       min_value=100, max_value=600, value=212)
                thalach = st.number_input("Max Heart Rate",
                                          min_value=60, max_value=250, value=168)
                oldpeak = st.number_input("ST Depression",
                                          min_value=0.0, max_value=10.0,
                                          value=1.0, step=0.1)

            with col3:
                st.markdown("**ECG & Tests**")
                restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                       format_func=lambda x: {
                                           0: "0 — Normal",
                                           1: "1 — ST-T Abnormality",
                                           2: "2 — LV Hypertrophy"}[x])
                exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                     format_func=lambda x: "Yes" if x == 1 else "No")
                slope = st.selectbox("ST Slope", options=[0, 1, 2],
                                     format_func=lambda x: {
                                         0: "0 — Upsloping",
                                         1: "1 — Flat",
                                         2: "2 — Downsloping"}[x])
                ca = st.number_input("Major Vessels (0-4)",
                                     min_value=0, max_value=4, value=0)
                thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                    format_func=lambda x: {
                                        0: "0 — Normal",
                                        1: "1 — Fixed Defect",
                                        2: "2 — Reversible Defect",
                                        3: "3 — Unknown"}[x])

            submitted = st.form_submit_button("Run Heart Risk Analysis")

        if submitted:
            payload = {
                "age": age, "sex": sex, "cp": cp,
                "trestbps": trestbps, "chol": chol, "fbs": fbs,
                "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }
            with st.spinner("Analyzing heart risk..."):
                try:
                    r = requests.post(
                        HEART_API_URL, json=payload,
                        headers={"X-API-Key": HEART_API_KEY},
                        timeout=60,
                    )
                    res = r.json()
                    if r.status_code != 200:
                        st.error(f"API Error: {res.get('detail')}")
                        st.stop()
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "disease": "Heart Disease",
                        "probability": res["probability"],
                        "risk_level": res["risk_level"],
                        "diagnosis": res["diagnosis"],
                    })
                    st.markdown("---")
                    st.markdown('<div class="section-label">Analysis Results</div>',
                                unsafe_allow_html=True)
                    show_results(res["probability"], res["diagnosis"],
                                 res["risk_level"], "Heart Disease")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Please try again in 30 seconds.")
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        st.markdown('<div class="section-label">Patient Information — Diabetes</div>',
                    unsafe_allow_html=True)

        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Personal Details**")
                pregnancies = st.number_input("Pregnancies",
                                              min_value=0, max_value=20, value=2)
                age = st.number_input("Age", min_value=1, max_value=120, value=35)
                bmi = st.number_input("BMI (kg/m²)",
                                      min_value=0.0, max_value=70.0,
                                      value=28.5, step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function",
                                      min_value=0.0, max_value=3.0,
                                      value=0.45, step=0.01)

            with col2:
                st.markdown("**Clinical Measurements**")
                glucose = st.number_input("Glucose (mg/dL)",
                                          min_value=0, max_value=400, value=120)
                blood_pressure = st.number_input("Blood Pressure (mmHg)",
                                                 min_value=0, max_value=250, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)",
                                                 min_value=0, max_value=100, value=25)
                insulin = st.number_input("Insulin (μU/mL)",
                                          min_value=0, max_value=900, value=80)

            submitted = st.form_submit_button("Run Diabetes Analysis")

        if submitted:
            payload = {
                "Pregnancies": pregnancies, "Glucose": glucose,
                "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
                "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age,
            }
            with st.spinner("Analyzing diabetes risk..."):
                try:
                    r = requests.post(
                        DIABETES_API_URL, json=payload,
                        headers={"X-API-Key": DIABETES_API_KEY},
                        timeout=60,
                    )
                    res = r.json()
                    if r.status_code != 200:
                        st.error(f"API Error: {res.get('detail')}")
                        st.stop()
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "disease": "Diabetes",
                        "probability": res["probability"],
                        "risk_level": res["risk_level"],
                        "diagnosis": res["diagnosis"],
                    })
                    st.markdown("---")
                    st.markdown('<div class="section-label">Analysis Results</div>',
                                unsafe_allow_html=True)
                    show_results(res["probability"], res["diagnosis"],
                                 res["risk_level"], "Diabetes")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Please try again in 30 seconds.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════
elif page == "History":
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div class='hero-title' style='font-size:2rem;'>Prediction History</div>
        <div class='hero-sub'>All predictions made in this session</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3rem;'>
            <div style='font-size:2rem;'>📋</div>
            <div style='color:#4a5568;margin-top:0.5rem;'>No predictions yet.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_h = pd.DataFrame(st.session_state.history)
        for i, row in df_h.iterrows():
            risk_class = {
                "High Risk": "risk-high",
                "Moderate Risk": "risk-moderate",
                "Low Risk": "risk-low",
            }[row["risk_level"]]
            icon = "🫀" if row["disease"] == "Heart Disease" else "🩺"
            st.markdown(f"""
            <div class='glass-card' style='display:flex;justify-content:space-between;align-items:center;'>
                <div style='display:flex;gap:1.5rem;align-items:center;'>
                    <div style='font-size:1.4rem;'>{icon}</div>
                    <div style='color:#4a5568;font-size:0.78rem;'>{row["time"]}</div>
                    <div style='color:#a0aec0;'>{row["disease"]}</div>
                    <div style='color:#63b3ed;font-weight:600;'>{row["probability"]:.1%}</div>
                </div>
                <span class='{risk_class}'>{row["risk_level"]}</span>
            </div>
            """, unsafe_allow_html=True)

        if len(df_h) > 1:
            st.markdown("---")
            risk_counts = df_h["risk_level"].value_counts()
            color_map = {
                "High Risk": "#fc8181",
                "Moderate Risk": "#f6ad55",
                "Low Risk": "#68d391",
            }
            fig_pie = go.Figure(go.Pie(
                labels=risk_counts.index.tolist(),
                values=risk_counts.values.tolist(),
                hole=0.6,
                marker_colors=[color_map.get(l, "#718096") for l in risk_counts.index],
                textfont={"color": "#e2e8f0"},
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#a0aec0"},
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(font={"color": "#a0aec0"}),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div class='hero-title' style='font-size:2rem;'>About Dr.ML</div>
        <div class='hero-sub'>AI-powered multi disease prediction platform</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Diseases Supported</div>
            <div style='color:#718096;font-size:0.9rem;line-height:2;'>
                🫀 Heart Disease — Cleveland dataset<br>
                🩺 Diabetes — Pima Indians dataset<br>
                🔬 More diseases coming soon...
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Tech Stack</div>
            <div style='color:#718096;font-size:0.9rem;line-height:2;'>
                Frontend: Streamlit + Plotly<br>
                Backend: FastAPI + Uvicorn<br>
                ML: XGBoost + Scikit-learn<br>
                Deployment: Render + Streamlit Cloud
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Disclaimer</div>
        <div style='color:#4a5568;font-size:0.85rem;line-height:1.8;'>
            For educational purposes only. Not a substitute for medical advice.
            Always consult a qualified healthcare provider for medical decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)