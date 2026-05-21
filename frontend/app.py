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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0f1e; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #1a1f35 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .main .block-container { background: #0a0f1e; padding: 2rem 3rem; }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed 0%, #76e4f7 50%, #b794f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }

    .hero-sub { font-size: 1.1rem; color: #718096; margin-top: 0.5rem; }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        color: #63b3ed;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .risk-high {
        background: rgba(245,101,101,0.15);
        border: 1px solid rgba(245,101,101,0.4);
        color: #fc8181;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .risk-moderate {
        background: rgba(237,137,54,0.15);
        border: 1px solid rgba(237,137,54,0.4);
        color: #f6ad55;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .risk-low {
        background: rgba(72,187,120,0.15);
        border: 1px solid rgba(72,187,120,0.4);
        color: #68d391;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .disease-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #553c9a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.02em;
    }

    hr { border-color: rgba(99,179,237,0.1); margin: 1.5rem 0; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0f1e; }
    ::-webkit-scrollbar-thumb { background: #3182ce; border-radius: 4px; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🏥</div>
        <div style='font-size:1.3rem;font-weight:700;color:#63b3ed;'>Dr.ML</div>
        <div style='font-size:0.75rem;color:#718096;margin-top:0.25rem;'>Multi Disease Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Select Disease</div>', unsafe_allow_html=True)

    disease = st.radio(
        "Disease",
        ["🫀 Heart Disease", "🩺 Diabetes"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Page",
        ["Prediction", "History", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if "Heart" in disease:
        st.markdown("""
        <div class='section-label'>Model Info</div>
        <div style='font-size:0.8rem;color:#718096;line-height:1.8;'>
            Model: XGBoost<br>
            ROC-AUC: 0.858<br>
            Dataset: Cleveland<br>
            Features: 13
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='section-label'>Model Info</div>
        <div style='font-size:0.8rem;color:#718096;line-height:1.8;'>
            Model: Best CV Model<br>
            ROC-AUC: ~0.85<br>
            Dataset: Pima Indians<br>
            Features: 8 + 4 engineered
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#4a5568;text-align:center;'>
        For educational purposes only.<br>
        Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


def show_results(prediction, probability, diagnosis, risk_level, label):
    st.markdown("---")
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

    risk_class = {"High Risk": "risk-high", "Moderate Risk": "risk-moderate", "Low Risk": "risk-low"}[risk_level]
    card_style = """
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(99,179,237,0.2);
        border-radius:12px;
        padding:0;text-align:center;
        height:110px;display:flex;
        flex-direction:column;
        justify-content:center;align-items:center;
    """

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""<div style='{card_style}'>
            <div style='font-size:0.7rem;color:#718096;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.75rem;'>Diagnosis</div>
            <div style='font-size:0.95rem;font-weight:600;color:#e2e8f0;'>{diagnosis}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div style='{card_style}'>
            <div style='font-size:0.7rem;color:#718096;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.75rem;'>Risk Level</div>
            <span class='{risk_class}'>{risk_level}</span>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div style='{card_style}'>
            <div style='font-size:0.7rem;color:#718096;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.75rem;'>Probability</div>
            <div style='font-size:1.8rem;font-weight:700;color:#63b3ed;'>{probability:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _, g_col, _ = st.columns([1, 2, 1])
    with g_col:
        gauge_color = "#fc8181" if probability >= 0.75 else "#f6ad55" if probability >= 0.45 else "#68d391"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 48, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#718096", "size": 12}},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(72,187,120,0.1)"},
                    {"range": [45, 75], "color": "rgba(237,137,54,0.1)"},
                    {"range": [75, 100], "color": "rgba(245,101,101,0.1)"},
                ],
                "threshold": {"line": {"color": gauge_color, "width": 4}, "thickness": 0.85, "value": round(probability * 100, 1)},
            },
            title={"text": f"{label} Risk Score", "font": {"color": "#718096", "size": 14}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"}, height=320,
            margin=dict(t=40, b=0, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════
if page == "Prediction":

    if "Heart" in disease:
        st.markdown("""
        <div style='margin-bottom:2rem;'>
            <p class='hero-title'>Heart Risk Analysis</p>
            <p class='hero-sub'>Enter patient vitals for cardiovascular risk assessment</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

        with st.form("heart_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Demographics**")
                age = st.number_input("Age", min_value=1, max_value=120, value=52)
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                cp = st.selectbox("Chest Pain Type", options=[0,1,2,3], format_func=lambda x: {0:"0 — Typical Angina",1:"1 — Atypical Angina",2:"2 — Non-Anginal Pain",3:"3 — Asymptomatic"}[x])
                fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
            with col2:
                st.markdown("**Vitals**")
                trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=125)
                chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=212)
                thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=168)
                oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            with col3:
                st.markdown("**ECG & Tests**")
                restecg = st.selectbox("Resting ECG", options=[0,1,2], format_func=lambda x: {0:"0 — Normal",1:"1 — ST-T Abnormality",2:"2 — LV Hypertrophy"}[x])
                exang = st.selectbox("Exercise Induced Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
                slope = st.selectbox("ST Slope", options=[0,1,2], format_func=lambda x: {0:"0 — Upsloping",1:"1 — Flat",2:"2 — Downsloping"}[x])
                ca = st.number_input("Major Vessels (0-4)", min_value=0, max_value=4, value=0)
                thal = st.selectbox("Thalassemia", options=[0,1,2,3], format_func=lambda x: {0:"0 — Normal",1:"1 — Fixed Defect",2:"2 — Reversible Defect",3:"3 — Unknown"}[x])

            submitted = st.form_submit_button("Run Heart Risk Analysis")

        if submitted:
            payload = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}
            with st.spinner("Analyzing..."):
                try:
                    r = requests.post(HEART_API_URL, json=payload, headers={"X-API-Key": HEART_API_KEY}, timeout=60)
                    res = r.json()
                    if r.status_code != 200:
                        st.error(f"API Error: {res.get('detail')}")
                        st.stop()
                    st.session_state.history.append({"time": datetime.now().strftime("%H:%M:%S"), "disease": "Heart Disease", "probability": res["probability"], "risk_level": res["risk_level"], "diagnosis": res["diagnosis"]})
                    show_results(res["prediction"], res["probability"], res["diagnosis"], res["risk_level"], "Heart Disease")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        st.markdown("""
        <div style='margin-bottom:2rem;'>
            <p class='hero-title'>Diabetes Risk Analysis</p>
            <p class='hero-sub'>Enter patient vitals for diabetes risk assessment</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Personal Details**")
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
                age = st.number_input("Age", min_value=1, max_value=120, value=35)
                bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=28.5, step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.45, step=0.01)
            with col2:
                st.markdown("**Clinical Measurements**")
                glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=400, value=120)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=250, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=25)
                insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)

            submitted = st.form_submit_button("Run Diabetes Analysis")

        if submitted:
            payload = {"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":blood_pressure,"SkinThickness":skin_thickness,"Insulin":insulin,"BMI":bmi,"DiabetesPedigreeFunction":dpf,"Age":age}
            with st.spinner("Analyzing..."):
                try:
                    r = requests.post(DIABETES_API_URL, json=payload, headers={"X-API-Key": DIABETES_API_KEY}, timeout=60)
                    res = r.json()
                    if r.status_code != 200:
                        st.error(f"API Error: {res.get('detail')}")
                        st.stop()
                    st.session_state.history.append({"time": datetime.now().strftime("%H:%M:%S"), "disease": "Diabetes", "probability": res["probability"], "risk_level": res["risk_level"], "diagnosis": res["diagnosis"]})
                    show_results(res["prediction"], res["probability"], res["diagnosis"], res["risk_level"], "Diabetes")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════
elif page == "History":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <p class='hero-title'>Prediction History</p>
        <p class='hero-sub'>All predictions made in this session</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3rem;'>
            <div style='font-size:2rem;'>📋</div>
            <div style='color:#718096;margin-top:0.5rem;'>No predictions yet.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_h = pd.DataFrame(st.session_state.history)
        for i, row in df_h.iterrows():
            risk_class = {"High Risk":"risk-high","Moderate Risk":"risk-moderate","Low Risk":"risk-low"}[row["risk_level"]]
            icon = "🫀" if row["disease"] == "Heart Disease" else "🩺"
            st.markdown(f"""
            <div class='glass-card' style='display:flex;justify-content:space-between;align-items:center;'>
                <div style='display:flex;gap:1.5rem;align-items:center;'>
                    <div style='font-size:1.5rem;'>{icon}</div>
                    <div style='color:#4a5568;font-size:0.8rem;'>{row["time"]}</div>
                    <div style='color:#e2e8f0;'>{row["disease"]}</div>
                    <div style='color:#63b3ed;font-weight:600;'>{row["probability"]:.1%}</div>
                </div>
                <span class='{risk_class}'>{row["risk_level"]}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <p class='hero-title'>About Dr.ML</p>
        <p class='hero-sub'>AI-powered multi disease prediction platform</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Diseases Supported</div>
            <div style='color:#a0aec0;font-size:0.9rem;line-height:2;'>
                🫀 Heart Disease — Cleveland dataset<br>
                🩺 Diabetes — Pima Indians dataset<br>
                🔬 More coming soon...
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Tech Stack</div>
            <div style='color:#a0aec0;font-size:0.9rem;line-height:2;'>
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
        <div style='color:#718096;font-size:0.85rem;line-height:1.8;'>
            For educational purposes only. Not a substitute for medical advice.
            Always consult a qualified healthcare provider.
        </div>
    </div>
    """, unsafe_allow_html=True)