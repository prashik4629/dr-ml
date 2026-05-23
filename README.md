# 🏥 Dr.ML — Multi Disease Prediction Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Render](https://img.shields.io/badge/Backend-Render-purple?style=flat-square)
![Streamlit Cloud](https://img.shields.io/badge/Frontend-Streamlit%20Cloud-red?style=flat-square)

A professional, production-inspired AI healthcare platform that predicts multiple diseases using machine learning. Built as a monorepo with two independent ML services and one unified frontend.

---

## 🌐 Live Demo

| Component | Link |
|---|---|
| 🖥️ Combined Frontend | [doctorpredictor.streamlit.app](https://doctorpredictor.streamlit.app) |
| 🫀 Heart Disease API | [medbuddy-ml-api.onrender.com](https://medbuddy-ml-api.onrender.com) |
| 🩺 Diabetes API | [diabetes-ml-api-jpvw.onrender.com](https://diabetes-ml-api-jpvw.onrender.com) |
| 📖 Heart API Docs | [medbuddy-ml-api.onrender.com/docs](https://medbuddy-ml-api.onrender.com/docs) |
| 📖 Diabetes API Docs | [diabetes-ml-api-jpvw.onrender.com/docs](https://diabetes-ml-api-jpvw.onrender.com/docs) |

> ⚠️ Backend services are hosted on Render's free tier. First request may take 30–50 seconds to wake up.

---

## 🧠 Project Overview

Dr.ML is a multi-disease prediction system that takes clinical features as input and predicts disease risk along with a confidence probability and risk level classification. The platform currently supports two disease modules with more planned.

This project demonstrates production-level ML engineering practices including model comparison, cross-validation, API design, authentication, and cloud deployment — all within a clean monorepo architecture.

---

## 🗂️ Repository Structure

```
dr-ml/
│
├── medbuddy-ml/                ← Heart Disease Prediction Service
│   ├── backend/
│   │   ├── core/config.py      # Centralized settings
│   │   ├── core/security.py    # API key authentication
│   │   ├── routers/prediction.py
│   │   ├── services/predictor.py
│   │   ├── main.py
│   │   └── training.py
│   ├── dataset/heart.csv
│   ├── model_dir/
│   └── requirements-backend.txt
│
├── diabetes-ml/                ← Diabetes Prediction Service
│   ├── backend/
│   │   ├── core/config.py
│   │   ├── core/security.py
│   │   ├── routers/prediction.py
│   │   ├── services/predictor.py
│   │   ├── main.py
│   │   └── training.py
│   ├── dataset/diabetes.csv
│   ├── model_dir/
│   └── requirements-backend.txt
│
├── frontend/                   ← Combined Streamlit Frontend
│   ├── app.py
│   └── requirements.txt
│
└── .gitignore
```

---

## ✨ Features

### Machine Learning
- Multiple models compared — Random Forest, XGBoost, Logistic Regression
- 5-fold Stratified Cross Validation for reliable evaluation
- Automatic best model selection based on ROC-AUC score
- Feature engineering for diabetes model (4 new derived features)
- Zero-value imputation for medically invalid entries
- Group-based train/test split to prevent data leakage

### Backend (Both Services)
- FastAPI with versioned endpoints (`/api/v1/predict`)
- API key authentication on all prediction endpoints
- Pydantic input validation with clinical range checks
- Structured logging to file and console
- Global exception handling
- CORS middleware
- `/health` endpoint for uptime monitoring

### Frontend
- Unified UI — switch between Heart Disease and Diabetes with one click
- Biopunk / Medical Sci-Fi dark theme with animated particle background
- Interactive risk gauge chart (Plotly)
- Session-based prediction history with distribution chart
- Risk level badges — Low / Moderate / High
- Clinical interpretation text per prediction

---

## 🏗️ Architecture

```
Combined Frontend (Streamlit Cloud)
           │
     ┌─────┴─────┐
     ▼           ▼
Heart Disease   Diabetes
  FastAPI         FastAPI
  (Render)        (Render)
     │               │
  XGBoost      Best CV Model
  Pipeline       Pipeline
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| Backend | FastAPI, Uvicorn |
| ML Models | XGBoost, Scikit-learn, Random Forest |
| Config | Pydantic Settings, python-dotenv |
| Serialization | Joblib |
| Hosting (Backend) | Render (Free Tier) |
| Hosting (Frontend) | Streamlit Community Cloud |
| Version Control | GitHub |

---

## 📊 Model Performance

### 🫀 Heart Disease (Cleveland Dataset — 1,025 records)

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.8735 | 0.9158 | 0.8795 | 0.9555 |
| **XGBoost** | **0.9332** | **0.9966** | **0.9365** | **0.9977** |
| Logistic Regression | 0.8332 | 0.8838 | 0.8426 | 0.9198 |

**Winner: XGBoost** — Test ROC-AUC: **0.858**

### 🩺 Diabetes (Pima Indians Dataset — 768 records)

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.7684 | 0.7983 | 0.7052 | 0.8336 |
| XGBoost | 0.7530 | 0.7322 | 0.6729 | 0.8147 |
| Logistic Regression | 0.7622 | 0.7279 | 0.6821 | 0.8422 |

**Winner: Best model auto-selected via ROC-AUC cross-validation**

---

## 🚀 Local Setup

```bash
# Clone
git clone https://github.com/prashik4629/dr-ml.git
cd dr-ml

# Heart Disease Service
cd medbuddy-ml
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # set PROJECT_ROOT
python -m backend.training
uvicorn backend.main:app --reload --port 8000

# Diabetes Service (new terminal)
cd diabetes-ml
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # set PROJECT_ROOT
python -m backend.training
uvicorn backend.main:app --reload --port 8001

# Combined Frontend (new terminal)
cd frontend
streamlit run app.py
```

---

## 🔒 API Authentication

```bash
curl -X POST https://medbuddy-ml-api.onrender.com/api/v1/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

---

## 📋 API Response

```json
{
  "prediction": 1,
  "probability": 0.9220,
  "diagnosis": "Heart Disease Detected",
  "risk_level": "High Risk"
}
```

---

## 🔮 Roadmap

- [ ] Kidney Disease prediction module
- [ ] Liver Disease prediction module
- [ ] SHAP explainability on frontend
- [ ] Patient report PDF export
- [ ] Persistent prediction history with database
- [ ] JWT authentication for multi-user support
- [ ] Model retraining pipeline

---

## ⚠️ Disclaimer

For **educational and demonstration purposes only**. Not a substitute for professional medical advice. Always consult a qualified healthcare provider.

---

## 👤 Author

**Prashik Meshram** — [@prashik4629](https://github.com/prashik4629)

---

## 📄 Datasets

- **Heart Disease** — Cleveland Heart Disease Dataset, UCI ML Repository
- **Diabetes** — Pima Indians Diabetes Dataset, UCI ML Repository
