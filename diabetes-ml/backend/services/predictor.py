import logging

import joblib
import pandas as pd
import numpy as np

from backend.core.config import settings


logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]


def load_model():
    model = joblib.load(settings.MODEL_PATH)
    logger.info("Diabetes model loaded successfully")
    return model


model = load_model()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI_Age_ratio"] = df["BMI"] / (df["Age"] + 1)
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_ratio"] = df["Insulin"] / (df["Glucose"] + 1)
    df["Age_Pregnancies"] = df["Age"] * df["Pregnancies"]
    return df


def predict(input_data: dict) -> dict:
    try:
        df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

        # Replace zeros with NaN for medical columns
        zero_not_valid = [
            "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI"
        ]
        df[zero_not_valid] = df[zero_not_valid].replace(0, np.nan)

        # Add engineered features
        df = add_features(df)

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        logger.info(
            f"Prediction: {prediction} | "
            f"Probability: {probability:.4f}"
        )

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise