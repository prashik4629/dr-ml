import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

from backend.core.config import settings


def setup_logging():
    settings.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOG_PATH),
        ],
    )


def load_data():
    df = pd.read_csv(settings.DATASET_PATH)
    logging.info(f"Dataset loaded — shape: {df.shape}")

    zero_not_valid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_not_valid] = df[zero_not_valid].replace(0, np.nan)

    df["BMI_Age_ratio"] = df["BMI"] / (df["Age"] + 1)
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_ratio"] = df["Insulin"] / (df["Glucose"] + 1)
    df["Age_Pregnancies"] = df["Age"] * df["Pregnancies"]

    X = df.drop(columns=[settings.TARGET_COL])
    y = df[settings.TARGET_COL]

    logging.info(f"Class distribution — 0: {(y==0).sum()} | 1: {(y==1).sum()}")
    return X, y


def get_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,
        stratify=y,
        random_state=settings.RANDOM_STATE,
    )
    logging.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_candidates():
    def make_pipeline(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    return {
        "RandomForest": make_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            random_state=settings.RANDOM_STATE, n_jobs=-1,
        )),
        "XGBoost": make_pipeline(XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=settings.RANDOM_STATE, verbosity=0,
        )),
        "LogisticRegression": make_pipeline(LogisticRegression(
            C=0.5, max_iter=1000,
            random_state=settings.RANDOM_STATE,
        )),
    }


def run_cross_validation(candidates, X_train, y_train):
    logging.info("Running 5-fold cross validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_STATE)
    scoring = ["accuracy", "recall", "f1", "roc_auc"]
    results = {}

    for name, pipeline in candidates.items():
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        results[name] = {
            "accuracy": cv_results["test_accuracy"].mean(),
            "recall":   cv_results["test_recall"].mean(),
            "f1":       cv_results["test_f1"].mean(),
            "roc_auc":  cv_results["test_roc_auc"].mean(),
        }
        logging.info(
            f"{name} CV — "
            f"Accuracy: {results[name]['accuracy']:.4f} | "
            f"Recall: {results[name]['recall']:.4f} | "
            f"F1: {results[name]['f1']:.4f} | "
            f"ROC-AUC: {results[name]['roc_auc']:.4f}"
        )
    return results


def select_best_model(cv_results, candidates):
    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc"])
    logging.info(f"Best model selected: {best_name}")
    return best_name, candidates[best_name]


def evaluate_on_test(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
    }

    logging.info(f"Test Accuracy : {metrics['accuracy']:.4f}")
    logging.info(f"Test Recall   : {metrics['recall']:.4f}")
    logging.info(f"Test F1       : {metrics['f1']:.4f}")
    logging.info(f"Test ROC-AUC  : {metrics['roc_auc']:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))
    return pipeline, metrics


def save_model(pipeline):
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.MODEL_PATH)
    logging.info(f"Model saved to {settings.MODEL_PATH}")


def train_model():
    setup_logging()
    logging.info("=== Dr.ML Diabetes Training Pipeline Started ===")

    X, y                             = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    candidates                       = build_candidates()
    cv_results                       = run_cross_validation(candidates, X_train, y_train)
    best_name, best                  = select_best_model(cv_results, candidates)
    best, metrics                    = evaluate_on_test(best, X_train, y_train, X_test, y_test)
    save_model(best)

    logging.info("=== Training Pipeline Completed ===")
    return best_name, metrics


if __name__ == "__main__":
    train_model()