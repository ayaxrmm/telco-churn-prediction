import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "model.joblib"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main():
    df = load_data(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("Метрики на тестовой выборке:")
    for name, value in metrics.items():
        print(f"  {name:>9}: {value:.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    joblib.dump(pipeline, MODEL_PATH)
    with open("metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=2)
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
