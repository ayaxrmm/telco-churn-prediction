"""
Запуск:      python train_mlflow.py
Интерфейс:   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
"""
import joblib
import mlflow
import mlflow.sklearn
import skops.io as sio
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train import (CATEGORICAL_FEATURES, DATA_PATH, MODEL_PATH,
                   NUMERIC_FEATURES, RANDOM_STATE, TARGET, load_data)
from train import build_pipeline as _build_lr_pipeline

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "telco-churn"
SELECTION_METRIC = "f1"
REGISTERED_MODEL_NAME = "telco-churn-model"

# Модели
CANDIDATES = {
    "logreg_C1": (LogisticRegression, {
        "C": 1.0, "class_weight": "balanced", "max_iter": 1000,
    }),
    "logreg_C0.1": (LogisticRegression, {
        "C": 0.1, "class_weight": "balanced", "max_iter": 1000,
    }),
    "random_forest": (RandomForestClassifier, {
        "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5,
        "class_weight": "balanced",
    }),
    "gradient_boosting": (GradientBoostingClassifier, {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
    }),
}


def build_pipeline(model_cls, params) -> Pipeline:
    preprocessor = _build_lr_pipeline().named_steps["preprocessor"]
    model = model_cls(**params, random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    best_name, best_score, best_pipeline, best_model_uri = None, -1.0, None, None

    for name, (model_cls, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(model_cls, params)
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_param("model_type", model_cls.__name__)
            mlflow.log_params(params)
            mlflow.log_param("test_size", 0.2)

            mlflow.log_metrics(metrics)

            trusted_types = sio.get_untrusted_types(data=sio.dumps(pipeline))
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=X_test.head(3),
                skops_trusted_types=trusted_types,
            )

            print(f"{name:<18} roc_auc={metrics['roc_auc']:.3f}  "
                  f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}")

            if metrics[SELECTION_METRIC] > best_score:
                best_name, best_score, best_pipeline = name, metrics[SELECTION_METRIC], pipeline
                best_model_uri = model_info.model_uri

    print(f"\nЛучшая модель: {best_name} ({SELECTION_METRIC} = {best_score:.3f})")

    register_best_model(best_model_uri, best_name, best_score)

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")


def register_best_model(model_uri: str, run_name: str, score: float):
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = mlflow.MlflowClient()
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        description=f"{run_name}, {SELECTION_METRIC}={score:.3f}. "
                    f"Selected as best of {len(CANDIDATES)} candidates.",
    )
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)

    print(f"Зарегистрирована: {REGISTERED_MODEL_NAME} версия {version.version} (alias: champion)")


if __name__ == "__main__":
    main()
