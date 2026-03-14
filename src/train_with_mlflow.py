"""
MLflow Experiment Tracking for Customer Churn Model Comparison.

Trains Logistic Regression, XGBoost, and Random Forest models,
logging parameters, metrics, and artifacts to MLflow.
"""

import os
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

# ---------------------------------------------------------------------------
# MLflow setup – local file store
# ---------------------------------------------------------------------------
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
EXPERIMENT_NAME = "churn-model-comparison"
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

# ---------------------------------------------------------------------------
# Feature columns (same preprocessing as notebooks)
# ---------------------------------------------------------------------------
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
models = [
    {
        "name": "LogisticRegression",
        "tag": "v1",
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    },
    {
        "name": "XGBoost",
        "tag": "v2",
        "estimator": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
        },
    },
    {
        "name": "RandomForest",
        "tag": "v3",
        "estimator": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "params": {"n_estimators": 200, "max_depth": 10},
    },
]

# ---------------------------------------------------------------------------
# Train & log each model
# ---------------------------------------------------------------------------
results = []

for model_cfg in models:
    with mlflow.start_run(run_name=model_cfg["name"]):
        print(f"\n{'='*60}")
        print(f"Training {model_cfg['name']} …")
        print(f"{'='*60}")

        # Build pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model_cfg["estimator"]),
            ]
        )
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log parameters
        mlflow.log_param("model_name", model_cfg["name"])
        for k, v in model_cfg["params"].items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("f1", round(f1, 4))
        mlflow.log_metric("auc_roc", round(auc, 4))

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")

        # Tag with version
        mlflow.set_tag("model_version", model_cfg["tag"])

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")

        results.append(
            {
                "Model": model_cfg["name"],
                "Version": model_cfg["tag"],
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "AUC-ROC": round(auc, 4),
            }
        )

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))
print()
