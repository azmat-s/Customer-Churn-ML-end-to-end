"""
Model Registry & Versioning.

Reads all MLflow runs from the "churn-model-comparison" experiment,
selects the best model by AUC-ROC, copies it to the production path,
and writes model_metadata.json.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import mlflow

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "lr_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
EXPERIMENT_NAME = "churn-model-comparison"


def get_next_version() -> int:
    """Read existing metadata and return next version number."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
        return meta.get("version", 0) + 1
    return 1


def main():
    # Retrieve experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Run train_with_mlflow.py first.")
        return

    # Search all runs, ordered by auc_roc descending
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.auc_roc DESC"],
    )

    if runs.empty:
        print("No runs found. Run train_with_mlflow.py first.")
        return

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_auc = best_run["metrics.auc_roc"]
    model_name = best_run["params.model_name"]

    print(f"Best run: {best_run_id}")
    print(f"  Model : {model_name}")
    print(f"  AUC   : {best_auc:.4f}")

    # Load the logged sklearn model from the best run
    model_uri = f"runs:/{best_run_id}/model"
    best_pipeline = mlflow.sklearn.load_model(model_uri)

    # Overwrite the production model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, PRODUCTION_MODEL_PATH)
    print(f"Production model saved to {PRODUCTION_MODEL_PATH}")

    # Write metadata
    version = get_next_version()
    metadata = {
        "model_name": model_name,
        "version": version,
        "auc_roc": round(best_auc, 4),
        "trained_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mlflow_run_id": best_run_id,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nRegistered {model_name} v{version} with AUC {best_auc:.4f} as production model")


if __name__ == "__main__":
    main()
