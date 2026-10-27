import os

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality")
MODEL_NAME = os.getenv("MODEL_NAME", "wine_quality")

EXPERIMENT_DESCRIPTION = (
    "Binary classification of high-quality wines (quality >= 7) from the UCI Wine Quality "
    "red and white datasets. Random forest baseline vs. a hyperopt-tuned XGBoost sweep."
)


def connect():
    """Point MLflow at the tracking server and make the lab's experiment active."""
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    if "mlflow.note.content" not in experiment.tags:
        client = MlflowClient()
        client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", EXPERIMENT_DESCRIPTION)
        client.set_experiment_tag(experiment.experiment_id, "task", "binary-classification")
        client.set_experiment_tag(experiment.experiment_id, "target", "quality >= 7")
    return experiment
