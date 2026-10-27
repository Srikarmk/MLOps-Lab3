import os

import mlflow

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
MODEL_NAME = os.getenv("MODEL_NAME", "wine_quality")


def connect():
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(TRACKING_URI)
    return TRACKING_URI
