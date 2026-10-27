import argparse
import time

import mlflow
from mlflow.tracking import MlflowClient

from common import MODEL_NAME, connect


def latest_run_id(filter_string, order_by=None):
    runs = mlflow.search_runs(filter_string=filter_string, order_by=order_by, max_results=1)
    if runs.empty:
        raise SystemExit(f"No run matches: {filter_string}")
    return runs.iloc[0].run_id


def main():
    parser = argparse.ArgumentParser(description="Register a model version and move it to Production")
    parser.add_argument("which", choices=["baseline", "best-xgboost"])
    args = parser.parse_args()

    connect()
    client = MlflowClient()

    if args.which == "baseline":
        run_id = latest_run_id('tags.mlflow.runName = "untuned_random_forest"', ["attributes.start_time DESC"])
        source = f"runs:/{run_id}/random_forest_model"
    else:
        run_id = latest_run_id("tags.mlflow.parentRunId != ''", ["metrics.auc DESC"])
        source = f"runs:/{run_id}/model"

    previous = client.get_latest_versions(MODEL_NAME, stages=["Production"]) if client.search_registered_models(
        filter_string=f"name = '{MODEL_NAME}'"
    ) else []

    version = mlflow.register_model(source, MODEL_NAME)
    time.sleep(2)

    for old in previous:
        client.transition_model_version_stage(MODEL_NAME, old.version, stage="Archived")
        print(f"Archived {MODEL_NAME} v{old.version}")
    client.transition_model_version_stage(MODEL_NAME, version.version, stage="Production")
    print(f"Registered {MODEL_NAME} v{version.version} from {source} and moved it to Production")


if __name__ == "__main__":
    main()
