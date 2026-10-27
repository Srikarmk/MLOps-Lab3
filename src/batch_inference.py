from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from common import MODEL_NAME, connect
from data import load_wine_data, split
from tracking import log_classification_eval, log_datasets, set_run_tags

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())

    model_uri = f"models:/{MODEL_NAME}/production"
    version = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    model = mlflow.pyfunc.load_model(model_uri)

    # Scoring is a tracked run too, so every prediction file can be traced back to a model version and dataset
    with mlflow.start_run(run_name="batch_inference"):
        set_run_tags(pipeline_step="batch_inference", model_family="registered_model")
        mlflow.log_params({
            "model_uri": model_uri,
            "model_version": version.version,
            "model_source_run_id": version.run_id,
        })
        log_datasets(test=(X_test, y_test))

        scores = model.predict(X_test)
        metrics = log_classification_eval(y_test, scores, "test")

        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / "batch_predictions.csv"
        X_test.assign(prediction=scores, actual=y_test.values).to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path), artifact_path="predictions")

    print(f"Loaded {model_uri} (v{version.version}): test AUC = {metrics['test_auc']:.4f}")
    print(f"Wrote {len(X_test)} predictions to {out_path} and logged them to the run")


if __name__ == "__main__":
    main()
