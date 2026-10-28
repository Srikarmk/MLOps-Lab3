import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from common import connect
from data import load_wine_data, split
from tracking import log_classification_eval, log_datasets, log_feature_importance, set_run_tags

SEED = 123
MODEL_CODE = Path(__file__).resolve().parent / "wine_model.py"


def main():
    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())

    with mlflow.start_run(run_name="untuned_random_forest") as run:
        set_run_tags(pipeline_step="baseline", model_family="random_forest")
        log_datasets(training=(X_train, y_train), validation=(X_val, y_val), test=(X_test, y_test))

        model = RandomForestClassifier(n_estimators=10, random_state=np.random.RandomState(SEED))
        model.fit(X_train, y_train)

        params = {k: v for k, v in model.get_params().items() if k != "random_state"}
        mlflow.log_params({**params, "random_state": SEED})

        val_metrics = log_classification_eval(y_val, model.predict_proba(X_val)[:, 1], "val")
        test_metrics = log_classification_eval(y_test, model.predict_proba(X_test)[:, 1], "test")
        log_feature_importance(model.feature_importances_, X_train.columns)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "random_forest.joblib"
            joblib.dump(model, model_path)
            model_info = mlflow.pyfunc.log_model(
                name="random_forest_model",
                python_model=str(MODEL_CODE),
                artifacts={"sklearn_model": str(model_path)},
                pip_requirements=[
                    f"scikit-learn=={sklearn.__version__}",
                    f"joblib=={joblib.__version__}",
                    f"mlflow=={mlflow.__version__}",
                ],
                signature=infer_signature(X_train, model.predict_proba(X_train)[:, 1]),
                input_example=X_train.head(5),
            )
        mlflow.set_tag("model_uri", model_info.model_uri)

    print(f"Baseline random forest: val AUC = {val_metrics['val_auc']:.4f}, test AUC = {test_metrics['test_auc']:.4f}")
    print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
