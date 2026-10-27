import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from common import connect
from data import load_wine_data, split
from tracking import log_classification_eval, log_datasets, log_feature_importance, set_run_tags

SEED = 123


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    """Makes the logged model return P(high quality) instead of a 0/1 label."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict_proba(model_input)[:, 1]


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

        wrapped = SklearnModelWrapper(model)
        mlflow.pyfunc.log_model(
            name="random_forest_model",
            python_model=wrapped,
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"mlflow=={mlflow.__version__}"],
            signature=infer_signature(X_train, wrapped.predict(None, X_train)),
            input_example=X_train.head(5),
        )

    print(f"Baseline random forest: val AUC = {val_metrics['val_auc']:.4f}, test AUC = {test_metrics['test_auc']:.4f}")
    print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
