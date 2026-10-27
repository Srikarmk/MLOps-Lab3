import mlflow
import mlflow.pyfunc
import numpy as np
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from common import connect
from data import load_wine_data, split


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    """Makes the logged model return P(high quality) instead of a 0/1 label."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def main():
    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())

    with mlflow.start_run(run_name="untuned_random_forest") as run:
        n_estimators = 10
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("auc", auc)

        wrapped = SklearnModelWrapper(model)
        signature = infer_signature(X_train, wrapped.predict(None, X_train))
        mlflow.pyfunc.log_model(
            name="random_forest_model",
            python_model=wrapped,
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"mlflow=={mlflow.__version__}"],
            signature=signature,
        )

    print(f"Baseline random forest: test AUC = {auc:.4f}")
    print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
