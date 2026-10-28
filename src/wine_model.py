"""MLflow "models from code" definition for the random forest baseline.

MLflow logs this file as source code instead of pickling a Python object, so loading the model
only runs code that is checked into the repo. The fitted forest itself is passed in as an artifact.
"""

import joblib
import mlflow
import pandas as pd
from mlflow.models import set_model


class WineQualityModel(mlflow.pyfunc.PythonModel):
    """Returns P(high quality) instead of a 0/1 label."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict_proba(model_input)[:, 1]


set_model(WineQualityModel())
