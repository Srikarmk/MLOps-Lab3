"""MLflow "models from code" definition shared by the baseline and the tuned model.

MLflow logs this file as source code instead of pickling a Python object, so loading the model
only runs code that is checked into the repo. The fitted scikit-learn pipeline (preprocessing plus
classifier) is passed in as an artifact.
"""

import joblib
import mlflow
import pandas as pd
from mlflow.models import set_model


class PositiveClassProbability(mlflow.pyfunc.PythonModel):
    """Returns P(income > 50K) instead of a 0/1 label."""

    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["sklearn_pipeline"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.pipeline.predict_proba(model_input)[:, 1]


set_model(PositiveClassProbability())
