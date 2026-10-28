import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import connect
from data import load_adult_data, slice_groups, split
from features import one_hot_preprocessor
from tracking import (
    log_classification_eval,
    log_datasets,
    log_feature_importance,
    log_proba_model,
    log_slice_metrics,
    set_run_tags,
)


def main():
    connect()
    data = load_adult_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(data)

    with mlflow.start_run(run_name="logistic_regression_baseline") as run:
        set_run_tags(pipeline_step="baseline", model_family="logistic_regression")
        log_datasets(training=(X_train, y_train), validation=(X_val, y_val), test=(X_test, y_test))

        classifier = LogisticRegression(C=1.0, max_iter=2000)
        pipeline = Pipeline([("prep", one_hot_preprocessor()), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)

        mlflow.log_params({
            "C": classifier.C,
            "penalty": "l2",
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "numeric_encoding": "standard_scaler",
            "categorical_encoding": "one_hot",
        })

        val_metrics = log_classification_eval(y_val, pipeline.predict_proba(X_val)[:, 1], "val")
        test_scores = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = log_classification_eval(y_test, test_scores, "test")
        log_slice_metrics(y_test, test_scores, slice_groups(data, X_test), "test")

        # Coefficients are comparable because the numeric inputs are standardised
        encoded_names = pipeline.named_steps["prep"].get_feature_names_out()
        log_feature_importance(np.abs(classifier.coef_[0]), encoded_names)

        model_info = log_proba_model(pipeline, X_train)
        mlflow.set_tag("model_uri", model_info.model_uri)

    print(f"Baseline logistic regression: val AUC = {val_metrics['val_auc']:.4f}, "
          f"test AUC = {test_metrics['test_auc']:.4f}")
    print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
