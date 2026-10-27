"""Helpers that log datasets, evaluation metrics and plots to the active MLflow run."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import TARGET, data_version

DATA_SOURCE_URL = "https://archive.ics.uci.edu/dataset/186/wine+quality"


def set_run_tags(pipeline_step, model_family):
    mlflow.set_tags({
        "pipeline_step": pipeline_step,
        "model_family": model_family,
        "data_version": data_version(),
    })


def log_datasets(**splits):
    """Log each (X, y) split as an MLflow dataset input, e.g. log_datasets(training=(X, y))."""
    for context, (X, y) in splits.items():
        df = pd.concat([X, y], axis=1)
        with warnings.catch_warnings():
            # The 0/1 target is meant to be an integer column, so the "missing values" hint doesn't apply
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            dataset = mlflow.data.from_pandas(
                df,
                source=DATA_SOURCE_URL,
                targets=TARGET,
                name=f"wine-quality-{context}",
            )
            mlflow.log_input(dataset, context=context)


def log_classification_eval(y_true, scores, split_name="test", threshold=0.5):
    """Log threshold-free and thresholded metrics plus ROC and confusion-matrix plots."""
    labels = (scores >= threshold).astype(int)
    metrics = {
        f"{split_name}_auc": roc_auc_score(y_true, scores),
        f"{split_name}_log_loss": log_loss(y_true, scores),
        f"{split_name}_accuracy": accuracy_score(y_true, labels),
        f"{split_name}_precision": precision_score(y_true, labels, zero_division=0),
        f"{split_name}_recall": recall_score(y_true, labels),
        f"{split_name}_f1": f1_score(y_true, labels),
    }
    mlflow.log_metrics(metrics)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax, name=split_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(f"ROC curve ({split_name})")
    mlflow.log_figure(fig, f"plots/roc_curve_{split_name}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, labels, display_labels=["not high", "high"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix ({split_name}, threshold {threshold})")
    mlflow.log_figure(fig, f"plots/confusion_matrix_{split_name}.png")
    plt.close(fig)

    return metrics


def log_feature_importance(importances, feature_names):
    table = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    mlflow.log_table(table, "feature_importance.json")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(table["feature"][::-1], table["importance"][::-1])
    ax.set_xlabel("importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)
    return table
