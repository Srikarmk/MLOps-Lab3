import argparse
import json

import lightgbm
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from common import connect
from data import CATEGORICAL_FEATURES, load_adult_data, slice_groups, split
from features import ordinal_preprocessor
from tracking import (
    log_classification_eval,
    log_datasets,
    log_feature_importance,
    log_proba_model,
    log_slice_metrics,
    set_run_tags,
)

SEARCH_SPACE = {
    "num_leaves": scope.int(hp.quniform("num_leaves", 8, 256, 1)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "min_child_samples": scope.int(hp.quniform("min_child_samples", 5, 200, 1)),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
    # "balanced" reweights the classes (24% earn > 50K); hyperopt decides whether that helps AUC
    "class_weight": hp.choice("class_weight", [None, "balanced"]),
}
SEARCH_SPACE_DOC = {
    "num_leaves": "quniform(8, 256, 1), cast to int",
    "learning_rate": "loguniform(0.01, 0.3)",
    "min_child_samples": "quniform(5, 200, 1), cast to int",
    "subsample": "uniform(0.5, 1.0), row sampling every iteration",
    "colsample_bytree": "uniform(0.5, 1.0)",
    "reg_lambda": "loguniform(0.001, 10)",
    "class_weight": "choice(None, 'balanced')",
}
TUNED_PARAMS = list(SEARCH_SPACE_DOC)
LOG_SCALE_PARAMS = {"learning_rate", "reg_lambda"}
CATEGORICAL_PARAMS = {"class_weight"}
FIXED_PARAMS = {
    "n_estimators": 2000,  # upper bound; early stopping picks the real number of trees
    "subsample_freq": 1,
    "metric": "auc",
    "random_state": 123,
    "deterministic": True,
    "force_row_wise": True,
    # On a dataset this small extra threads only add overhead (1 thread was 10x faster than 18 here), and a
    # fixed thread count keeps the trees identical on machines with different numbers of cores
    "n_jobs": 1,
    "verbose": -1,
}


def fit_classifier(params, train, validation):
    """Fit on already-encoded data, stopping when validation AUC hasn't improved for 50 rounds."""
    (X_train, y_train), (X_val, y_val) = train, validation
    classifier = LGBMClassifier(**FIXED_PARAMS, **params)
    classifier.fit(
        X_train, y_train,
        eval_X=X_val,
        eval_y=y_val,
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lightgbm.early_stopping(50, verbose=False)],
    )
    return classifier


def make_objective(train, validation):
    trial_number = iter(range(1, 10_000))
    X_val, y_val = validation

    def objective(params):
        n = next(trial_number)
        # Trials only log what's needed to compare them; the best config is refit and logged once afterwards
        with mlflow.start_run(run_name=f"trial-{n:03d}", nested=True):
            set_run_tags(pipeline_step="tuning_trial", model_family="lightgbm")
            mlflow.set_tag("trial_number", n)
            mlflow.log_params(params)

            classifier = fit_classifier(params, train, validation)
            scores = classifier.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, scores)
            mlflow.log_metrics({
                "val_auc": val_auc,
                "val_log_loss": log_loss(y_val, scores),
                "best_iteration": classifier.best_iteration_,
            })
        return {"status": STATUS_OK, "loss": -val_auc}

    return objective


def log_sweep_summary(parent_run_id):
    """Pull every child run back out of MLflow and summarise the sweep on the parent run."""
    trials = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["attributes.start_time ASC"],
    )
    columns = ["tags.trial_number", "metrics.val_auc", "metrics.val_log_loss", "metrics.best_iteration"]
    columns += [f"params.{p}" for p in TUNED_PARAMS]
    table = trials[columns + ["run_id"]].rename(columns=lambda c: c.split(".", 1)[-1])
    table["trial_number"] = table["trial_number"].astype(int)
    table = table.sort_values("trial_number").reset_index(drop=True)
    mlflow.log_table(table, "sweep/trials.json")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["trial_number"], table["val_auc"], "o", alpha=0.6, label="trial")
    ax.plot(table["trial_number"], table["val_auc"].cummax(), "-", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("validation AUC")
    ax.set_title("Hyperopt optimisation history")
    ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "sweep/optimisation_history.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(TUNED_PARAMS), figsize=(3.5 * len(TUNED_PARAMS), 3.5), sharey=True)
    for ax, param in zip(axes, TUNED_PARAMS):
        values = table[param] if param in CATEGORICAL_PARAMS else table[param].astype(float)
        ax.scatter(values, table["val_auc"], alpha=0.7)
        ax.set_xlabel(param)
        if param in LOG_SCALE_PARAMS:
            ax.set_xscale("log")
    axes[0].set_ylabel("validation AUC")
    fig.suptitle("Validation AUC by hyperparameter")
    fig.tight_layout()
    mlflow.log_figure(fig, "sweep/params_vs_auc.png")
    plt.close(fig)

    return table


def main():
    parser = argparse.ArgumentParser(description="Hyperopt sweep over LightGBM, logged to MLflow")
    parser.add_argument("--max-evals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123, help="seed for the hyperopt search itself")
    args = parser.parse_args()

    connect()
    data = load_adult_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(data)

    # Encoding is fitted once on the training set and shared by every trial
    prep = ordinal_preprocessor().fit(X_train)
    train = (prep.transform(X_train), y_train)
    validation = (prep.transform(X_val), y_val)

    with mlflow.start_run(run_name="lightgbm_sweep") as parent:
        set_run_tags(pipeline_step="tuning", model_family="lightgbm")
        log_datasets(training=(X_train, y_train), validation=(X_val, y_val), test=(X_test, y_test))
        mlflow.log_params({"max_evals": args.max_evals, "search_seed": args.seed, "algorithm": "tpe",
                           "categorical_encoding": "ordinal (native LightGBM categories)"})
        mlflow.log_dict(SEARCH_SPACE_DOC, "sweep/search_space.json")

        trials = Trials()
        best = fmin(
            fn=make_objective(train, validation),
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
            rstate=np.random.default_rng(args.seed),
            show_progressbar=False,
        )
        best_params = space_eval(SEARCH_SPACE, best)
        table = log_sweep_summary(parent.info.run_id)
        best_trial = table.loc[table["val_auc"].idxmax()]

        # Refit the winning config (same seed, so the same trees) and log it, with its encoder, as one model
        classifier = fit_classifier(best_params, train, validation)
        pipeline = Pipeline([("prep", prep), ("classifier", classifier)])
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_iteration", classifier.best_iteration_)
        mlflow.set_tag("best_trial_run_id", best_trial["run_id"])

        val_metrics = log_classification_eval(y_val, pipeline.predict_proba(X_val)[:, 1], "val")
        test_scores = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = log_classification_eval(y_test, test_scores, "test")
        log_slice_metrics(y_test, test_scores, slice_groups(data, X_test), "test")
        gain = classifier.booster_.feature_importance(importance_type="gain")
        log_feature_importance(gain, classifier.booster_.feature_name())

        model_info = log_proba_model(pipeline, X_train, extra_requirements=[f"lightgbm=={lightgbm.__version__}"])
        mlflow.set_tag("model_uri", model_info.model_uri)

    print(f"Sweep of {args.max_evals} trials (run {parent.info.run_id})")
    print(f"  best trial {int(best_trial['trial_number'])}: val AUC = {val_metrics['val_auc']:.4f}, "
          f"test AUC = {test_metrics['test_auc']:.4f}")
    print(f"  best params: {json.dumps(best_params, default=float)}")


if __name__ == "__main__":
    main()
