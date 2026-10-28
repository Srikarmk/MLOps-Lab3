import argparse
import json

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from mlflow.models import infer_signature
from sklearn.metrics import log_loss, roc_auc_score

from common import connect
from data import load_wine_data, split
from tracking import log_classification_eval, log_datasets, log_feature_importance, set_run_tags

SEARCH_SPACE = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "binary:logistic",
    "seed": 123,
}
SEARCH_SPACE_DOC = {
    "max_depth": "quniform(4, 100, 1), cast to int",
    "learning_rate": "loguniform(e^-3, e^0)",
    "reg_alpha": "loguniform(e^-5, e^-1)",
    "reg_lambda": "loguniform(e^-6, e^-1)",
    "min_child_weight": "loguniform(e^-1, e^3)",
}
TUNED_PARAMS = list(SEARCH_SPACE_DOC)


def fit_booster(params, train, validation):
    return xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(validation, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )


def make_objective(train, validation, y_val):
    trial_number = iter(range(1, 10_000))

    def objective(params):
        n = next(trial_number)
        # Trials only log what's needed to compare them; the best config is refit and logged once afterwards
        with mlflow.start_run(run_name=f"trial-{n:03d}", nested=True):
            set_run_tags(pipeline_step="tuning_trial", model_family="xgboost")
            mlflow.set_tag("trial_number", n)
            mlflow.log_params(params)

            booster = fit_booster(params, train, validation)
            scores = booster.predict(validation)
            val_auc = roc_auc_score(y_val, scores)
            mlflow.log_metrics({
                "val_auc": val_auc,
                "val_log_loss": log_loss(y_val, scores),
                "best_iteration": booster.best_iteration,
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

    fig, axes = plt.subplots(1, len(TUNED_PARAMS), figsize=(4 * len(TUNED_PARAMS), 3.5), sharey=True)
    for ax, param in zip(axes, TUNED_PARAMS):
        ax.scatter(table[param].astype(float), table["val_auc"], alpha=0.7)
        ax.set_xlabel(param)
        if param != "max_depth":
            ax.set_xscale("log")
    axes[0].set_ylabel("validation AUC")
    fig.suptitle("Validation AUC by hyperparameter")
    fig.tight_layout()
    mlflow.log_figure(fig, "sweep/params_vs_auc.png")
    plt.close(fig)

    return table


def main():
    parser = argparse.ArgumentParser(description="Hyperopt sweep over XGBoost, logged to MLflow")
    parser.add_argument("--max-evals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123, help="seed for the hyperopt search itself")
    args = parser.parse_args()

    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)

    with mlflow.start_run(run_name="xgboost_models") as parent:
        set_run_tags(pipeline_step="tuning", model_family="xgboost")
        log_datasets(training=(X_train, y_train), validation=(X_val, y_val), test=(X_test, y_test))
        mlflow.log_params({"max_evals": args.max_evals, "search_seed": args.seed, "algorithm": "tpe"})
        mlflow.log_dict(SEARCH_SPACE_DOC, "sweep/search_space.json")

        trials = Trials()
        best = fmin(
            fn=make_objective(train, validation, y_val),
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

        # Refit the winning config (same seed, so the same booster) and log it as the sweep's single model
        booster = fit_booster(best_params, train, validation)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if k in TUNED_PARAMS})
        mlflow.log_metric("best_iteration", booster.best_iteration)
        mlflow.set_tag("best_trial_run_id", best_trial["run_id"])

        val_metrics = log_classification_eval(y_val, booster.predict(validation), "val")
        test_metrics = log_classification_eval(y_test, booster.predict(xgb.DMatrix(X_test)), "test")
        gain = booster.get_score(importance_type="gain")
        log_feature_importance([gain.get(c, 0.0) for c in X_train.columns], X_train.columns)

        mlflow.xgboost.log_model(
            booster,
            name="model",
            signature=infer_signature(X_train, booster.predict(train)),
            input_example=X_train.head(5),
        )

    print(f"Sweep of {args.max_evals} trials (run {parent.info.run_id})")
    print(f"  best trial {int(best_trial['trial_number'])}: val AUC = {val_metrics['val_auc']:.4f}, "
          f"test AUC = {test_metrics['test_auc']:.4f}")
    print(f"  best params: {json.dumps({k: best_params[k] for k in TUNED_PARAMS}, default=float)}")


if __name__ == "__main__":
    main()
