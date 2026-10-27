import argparse

import mlflow
import mlflow.xgboost
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score

from common import connect
from data import load_wine_data, split
from tracking import log_datasets, set_run_tags

SEARCH_SPACE = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "binary:logistic",
    "seed": 123,
}


def make_objective(X_train, y_train, X_val, y_val):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)

    def train_model(params):
        mlflow.xgboost.autolog(log_models=False, silent=True)
        with mlflow.start_run(nested=True):
            set_run_tags(pipeline_step="tuning_trial", model_family="xgboost")
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(validation, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            auc = roc_auc_score(y_val, booster.predict(validation))
            mlflow.log_metric("auc", auc)

            signature = infer_signature(X_train, booster.predict(train))
            mlflow.xgboost.log_model(booster, name="model", signature=signature, input_example=X_train.head(5))
        return {"status": STATUS_OK, "loss": -auc}

    return train_model


def main():
    parser = argparse.ArgumentParser(description="Hyperopt sweep over XGBoost, logged to MLflow")
    parser.add_argument("--max-evals", type=int, default=32)
    args = parser.parse_args()

    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())

    # Each trial becomes a child run of this parent run
    with mlflow.start_run(run_name="xgboost_models"):
        set_run_tags(pipeline_step="tuning", model_family="xgboost")
        log_datasets(training=(X_train, y_train), validation=(X_val, y_val))
        fmin(
            fn=make_objective(X_train, y_train, X_val, y_val),
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=Trials(),
        )

    best = mlflow.search_runs(
        filter_string="tags.mlflow.parentRunId != ''", order_by=["metrics.auc DESC"], max_results=1
    ).iloc[0]
    print(f"Best of {args.max_evals} trials: validation AUC = {best['metrics.auc']:.4f} (run {best.run_id})")


if __name__ == "__main__":
    main()
