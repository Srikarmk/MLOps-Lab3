"""Register models and manage them with registry aliases instead of the deprecated stages.

    python register_model.py baseline        register the latest baseline, then run the promotion check
    python register_model.py best-xgboost    register the latest sweep's best model, then run the promotion check
    python register_model.py rollback        point @champion back at @previous-champion
    python register_model.py status          show every version and its aliases

Aliases used:
    @champion            the version that batch inference and serving load
    @challenger          the most recently registered version
    @previous-champion   the version that was champion before the last promotion (for rollback)
"""

import argparse

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from common import MODEL_NAME, connect
from data import load_wine_data, split

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous-champion"
MODEL_DESCRIPTION = (
    "Predicts the probability that a wine is high quality (quality >= 7) from 11 physicochemical "
    "measurements plus an is_red flag. Output: P(high quality) between 0 and 1."
)

SOURCES = {
    "baseline": "tags.pipeline_step = 'baseline'",
    "best-xgboost": "tags.pipeline_step = 'tuning'",
}


def alias_version(client, alias):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def registry_auc(model_uri, X, y):
    return roc_auc_score(y, mlflow.pyfunc.load_model(model_uri).predict(X))


def describe_registered_model(client):
    model = client.get_registered_model(MODEL_NAME)
    if not model.description:
        client.update_registered_model(MODEL_NAME, description=MODEL_DESCRIPTION)
        for key, value in {"task": "binary-classification", "target": "quality >= 7", "dataset": "UCI wine quality"}.items():
            client.set_registered_model_tag(MODEL_NAME, key, value)


def register(client, which, min_improvement):
    runs = mlflow.search_runs(filter_string=SOURCES[which], order_by=["attributes.start_time DESC"], max_results=1)
    if runs.empty:
        raise SystemExit(f"No {which} run found. Train one first.")
    run = runs.iloc[0]
    source_uri = run["tags.model_uri"]

    existing = client.search_model_versions(f"name = '{MODEL_NAME}'")
    for v in existing:
        if v.run_id == run.run_id:
            raise SystemExit(f"Run {run.run_id} is already registered as v{v.version}. Train a new model first.")

    version = mlflow.register_model(source_uri, MODEL_NAME, tags={
        "model_family": run["tags.model_family"],
        "data_version": run["tags.data_version"],
        "source_pipeline_step": run["tags.pipeline_step"],
        "val_auc": f"{run['metrics.val_auc']:.4f}",
        "test_auc": f"{run['metrics.test_auc']:.4f}",
    })
    describe_registered_model(client)
    client.set_registered_model_alias(MODEL_NAME, CHALLENGER, version.version)
    print(f"Registered {MODEL_NAME} v{version.version} ({run['tags.model_family']}, run {run.run_id}) as @{CHALLENGER}")

    promote_if_better(client, version, min_improvement)


def promote_if_better(client, challenger, min_improvement):
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())
    champion = alias_version(client, CHAMPION)

    # Both models are loaded from the registry and compared on the validation set. The test set is only
    # reported: choosing between models by test AUC would make it part of model selection.
    challenger_auc = registry_auc(f"models:/{MODEL_NAME}/{challenger.version}", X_val, y_val)
    if champion is None:
        decision, reason = "promoted", "no champion yet"
    else:
        champion_auc = registry_auc(f"models:/{MODEL_NAME}@{CHAMPION}", X_val, y_val)
        gain = challenger_auc - champion_auc
        print(f"  challenger v{challenger.version} val AUC = {challenger_auc:.4f}")
        print(f"  champion   v{champion.version} val AUC = {champion_auc:.4f}  (gain {gain:+.4f}, need > {min_improvement:+.4f})")
        if gain > min_improvement:
            decision, reason = "promoted", f"val AUC {challenger_auc:.4f} beats v{champion.version} ({champion_auc:.4f})"
        else:
            decision, reason = "rejected", f"val AUC {challenger_auc:.4f} does not beat v{champion.version} ({champion_auc:.4f})"

    client.set_model_version_tag(MODEL_NAME, challenger.version, "promotion_check", decision)
    client.update_model_version(MODEL_NAME, challenger.version, description=f"Promotion check {decision}: {reason}.")

    if decision == "promoted":
        if champion is not None:
            client.set_registered_model_alias(MODEL_NAME, PREVIOUS, champion.version)
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, challenger.version)
        print(f"Promoted v{challenger.version} to @{CHAMPION} ({reason})")
    else:
        print(f"Kept v{champion.version} as @{CHAMPION}; v{challenger.version} stays @{CHALLENGER} ({reason})")


def rollback(client):
    previous = alias_version(client, PREVIOUS)
    current = alias_version(client, CHAMPION)
    if previous is None:
        raise SystemExit(f"No @{PREVIOUS} to roll back to.")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous.version)
    client.delete_registered_model_alias(MODEL_NAME, PREVIOUS)
    client.set_model_version_tag(MODEL_NAME, current.version, "rolled_back", "true")
    print(f"Rolled back @{CHAMPION} from v{current.version} to v{previous.version}")


def status(client):
    model = client.get_registered_model(MODEL_NAME)
    aliases = {}
    for alias, version in model.aliases.items():
        aliases.setdefault(version, []).append(f"@{alias}")
    print(f"{MODEL_NAME}: {model.description}")
    print(f"{'version':<8} {'family':<14} {'val_auc':<8} {'test_auc':<9} {'check':<9} aliases")
    for v in sorted(client.search_model_versions(f"name = '{MODEL_NAME}'"), key=lambda v: int(v.version)):
        t = v.tags
        print(f"v{v.version:<7} {t.get('model_family', ''):<14} {t.get('val_auc', ''):<8} {t.get('test_auc', ''):<9} "
              f"{t.get('promotion_check', ''):<9} {' '.join(aliases.get(v.version, []))}")


def main():
    parser = argparse.ArgumentParser(description="Model registry with champion/challenger aliases")
    parser.add_argument("action", choices=["baseline", "best-xgboost", "rollback", "status"])
    parser.add_argument("--min-improvement", type=float, default=0.0,
                        help="validation AUC the challenger must gain over the champion to be promoted")
    args = parser.parse_args()

    connect()
    client = MlflowClient()
    if args.action == "rollback":
        rollback(client)
    elif args.action == "status":
        status(client)
    else:
        register(client, args.action, args.min_improvement)
    if args.action != "status":
        print()
        status(client)


if __name__ == "__main__":
    main()
