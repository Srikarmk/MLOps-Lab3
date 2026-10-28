"""Check the REST endpoint started by serve_model.sh against the same registry version loaded locally.

    python score_endpoint.py                       check the server on port 5001 serving @champion
    python score_endpoint.py --alias challenger    compare against @challenger instead
"""

import argparse
import statistics
import time

import mlflow
import mlflow.pyfunc
import numpy as np
import requests
from mlflow.tracking import MlflowClient

from common import MODEL_NAME, connect
from data import load_wine_data, split
from tracking import log_classification_eval, set_run_tags

SINGLE_REQUESTS = 50
TOLERANCE = 1e-6


def wait_for_server(url, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/ping", timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise SystemExit(f"No model server answering at {url}/ping. Start it with ./serve_model.sh first.")


def invoke(url, payload):
    start = time.perf_counter()
    response = requests.post(f"{url}/invocations", json=payload, timeout=30)
    return response, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description="Score the test set through the model server and verify it")
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--alias", default="champion", help="registry alias the server is expected to serve")
    parser.add_argument("--wait", type=int, default=60, help="seconds to wait for the server to come up")
    args = parser.parse_args()

    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())
    wait_for_server(args.url, args.wait)

    model_uri = f"models:/{MODEL_NAME}@{args.alias}"
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, args.alias)
    local_scores = mlflow.pyfunc.load_model(model_uri).predict(X_test)

    with mlflow.start_run(run_name="endpoint_check"):
        set_run_tags(pipeline_step="serving_check", model_family=version.tags.get("model_family", ""))
        mlflow.log_params({"endpoint": f"{args.url}/invocations", "model_uri": model_uri, "model_version": version.version})

        # 1. Whole test set in one request, pandas "split" orientation (same payload as the original lab)
        response, batch_ms = invoke(args.url, {"dataframe_split": X_test.to_dict(orient="split")})
        response.raise_for_status()
        served_scores = np.asarray(response.json()["predictions"], dtype=float)
        max_diff = float(np.max(np.abs(served_scores - local_scores)))
        matches = served_scores.shape == local_scores.shape and max_diff <= TOLERANCE
        metrics = log_classification_eval(y_test, served_scores, "test")

        # 2. One wine per request, "records" orientation, the way a real-time caller would use it
        latencies = []
        for _, row in X_test.head(SINGLE_REQUESTS).iterrows():
            single, ms = invoke(args.url, {"dataframe_records": [row.to_dict()]})
            single.raise_for_status()
            latencies.append(ms)
        p50 = statistics.median(latencies)
        p95 = float(np.percentile(latencies, 95))

        # 3. A request missing columns must be rejected by the model signature, not scored
        bad, _ = invoke(args.url, {"dataframe_split": X_test[["alcohol", "pH"]].head(1).to_dict(orient="split")})
        rejects_bad_input = bad.status_code == 400

        mlflow.log_metrics({
            "max_abs_diff_vs_local": max_diff,
            "batch_latency_ms": batch_ms,
            "single_latency_p50_ms": p50,
            "single_latency_p95_ms": p95,
        })
        passed = matches and rejects_bad_input
        mlflow.set_tags({"matches_local_model": matches, "rejects_bad_input": rejects_bad_input,
                         "endpoint_check": "pass" if passed else "fail"})

    print(f"Endpoint {args.url}/invocations, expected {model_uri} (v{version.version}, {version.tags.get('model_family')})")
    print(f"  batch of {len(X_test)} rows: {batch_ms:.0f} ms, test AUC = {metrics['test_auc']:.4f}")
    print(f"  served vs local predictions: max abs diff = {max_diff:.2e} -> {'match' if matches else 'MISMATCH'}")
    print(f"  {SINGLE_REQUESTS} single-row requests: p50 = {p50:.1f} ms, p95 = {p95:.1f} ms")
    print(f"  request missing {X_test.shape[1] - 2} of {X_test.shape[1]} columns: HTTP {bad.status_code} -> {'rejected' if rejects_bad_input else 'NOT rejected'}")
    if not matches:
        raise SystemExit(f"The server is not serving @{args.alias} (v{version.version}). "
                         "If the alias moved since the server started, restart ./serve_model.sh.")
    if not rejects_bad_input:
        raise SystemExit("The server accepted a request that does not match the model signature.")
    print("Endpoint check passed.")


if __name__ == "__main__":
    main()
