#!/usr/bin/env bash
# Serves a registered model version over REST with `mlflow models serve`.
#
#   ./serve_model.sh                 serve models:/adult_income@champion on http://127.0.0.1:5001
#   ./serve_model.sh challenger      serve @challenger instead
#   PORT=5002 ./serve_model.sh       use another port
#
# The alias is resolved once at startup: after a promotion or rollback, restart this to serve the new version.

set -euo pipefail

ALIAS="${1:-champion}"
PORT="${PORT:-5001}"
HOST="${HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-adult_income}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:8080}"
export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false

cd "$(dirname "$0")"

# mlflow models serve starts uvicorn from PATH, so the virtual environment has to be active
if ! command -v uvicorn >/dev/null 2>&1; then
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    else
        echo "No .venv found. Create it and install requirements.txt first (see README)." >&2
        exit 1
    fi
fi

if ! curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null; then
    echo "No MLflow tracking server at $MLFLOW_TRACKING_URI. Start it first (see README)." >&2
    exit 1
fi

echo "Serving models:/$MODEL_NAME@$ALIAS on http://$HOST:$PORT/invocations (Ctrl+C to stop)"
# --env-manager local reuses this environment instead of building a new one from the model's requirements
exec mlflow models serve -m "models:/$MODEL_NAME@$ALIAS" --env-manager local -h "$HOST" -p "$PORT"
