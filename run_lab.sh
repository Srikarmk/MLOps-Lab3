#!/usr/bin/env bash
# Runs the whole lab end to end:
#   venv + installs -> tracking server -> baseline -> register -> hyperopt sweep -> register
#   -> batch inference -> REST serving -> endpoint check -> stop the servers it started
#
#   ./run_lab.sh                  run everything against the tracking server on port 8080 (started if needed)
#   ./run_lab.sh --fresh          delete mlflow.db, mlartifacts/ and outputs/ first, so the registry starts empty
#   ./run_lab.sh --keep-server    leave the tracking server running afterwards to browse the UI
#
# Overrides: PYTHON=python3.12  TRACKING_PORT=8080  SERVE_PORT=5001  MAX_EVALS=32

set -euo pipefail

TRACKING_PORT="${TRACKING_PORT:-8080}"
SERVE_PORT="${SERVE_PORT:-5001}"
MAX_EVALS="${MAX_EVALS:-32}"
FRESH=false
KEEP_SERVER=false
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=true ;;
        --keep-server) KEEP_SERVER=true ;;
        *) echo "Unknown option: $arg (use --fresh and/or --keep-server)" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")"
ROOT="$(pwd)"
LOG_DIR="$ROOT/outputs/logs"
export MLFLOW_TRACKING_URI="http://127.0.0.1:$TRACKING_PORT"
export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false
export MLFLOW_DISABLE_AGENT_HINT=1

TRACKING_PID=""
SERVE_PID=""

step() { printf '\n==> %s\n' "$*"; }

# mlflow starts uvicorn as a child process, so stop the whole tree, not just the PID we launched
stop_tree() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        stop_tree "$child"
    done
    kill "$pid" 2>/dev/null || true
}

cleanup() {
    if [ -n "$SERVE_PID" ]; then
        stop_tree "$SERVE_PID"
        echo "Stopped the model server on port $SERVE_PORT"
    fi
    if [ -n "$TRACKING_PID" ] && ! $KEEP_SERVER; then
        stop_tree "$TRACKING_PID"
        echo "Stopped the tracking server on port $TRACKING_PORT"
    fi
}
trap cleanup EXIT

wait_for() {
    local url="$1" seconds="$2" what="$3" log="$4"
    for _ in $(seq 1 "$seconds"); do
        curl -sf "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "$what did not come up within ${seconds}s. Last lines of $log:" >&2
    tail -n 20 "$log" >&2
    exit 1
}

port_in_use() { curl -s -o /dev/null "http://127.0.0.1:$1" 2>/dev/null; }

step "Checking Python (3.12 or 3.13 required)"
if [ -z "${PYTHON:-}" ]; then
    for candidate in python3.12 python3.13 python3; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 13))'; then
            PYTHON="$candidate"
            break
        fi
    done
fi
if [ -z "${PYTHON:-}" ] && [ ! -x .venv/bin/python ]; then
    echo "Python 3.12 or 3.13 not found. Install it (see README, 'Prerequisites') or set PYTHON=/path/to/python3.12." >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    step "Creating the virtual environment in .venv with $PYTHON"
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
python -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 13))' || {
    echo ".venv uses Python $(python -V 2>&1 | cut -d' ' -f2), which these pins don't support. Delete .venv and rerun." >&2
    exit 1
}
python --version

step "Installing pinned requirements"
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
python -c 'import mlflow, xgboost, sklearn, hyperopt; print(f"mlflow {mlflow.__version__}, xgboost {xgboost.__version__}, scikit-learn {sklearn.__version__}, hyperopt {hyperopt.__version__}")'

if $FRESH; then
    if curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null 2>&1; then
        echo "--fresh deletes mlflow.db, but a tracking server is already running on port $TRACKING_PORT. Stop it first." >&2
        exit 1
    fi
    step "Removing mlflow.db, mlartifacts/ and outputs/ (--fresh)"
    rm -rf mlflow.db mlartifacts outputs
fi
mkdir -p "$LOG_DIR"

step "Tracking server on $MLFLOW_TRACKING_URI"
if curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null 2>&1; then
    echo "Already running, reusing it"
else
    if port_in_use "$TRACKING_PORT"; then
        echo "Port $TRACKING_PORT is taken by something that isn't an MLflow server. Use TRACKING_PORT=<free port>." >&2
        exit 1
    fi
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
        --host 127.0.0.1 --port "$TRACKING_PORT" > "$LOG_DIR/tracking-server.log" 2>&1 &
    TRACKING_PID=$!
    wait_for "$MLFLOW_TRACKING_URI/health" 90 "The tracking server" "$LOG_DIR/tracking-server.log"
    echo "Started (PID $TRACKING_PID, log: outputs/logs/tracking-server.log)"
fi

cd src

step "1/7 Baseline random forest"
python train_baseline.py

step "2/7 Register the baseline (promotion check against @champion)"
python register_model.py baseline

step "3/7 Hyperopt sweep over XGBoost ($MAX_EVALS trials, one nested run each)"
python tune_xgboost.py --max-evals "$MAX_EVALS"

step "4/7 Register the sweep's best model (promotion check against @champion)"
python register_model.py best-xgboost

step "5/7 Batch inference with models:/wine_quality@champion"
python batch_inference.py

step "6/7 Serve models:/wine_quality@champion on http://127.0.0.1:$SERVE_PORT"
if port_in_use "$SERVE_PORT"; then
    echo "Port $SERVE_PORT is already in use (an old model server?). Stop it or use SERVE_PORT=<free port>." >&2
    exit 1
fi
PORT="$SERVE_PORT" "$ROOT/serve_model.sh" > "$LOG_DIR/model-server.log" 2>&1 &
SERVE_PID=$!
wait_for "http://127.0.0.1:$SERVE_PORT/ping" 90 "The model server" "$LOG_DIR/model-server.log"
echo "Started (PID $SERVE_PID, log: outputs/logs/model-server.log)"

step "7/7 Check the endpoint against the registry model"
python score_endpoint.py --url "http://127.0.0.1:$SERVE_PORT"

step "Done"
if $KEEP_SERVER || [ -z "$TRACKING_PID" ]; then
    echo "MLflow UI: $MLFLOW_TRACKING_URI  (experiment 'wine-quality', model 'wine_quality')"
else
    echo "The tracking server will now stop. Rerun with --keep-server, or start it as in the README, to browse the UI."
fi
