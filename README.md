# MLflow Lab: Wine Quality Prediction

This lab trains a model to predict whether a wine is high quality (score of 7 or more) from 11 chemical measurements, using the [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) red and white datasets. The focus is MLflow: tracking experiments, tuning with nested runs, managing a model registry, batch inference and serving the model over REST.

It is based on the course's MLflow Lab2 (`starter.ipynb`, a Databricks notebook that needs Spark). The original write-up is kept unchanged in [ORIGINAL_LAB.md](ORIGINAL_LAB.md), along with the original notebook and PDF. This version runs locally with no Spark or Databricks. It replaces the deprecated registry stages with aliases and an automatic promotion check, and it verifies the served model. [What changed](#what-changed-from-the-original-lab) lists every change.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the whole lab in one command](#run-the-whole-lab-in-one-command)
- [Run it step by step](#run-it-step-by-step)
- [What a successful run looks like](#what-a-successful-run-looks-like)
- [What changed from the original lab](#what-changed-from-the-original-lab)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

You need three things: Git, curl, and **Python 3.12 or 3.13**. Python 3.12 is the minimum because numpy 2.5 and xgboost 3.4 require it. The lab was tested on Python 3.12.15 and 3.13.16. Java, Spark, Docker and conda are **not** needed.

**Python**

| OS | Install |
|---|---|
| macOS | Install [Homebrew](https://brew.sh), then `brew install python@3.12` |
| Ubuntu 24.04 / Debian | `sudo apt update && sudo apt install -y python3.12 python3.12-venv git curl` |
| Windows 10/11 | `winget install Python.Python.3.12`, or the installer from [python.org](https://www.python.org/downloads/) (tick "Add python.exe to PATH") |
| Any OS, with [uv](https://docs.astral.sh/uv/) | `uv python install 3.12` |

Check it:

```bash
python3.12 --version
```

This should print `Python 3.12.x`. On Windows, run `py -3.12 --version` instead.

**Git and curl:** both come with macOS (if they're missing, `xcode-select --install` adds Git). On Windows, Git comes from `winget install Git.Git`, and curl ships with Windows 10 and later.

**Python packages.** These are pinned in `requirements.txt` and installed in [Setup](#setup):

| Package | Version | Used for |
|---|---|---|
| mlflow | 3.17.1 | tracking server, registry, serving |
| scikit-learn | 1.9.1 | random forest baseline, metrics, data split |
| xgboost | 3.4.1 | tuned model |
| hyperopt | 0.3.0 | TPE hyperparameter search |
| pandas | 3.0.6 | data loading |
| numpy | 2.5.4 | arrays, seeding |
| matplotlib | 3.11.2 | plots logged to MLflow |
| joblib | 1.6.0 | saving the random forest inside its MLflow model |
| requests | 2.34.2 | REST client for the served model |

## Setup

```bash
git clone https://github.com/Srikarmk/MLOps-Lab3.git
cd MLOps-Lab3

python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the install:

```bash
python -c "import mlflow, xgboost, sklearn; print(mlflow.__version__, xgboost.__version__, sklearn.__version__)"
```

This should print `3.17.1 3.4.1 1.9.1`.

## Run the whole lab in one command

On macOS, Linux or WSL:

```bash
./run_lab.sh
```

`run_lab.sh` does all of the following:
- It creates `.venv` and installs the requirements if they aren't there yet, so you can skip Setup.
- It starts the tracking server on port 8080, or reuses one that's already running.
- It runs the seven steps below in order, then stops the servers it started.
- It stops at the first failure.

A full run takes about 2 minutes on a laptop, including the install.

| Option | Effect |
|---|---|
| `./run_lab.sh --fresh` | deletes `mlflow.db`, `mlartifacts/` and `outputs/` first, so the registry starts empty and the output matches the section below |
| `./run_lab.sh --keep-server` | leaves the tracking server running so you can browse http://127.0.0.1:8080 afterwards |
| `MAX_EVALS=96 ./run_lab.sh` | number of hyperopt trials (default 32) |
| `TRACKING_PORT=8090 SERVE_PORT=5002 ./run_lab.sh` | use other ports |
| `PYTHON=/path/to/python3.12 ./run_lab.sh` | choose the interpreter used to create `.venv` |

On Windows without WSL, follow the step-by-step commands below in PowerShell. They're the same, apart from activating the venv and running `serve_model.sh`; for that step, see [Troubleshooting](#troubleshooting).

## Run it step by step

The step-by-step run uses three terminals. Activate the venv in each one (`source .venv/bin/activate`), from the repository root.

**Terminal 1: the tracking server.** It stores runs and the model registry in `mlflow.db` (SQLite) and model files in `mlartifacts/`. The registry needs a database backend; it doesn't work with plain files.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 8080
```

Leave it running and open http://127.0.0.1:8080. The scripts find the server through `MLFLOW_TRACKING_URI`, which defaults to `http://127.0.0.1:8080`.

**Terminal 2: training, registry and batch inference.**

```bash
cd src
python train_baseline.py                 # 1. random forest baseline
python register_model.py baseline        # 2. register it; it becomes @champion if it beats the current one
python tune_xgboost.py --max-evals 32    # 3. seeded hyperopt sweep, one nested run per trial
python register_model.py best-xgboost    # 4. register the sweep's best model, with the same promotion check
python batch_inference.py                # 5. score the test set with models:/wine_quality@champion
```

**Terminal 3: serve the champion over REST.**

```bash
./serve_model.sh                         # 6. http://127.0.0.1:5001/invocations
```

**Back in terminal 2: verify the endpoint.**

```bash
python score_endpoint.py                 # 7. the served predictions must match the registry model
```

Stop the servers with Ctrl+C in terminals 1 and 3. Your runs and models stay in `mlflow.db` and `mlartifacts/`; delete those two (and `outputs/`) to start over.

### Registry commands

```bash
python register_model.py status          # every version with its metrics, promotion result and aliases
python register_model.py rollback        # point @champion back at @previous-champion
python register_model.py best-xgboost --min-improvement 0.01   # require a bigger gain to promote
```

### Calling the endpoint yourself

The server accepts a pandas DataFrame as JSON, in either `dataframe_split` or `dataframe_records` form. It returns `P(high quality)` for each row:

```bash
curl -s http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_records": [{
  "fixed_acidity": 7.0, "volatile_acidity": 0.27, "citric_acid": 0.36, "residual_sugar": 20.7,
  "chlorides": 0.045, "free_sulfur_dioxide": 45.0, "total_sulfur_dioxide": 170.0, "density": 1.001,
  "pH": 3.0, "sulphates": 0.45, "alcohol": 8.8, "is_red": 0.0}]}'
```

The reply looks like `{"predictions": [0.0...]}`. If you leave a column out, the model's signature rejects the request with HTTP 400.

## What a successful run looks like

Here is the output of `./run_lab.sh --fresh` on an empty registry, trimmed. The numbers are reproducible: the data split, the random forest and the hyperopt search are all seeded with 123.

```text
==> 1/7 Baseline random forest
Baseline random forest: val AUC = 0.8717, test AUC = 0.8539

==> 2/7 Register the baseline (promotion check against @champion)
Registered wine_quality v1 (random_forest, run ...) as @challenger
Promoted v1 to @champion (no champion yet)

==> 3/7 Hyperopt sweep over XGBoost (32 trials, one nested run each)
Sweep of 32 trials (run ...)
  best trial 14: val AUC = 0.8972, test AUC = 0.9018

==> 4/7 Register the sweep's best model (promotion check against @champion)
  challenger v2 val AUC = 0.8972
  champion   v1 val AUC = 0.8717  (gain +0.0255, need > +0.0000)
Promoted v2 to @champion (val AUC 0.8972 beats v1 (0.8717))

version  family         val_auc  test_auc  check     aliases
v1       random_forest  0.8717   0.8539    promoted  @previous-champion
v2       xgboost        0.8972   0.9018    promoted  @challenger @champion

==> 5/7 Batch inference with models:/wine_quality@champion
Loaded models:/wine_quality@champion (v2): test AUC = 0.9018
Wrote 1300 predictions to .../outputs/batch_predictions.csv and logged them to the run

==> 7/7 Check the endpoint against the registry model
Endpoint http://127.0.0.1:5001/invocations, expected models:/wine_quality@champion (v2, xgboost)
  batch of 1300 rows: 14 ms, test AUC = 0.9018
  served vs local predictions: max abs diff = 0.00e+00 -> match
  50 single-row requests: p50 = 3.9 ms, p95 = 4.5 ms
  request missing 10 of 12 columns: HTTP 400 -> rejected
Endpoint check passed.
```

Latencies depend on your machine. Everything else should match. The original notebook's baseline scored a test AUC of 0.854, which the ported baseline reproduces.

If you run it again without `--fresh`, the new versions go through the same promotion check against the current champion:
- The repeat baseline (val AUC 0.8717) is registered and rejected, so `@champion` doesn't move.
- The repeat sweep finds the same configuration. It has the same validation AUC, not a higher one, so it is rejected as well.

**In the MLflow UI** (http://127.0.0.1:8080):

- **Experiments → wine-quality** has a description and tags. Its runs are:
  - `untuned_random_forest`
  - `xgboost_models`, which expands into 32 `trial-NNN` child runs
  - `batch_inference`
  - `endpoint_check`

  Filter them with tags, e.g. `tags.pipeline_step = 'tuning_trial'`.
- **A training run's** Overview shows its params, its tags (`pipeline_step`, `model_family`, `data_version`) and its training/validation/test dataset inputs. Its Model metrics show `val_*` and `test_*` AUC, log loss, accuracy, precision, recall and F1.
  - Artifacts holds `plots/` (ROC curves, confusion matrices, feature importance) and the logged model.
  - On `xgboost_models` there is also `sweep/`: the search space, a table of every trial, the optimisation history and validation AUC by each hyperparameter.
- **Models → wine_quality** has a description. Each version carries tags (family, data version, val/test AUC, `promotion_check`), a description saying why it was promoted or rejected, and its aliases.

## What changed from the original lab

| Area | Original lab | This version | MLflow concept |
|---|---|---|---|
| Environment | Databricks notebook; needs Spark (`SparkTrials`, `spark_udf`); unpinned requirements apart from xgboost 1.7.6 | Python scripts in `src/` against a local `mlflow server` with a SQLite backend and an artifact store; every package pinned | Tracking server, backend and artifact stores |
| Tracking | One `auc` metric per run, written into the default experiment | A named experiment with a description and tags. Each run is tagged with its pipeline step, model family and a hash of the data files. It logs the train/val/test splits as dataset inputs, val and test metrics, ROC curves, confusion matrices and feature importance | Experiments, tags, dataset lineage, logged artifacts |
| Baseline model | A wrapper class defined in the notebook and saved with cloudpickle | The model is defined in `wine_model.py` and logged from that file ("models from code"), with the forest attached as an artifact. It has a signature and an input example | pyfunc models from code, signatures |
| Hyperparameter tuning | 96 unseeded `SparkTrials` trials; autolog saves a full model for every trial | A seeded TPE search: the same seed gives the same best parameters. Each trial is a nested run that logs only its params and metrics. The parent run gets a summary of the sweep pulled back with `search_runs`, and the best configuration is refit once and logged as the sweep's single model | Nested runs, `search_runs`, reproducibility |
| Registry | Versions moved by hand to the `Production` / `Archived` stages, which MLflow 3 deprecates. The new model is promoted without checking whether it is better | `@champion`, `@challenger` and `@previous-champion` aliases. A new version is promoted only if its validation AUC beats the champion's, both loaded back from the registry. Test AUC is recorded but never used to choose, so the test set stays a fair estimate. The decision is recorded on the version. `rollback` undoes a promotion, and re-registering the same run is refused | Model aliases, version tags and descriptions, promotion gate |
| Batch inference | `spark_udf` on `models:/wine_quality/production` | Loads `models:/wine_quality@champion` in a tracked run, which records the model version and source run and saves the predictions as an artifact | Inference lineage |
| Serving | `mlflow models serve` on the `production` stage; the response is printed | `serve_model.sh` serves `@champion`. `score_endpoint.py` checks that the served predictions match the same registry version loaded locally, measures latency, confirms that malformed requests are rejected by the signature, and logs the check as a run | Model serving, signature enforcement |
| Automation | Manual notebook cells | `run_lab.sh` runs everything from install to endpoint check and fails loudly at the first error | Reproducible pipeline |

## Project layout

```text
MLOps-Lab3/
├── run_lab.sh              whole lab end to end
├── serve_model.sh          mlflow models serve for a registry alias
├── requirements.txt        pinned dependencies
├── data/                   UCI wine quality CSVs and the dataset description
├── src/
│   ├── common.py           tracking URI, experiment and model names
│   ├── data.py             loading, target definition, seeded 60/20/20 split, data version hash
│   ├── tracking.py         helpers for tags, dataset inputs, metrics and plots
│   ├── wine_model.py       pyfunc wrapper for the random forest (models from code)
│   ├── train_baseline.py   step 1
│   ├── register_model.py   steps 2 and 4, plus status and rollback
│   ├── tune_xgboost.py     step 3
│   ├── batch_inference.py  step 5
│   └── score_endpoint.py   step 7
├── starter.ipynb           original notebook (unchanged)
├── ORIGINAL_LAB.md         original write-up (unchanged)
└── Wine Quality Prediction Lab Documentation.pdf
```

Files created at run time are git-ignored: `.venv/`, `mlflow.db`, `mlartifacts/`, and `outputs/`, which holds the batch predictions and the `run_lab.sh` server logs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No MLflow tracking server at http://127.0.0.1:8080` | Start the tracking server (terminal 1 in [Run it step by step](#run-it-step-by-step)), or use `./run_lab.sh` |
| Why port 8080 and not MLflow's default 5000? | macOS uses port 5000 for AirPlay Receiver |
| `uvicorn: not found` from `mlflow models serve` | The venv isn't active. Run `source .venv/bin/activate`, or use `./serve_model.sh`, which activates it |
| `score_endpoint.py` reports `MISMATCH` | The server is serving a different version from the alias you're checking. The alias is resolved once, when the server starts, so restart `./serve_model.sh` after a promotion or rollback |
| `Port 5001 is already in use` | An old model server is still running. Stop it, or use `SERVE_PORT=5002 ./run_lab.sh` |
| `Run ... is already registered` | That run already has a version. Train a new one first (`train_baseline.py` or `tune_xgboost.py`) |
| `pip install` can't find numpy 2.5.4 or xgboost 3.4.1 | The venv uses Python older than 3.12. Delete `.venv` and recreate it with `python3.12 -m venv .venv` |
| `./run_lab.sh: Permission denied` | `chmod +x run_lab.sh serve_model.sh`, or run `bash run_lab.sh` |
| Windows: serving without bash | In an activated PowerShell venv, set `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:8080"`, then run `mlflow models serve -m "models:/wine_quality@champion" --env-manager local -h 127.0.0.1 -p 5001` |
