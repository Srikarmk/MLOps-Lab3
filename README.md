# MLflow Lab: Adult Income Prediction

This lab predicts whether a person earns more than $50K a year from census fields such as age, education, occupation and hours worked. It uses the [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) dataset, which comes from the 1994 US census. The focus is MLflow:
- tracking experiments and dataset lineage
- tuning with nested runs
- a model registry with an automatic promotion check
- batch inference
- serving the model over REST
- reporting metrics per group for sensitive attributes (sex and race)

It is based on the course's MLflow Lab2, a Databricks notebook (`starter.ipynb`) that predicts wine quality with a random forest and XGBoost and needs Spark. The original write-up, notebook, PDF and wine data are all kept unchanged; see [ORIGINAL_LAB.md](ORIGINAL_LAB.md).

This version changes four things:
- It runs locally, with no Spark or Databricks.
- It uses a different dataset and different models (logistic regression and LightGBM), with the preprocessing stored inside each logged model.
- It replaces the deprecated registry stages with aliases and a promotion check.
- It verifies the served model.

[What changed](#what-changed-from-the-original-lab) lists every change.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the whole lab in one command](#run-the-whole-lab-in-one-command)
- [Run it step by step](#run-it-step-by-step)
- [Data and models](#data-and-models)
- [What a successful run looks like](#what-a-successful-run-looks-like)
- [What changed from the original lab](#what-changed-from-the-original-lab)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

You need Git, curl, **Python 3.12 or 3.13**, and on macOS the OpenMP runtime that LightGBM needs. Python 3.12 is the minimum because numpy 2.5 requires it. The lab was tested on Python 3.12.15 and 3.13.16. Java, Spark, Docker and conda are **not** needed.

**Python and the system libraries**

| OS | Install |
|---|---|
| macOS | Install [Homebrew](https://brew.sh), then `brew install python@3.12 libomp` |
| Ubuntu 24.04 / Debian | `sudo apt update && sudo apt install -y python3.12 python3.12-venv libgomp1 git curl` |
| Windows 10/11 | `winget install Python.Python.3.12`, or the installer from [python.org](https://www.python.org/downloads/) (tick "Add python.exe to PATH") |
| Any OS, with [uv](https://docs.astral.sh/uv/) | `uv python install 3.12` (on macOS you still need `brew install libomp`) |

`libomp` (macOS) and `libgomp1` (Linux) are the OpenMP runtime that LightGBM's compiled library links against. Without it, `import lightgbm` fails; `run_lab.sh` checks for this and prints the fix.

Check Python:

```bash
python3.12 --version
```

This should print `Python 3.12.x`. On Windows, run `py -3.12 --version` instead.

**Git and curl:** both come with macOS (if they're missing, `xcode-select --install` adds Git). On Windows, Git comes from `winget install Git.Git`, and curl ships with Windows 10 and later.

**Python packages.** These are pinned in `requirements.txt` and installed in [Setup](#setup):

| Package | Version | Used for |
|---|---|---|
| mlflow | 3.17.1 | tracking server, registry, serving |
| scikit-learn | 1.9.1 | logistic regression, preprocessing pipelines, metrics, data split |
| lightgbm | 4.7.0 | tuned model |
| hyperopt | 0.3.0 | TPE hyperparameter search |
| pandas | 3.0.6 | data loading |
| numpy | 2.5.4 | arrays, seeding |
| matplotlib | 3.11.2 | plots logged to MLflow |
| joblib | 1.6.0 | saving each fitted pipeline inside its MLflow model |
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
python -c "import mlflow, lightgbm, sklearn; print(mlflow.__version__, lightgbm.__version__, sklearn.__version__)"
```

This should print `3.17.1 4.7.0 1.9.1`.

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
python train_baseline.py                 # 1. logistic regression baseline
python register_model.py baseline        # 2. register it; it becomes @champion if it beats the current one
python tune_lightgbm.py --max-evals 32   # 3. seeded hyperopt sweep over LightGBM, one nested run per trial
python register_model.py best-lightgbm   # 4. register the sweep's best model, with the same promotion check
python batch_inference.py                # 5. score the test set with models:/adult_income@champion
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
python register_model.py best-lightgbm --min-improvement 0.01   # require a bigger gain to promote
```

### Calling the endpoint yourself

The server accepts a pandas DataFrame as JSON, in either `dataframe_split` or `dataframe_records` form. You send the raw census values; the encoding happens inside the model. This request scores three people:
1. A row from the test set.
2. A married executive with a large capital gain.
3. The same executive, but with a workclass and country the model has never seen.

```bash
curl -s http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_records": [
  {"age": 25, "education_num": 7, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
   "workclass": "Private", "marital_status": "Never-married", "occupation": "Machine-op-inspct",
   "relationship": "Own-child", "native_country": "United-States"},
  {"age": 45, "education_num": 14, "capital_gain": 15024, "capital_loss": 0, "hours_per_week": 50,
   "workclass": "Private", "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
   "relationship": "Husband", "native_country": "United-States"},
  {"age": 45, "education_num": 14, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 50,
   "workclass": "Space-agency", "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
   "relationship": "Husband", "native_country": "Mars"}]}'
```

With the LightGBM champion the reply is `{"predictions": [0.0046, 0.9999, 0.8859]}`. An unseen category doesn't cause an error: it's treated as missing.

Valid category values are listed in `data/adult/adult.names`, plus `"Unknown"` for fields that were blank in the census data. If you leave a column out, the model's signature rejects the request with HTTP 400.

## Data and models

**Data.** `data/adult/` holds `adult.data`, `adult.test` and `adult.names`, unchanged from UCI. The dataset is by Becker & Kohavi (1996) and licensed CC BY 4.0. There are 48,842 people, and 24% of them earn more than $50K.

| Split | Rows | Source |
|---|---|---|
| train | 26,048 | 80% of `adult.data`, stratified, seed 123 |
| validation | 6,513 | the other 20% of `adult.data` |
| test | 16,281 | `adult.test`, UCI's own held-out file |

The **model inputs** are 5 numbers and 5 categories:
- Numbers: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`.
- Categories: `workclass`, `marital_status`, `occupation`, `relationship`, `native_country`.

Blank values (`?` in the files) become the category `Unknown`.

Four columns are **left out of the model**:
- `fnlwgt` is a census sampling weight, not a fact about the person.
- `education` repeats `education_num` as text.
- `sex` and `race` are kept out on purpose. They're used only to report metrics per group (see below), so the served model never receives them.

**Models.** Both models are scikit-learn pipelines, preprocessing included, logged through one shared pyfunc wrapper (`proba_model.py`) that returns P(income > 50K).

| | Baseline | Tuned |
|---|---|---|
| Model | `LogisticRegression(C=1.0)` | `LGBMClassifier`, up to 2000 trees with early stopping on validation AUC |
| Preprocessing | standard-scaled numbers, one-hot categories (83 columns) | numbers unchanged, categories as integer codes that LightGBM splits on natively |
| Tuning | none | 32 hyperopt TPE trials (seed 123) over `num_leaves`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda` and `class_weight` (none or balanced) |

LightGBM runs on one thread. On a dataset this small, using all cores made it 10× slower, and a fixed thread count gives identical trees on every machine.

**Why the tuned model's scores aren't calibrated.** The sweep picks `class_weight="balanced"`, which weights the 24% minority class up. That raises recall at the 0.5 threshold (0.86 against the baseline's 0.59) at the cost of precision (0.60 against 0.73), and it pushes scores upward. Treat the scores as a ranking for AUC, not as calibrated probabilities.

**Metrics per group.** Every evaluation of the test set logs:
- `slices/test_slices.json`, a table with one row per sex and per race group: rows, actual positive rate, predicted positive rate, AUC and true positive rate.
- `test_<metric>_gap_<attribute>`, the largest difference between groups with at least 100 rows.

Keeping sex and race out of the inputs does not make the model fair, because other fields carry the same information. On the test set:

| Gap between groups | Logistic regression | LightGBM |
|---|---|---|
| AUC, by sex | 0.051 | 0.038 |
| Predicted positive rate, by sex | 0.171 | 0.283 |
| True positive rate, by sex | 0.065 | 0.080 |
| True positive rate, by race | 0.229 | 0.068 |

## What a successful run looks like

Here is the output of `./run_lab.sh --fresh` on an empty registry, trimmed. The numbers are reproducible: the split, the hyperopt search and LightGBM are all seeded with 123, and the results were identical on Python 3.12 and 3.13.

```text
==> 1/7 Baseline logistic regression
Baseline logistic regression: val AUC = 0.9021, test AUC = 0.9050

==> 2/7 Register the baseline (promotion check against @champion)
Registered adult_income v1 (logistic_regression, run ...) as @challenger
Promoted v1 to @champion (no champion yet)

==> 3/7 Hyperopt sweep over LightGBM (32 trials, one nested run each)
Sweep of 32 trials (run ...)
  best trial 27: val AUC = 0.9283, test AUC = 0.9279
  best params: {"class_weight": "balanced", "colsample_bytree": 0.5033..., "learning_rate": 0.0619..., "min_child_samples": 41, "num_leaves": 11, "reg_lambda": 2.1049..., "subsample": 0.9996...}

==> 4/7 Register the sweep's best model (promotion check against @champion)
Registered adult_income v2 (lightgbm, run ...) as @challenger
  challenger v2 val AUC = 0.9283
  champion   v1 val AUC = 0.9021  (gain +0.0262, need > +0.0000)
Promoted v2 to @champion (val AUC 0.9283 beats v1 (0.9021))

version  family               val_auc  test_auc  check     aliases
v1       logistic_regression  0.9021   0.9050    promoted  @previous-champion
v2       lightgbm             0.9283   0.9279    promoted  @challenger @champion

==> 5/7 Batch inference with models:/adult_income@champion
Loaded models:/adult_income@champion (v2): test AUC = 0.9279
Wrote 16281 predictions to .../outputs/batch_predictions.csv and logged them to the run

==> 7/7 Check the endpoint against the registry model
Endpoint http://127.0.0.1:5001/invocations, expected models:/adult_income@champion (v2, lightgbm)
  batch of 16281 rows: 315 ms, test AUC = 0.9279
  served vs local predictions: max abs diff = 0.00e+00 -> match
  50 single-row requests: p50 = 4.8 ms, p95 = 5.6 ms
  request missing 8 of 10 columns: HTTP 400 -> rejected
Endpoint check passed.
```

Latencies depend on your machine. Everything else should match.

If you run it again without `--fresh`, the new versions go through the same promotion check against the current champion:
- The repeat baseline (val AUC 0.9021) is registered and rejected, so `@champion` doesn't move.
- The repeat sweep finds the same configuration. It has the same validation AUC, not a higher one, so it is rejected as well.

**In the MLflow UI** (http://127.0.0.1:8080):

- **Experiments → adult-income** has a description and tags. Its runs are:
  - `logistic_regression_baseline`
  - `lightgbm_sweep`, which expands into 32 `trial-NNN` child runs
  - `batch_inference`
  - `endpoint_check`

  Filter them with tags, e.g. `tags.pipeline_step = 'tuning_trial'`.
- **A training run's** Overview shows its params, its tags (`pipeline_step`, `model_family`, `data_version`) and its training/validation/test dataset inputs. Its Model metrics show:
  - `val_*` and `test_*` AUC, log loss, accuracy, precision, recall and F1
  - the `test_*_gap_sex` and `test_*_gap_race` metrics
- **A training run's Artifacts** hold:
  - `plots/`: ROC curves, confusion matrices and feature importance (absolute coefficients for logistic regression, gain for LightGBM)
  - `slices/test_slices.json`
  - `feature_importance.json`
  - the logged model
  - on `lightgbm_sweep` only, `sweep/`: the search space, a table of every trial, the optimisation history and validation AUC by each hyperparameter
- **Models → adult_income** has a description. Each version carries tags (family, data version, val/test AUC, `promotion_check`), a description saying why it was promoted or rejected, and its aliases.

## What changed from the original lab

| Area | Original lab | This version | MLflow / MLOps concept |
|---|---|---|---|
| Environment | Databricks notebook; needs Spark (`SparkTrials`, `spark_udf`); unpinned requirements apart from xgboost 1.7.6 | Python scripts in `src/` against a local `mlflow server` with a SQLite backend and an artifact store; every package pinned | Tracking server, backend and artifact stores |
| Dataset | UCI Wine Quality (6,497 wines, 11 numeric measurements), random 60/20/20 split | UCI Adult (48,842 people, numbers and categories), UCI's official test file plus a stratified train/validation split. A hash of the files is logged on every run | Dataset lineage |
| Models | Random forest (10 trees) and XGBoost on raw numeric features | Logistic regression and LightGBM, each a scikit-learn pipeline with its own encoding. The served model takes raw values and copes with unseen categories | Preprocessing packaged with the model |
| Tracking | One `auc` metric per run, written into the default experiment | A named experiment with a description and tags. Each run is tagged with its pipeline step, model family and data version. It logs the splits as dataset inputs, val and test metrics, ROC curves, confusion matrices and feature importance | Experiments, tags, logged artifacts |
| Evaluation by group | None | Metrics per sex and race group on every test evaluation, plus the largest gaps as metrics. Sex and race are not model inputs | Sliced evaluation |
| Model packaging | A wrapper class defined in the notebook and saved with cloudpickle | One wrapper, `proba_model.py`, logged as code ("models from code"), with the fitted pipeline attached as an artifact. Each model has a signature and an input example | pyfunc models from code, signatures |
| Hyperparameter tuning | 96 unseeded `SparkTrials` trials; autolog saves a full model for every trial | A seeded TPE search: the same seed gives the same best parameters. Each trial is a nested run that logs only its params and metrics. The parent run gets a summary of the sweep pulled back with `search_runs`, and the best configuration is refit once and logged as the sweep's single model | Nested runs, `search_runs`, reproducibility |
| Registry | Versions moved by hand to the `Production` / `Archived` stages, which MLflow 3 deprecates. The new model is promoted without checking whether it is better | `@champion`, `@challenger` and `@previous-champion` aliases. A new version is promoted only if its validation AUC beats the champion's, both loaded back from the registry. Test AUC is recorded but never used to choose, so the test set stays a fair estimate. The decision is recorded on the version. `rollback` undoes a promotion, and re-registering the same run is refused | Model aliases, version tags and descriptions, promotion gate |
| Batch inference | `spark_udf` on `models:/wine_quality/production` | Loads `models:/adult_income@champion` in a tracked run, which records the model version and source run, logs per-group metrics and saves the predictions as an artifact | Inference lineage |
| Serving | `mlflow models serve` on the `production` stage; the response is printed | `serve_model.sh` serves `@champion`. `score_endpoint.py` checks that the served predictions match the same registry version loaded locally, measures latency, confirms that malformed requests are rejected by the signature, and logs the check as a run | Model serving, signature enforcement |
| Automation | Manual notebook cells | `run_lab.sh` runs everything from install to endpoint check and fails loudly at the first error | Reproducible pipeline |

## Project layout

```text
MLOps-Lab3/
├── run_lab.sh              whole lab end to end
├── serve_model.sh          mlflow models serve for a registry alias
├── requirements.txt        pinned dependencies
├── data/
│   ├── adult/              UCI Adult: adult.data, adult.test, adult.names (used by this lab)
│   └── winequality-*       the original lab's wine data and description (unchanged, not used)
├── src/
│   ├── common.py           tracking URI, experiment and model names, server check
│   ├── data.py             loading, target, features, UCI train/val/test split, data version hash
│   ├── features.py         one-hot and ordinal preprocessing for the two pipelines
│   ├── proba_model.py      pyfunc wrapper shared by both models (models from code)
│   ├── tracking.py         helpers for tags, dataset inputs, metrics, slices, plots and model logging
│   ├── train_baseline.py   step 1
│   ├── register_model.py   steps 2 and 4, plus status and rollback
│   ├── tune_lightgbm.py    step 3
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
| `import lightgbm` fails with `Library not loaded: ... libomp.dylib` (macOS) or `libgomp.so.1` (Linux) | Install the OpenMP runtime: `brew install libomp` on macOS, `sudo apt install libgomp1` on Ubuntu/Debian |
| Why port 8080 and not MLflow's default 5000? | macOS uses port 5000 for AirPlay Receiver |
| `uvicorn: not found` from `mlflow models serve` | The venv isn't active. Run `source .venv/bin/activate`, or use `./serve_model.sh`, which activates it |
| `score_endpoint.py` reports `MISMATCH` | The server is serving a different version from the alias you're checking. The alias is resolved once, when the server starts, so restart `./serve_model.sh` after a promotion or rollback |
| `Port 5001 is already in use` | An old model server is still running. Stop it, or use `SERVE_PORT=5002 ./run_lab.sh` |
| `Run ... is already registered` | That run already has a version. Train a new one first (`train_baseline.py` or `tune_lightgbm.py`) |
| `pip install` can't find numpy 2.5.4 | The venv uses Python older than 3.12. Delete `.venv` and recreate it with `python3.12 -m venv .venv` |
| `./run_lab.sh: Permission denied` | `chmod +x run_lab.sh serve_model.sh`, or run `bash run_lab.sh` |
| Windows: serving without bash | In an activated PowerShell venv, set `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:8080"`, then run `mlflow models serve -m "models:/adult_income@champion" --env-manager local -h 127.0.0.1 -p 5001` |
