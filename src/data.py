import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "adult"
DATA_FILES = ["adult.data", "adult.test"]
RANDOM_STATE = 123

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
TARGET = "income"
CLASS_LABELS = ["<=50K", ">50K"]
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = ["workclass", "marital_status", "occupation", "relationship", "native_country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Not model inputs: fnlwgt is a census sampling weight and education repeats education_num as text.
# sex and race are kept out of the model too, and used only to report metrics per group.
SLICE_COLUMNS = ["sex", "race"]


def load_adult_data():
    frames = []
    for name in DATA_FILES:
        # adult.test starts with a "|1x3 Cross validator" line; both files mark missing values with "?"
        frame = pd.read_csv(DATA_DIR / name, header=None, names=COLUMNS, skipinitialspace=True,
                            skiprows=1 if name == "adult.test" else 0, na_values="?")
        frame["source_file"] = name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    # adult.test writes the labels as ">50K." and "<=50K."
    data[TARGET] = (data[TARGET].str.rstrip(".") == ">50K").astype(int)

    # Missing categories become their own value, so callers of the served model can send "Unknown" too
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].fillna("Unknown")
    # Store numeric features as float so the logged model schema accepts missing values at inference time
    data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype("float64")
    return data


def data_version():
    """Short SHA-256 over the raw files, logged as a tag so every run records exactly which data it saw."""
    digest = hashlib.sha256()
    for name in DATA_FILES:
        digest.update((DATA_DIR / name).read_bytes())
    return digest.hexdigest()[:12]


def split(data):
    """UCI's own split: adult.test is the test set, adult.data is split 80/20 into train and validation."""
    train_file = data[data["source_file"] == "adult.data"]
    test_file = data[data["source_file"] == "adult.test"]
    X_train, X_val, y_train, y_val = train_test_split(
        train_file[FEATURES], train_file[TARGET], test_size=0.2, stratify=train_file[TARGET],
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, test_file[FEATURES], y_train, y_val, test_file[TARGET]


def slice_groups(data, X):
    """The sensitive attributes for the rows in X, for per-group metrics."""
    return data.loc[X.index, SLICE_COLUMNS]
