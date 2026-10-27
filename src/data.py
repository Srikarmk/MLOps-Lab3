from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 123


def load_wine_data():
    # The two files use different separators
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=",")

    red["is_red"] = 1
    white["is_red"] = 0
    data = pd.concat([red, white], axis=0, ignore_index=True)
    data = data.rename(columns=lambda c: c.replace(" ", "_"))

    # A wine is "high quality" if it scored 7 or more
    data["quality"] = (data["quality"] >= 7).astype(int)
    return data


def split(data):
    X = data.drop(columns=["quality"])
    y = data["quality"]
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test
