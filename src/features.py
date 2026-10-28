"""Preprocessing that is fitted on the training set and saved inside each logged model, so the served
model takes the raw columns (e.g. workclass="Private") and nobody has to repeat the encoding."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def one_hot_preprocessor():
    """For linear models: scaled numbers and one column per category."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def ordinal_preprocessor():
    """For LightGBM: numbers unchanged, categories as integer codes that LightGBM splits on natively.
    A category never seen in training becomes -1, which LightGBM treats as missing."""
    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
