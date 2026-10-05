"""Task 5: split and preprocess."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from . import config


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test.

    Stratified on config.TARGET, seeded with config.SEED.
    Neither the target nor 'mag' may remain in X.
    """

    # Target variable
    y = df[config.TARGET]

    # Remove target and magnitude from input features
    X = df.drop(columns=[config.TARGET, "mag"], errors="ignore")

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.SEED,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(scaler: str = "robust") -> ColumnTransformer:
    """ColumnTransformer over config.NUMERIC and config.NOMINAL.

    numeric: median imputation, then a scaler chosen by name
             ('standard', 'minmax', 'robust')
    nominal: most-frequent imputation, then one-hot
             (handle_unknown='ignore')
    """

    # Select numeric scaler
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    else:
        raise ValueError(
            "scaler must be 'standard', 'minmax', or 'robust'"
        )

    # Numeric pipeline
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_obj),
        ]
    )

    # Categorical pipeline
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC),
            ("nominal", nominal_pipeline, config.NOMINAL),
        ]
    )

    return preprocessor