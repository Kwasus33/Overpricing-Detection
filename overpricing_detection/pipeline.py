import os
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .consts import ONE_HOT_ENCODE
from .features_extraction import process_data
from .models import AdvancedConfig, get_models

DATA_VERSION = "v4"


def prep_pipeline(X):
    num_cols = [column for column in X.columns if column not in ONE_HOT_ENCODE]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("selector", VarianceThreshold()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, ONE_HOT_ENCODE),
        ]
    )

    return preprocessor


def listing_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Split by listing_id: a listing's rows are entirely in train or entirely in val."""
    rng = pd.Series(df["listing_id"].unique()).sample(frac=1, random_state=seed)
    n_val = int(len(rng) * val_frac)
    val_ids = set(rng.iloc[:n_val])
    val_mask = df["listing_id"].isin(val_ids)
    y = df["log_price"]
    X = df.drop(columns=["log_price", "price", "listing_id"])
    return X[~val_mask], X[val_mask], y[~val_mask], y[val_mask]


def run_train_pipeline(models, df, save_path=""):
    X_train, X_val, y_train, y_val = listing_split(df, val_frac=0.2)
    preprocessor = prep_pipeline(X_train)

    for model_name in models:
        model = models[model_name]
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
        pipeline.fit(X_train, y_train)

        y_pred = run_eval_pipeline(pipeline, X_val)
        res_mae_log = mean_absolute_error(y_val, y_pred)
        res_mae_price = mean_absolute_error(np.exp(y_val), np.exp(y_pred))

        print(f"Model: {model_name} | MAE(log_price): {res_mae_log:.4f} | MAE(price): {res_mae_price:.2f}\n")

        joblib.dump(pipeline, os.path.join(save_path, f"{model_name}_v1.pkl"))


def run_eval_pipeline(pipeline, X):
    return pipeline.predict(X)


if __name__ == "__main__":
    relative_path = ["..", "data"]
    workdir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(workdir, *relative_path)

    df = process_data(relative_path + [DATA_VERSION])
    advanced_config = AdvancedConfig(learning_rate=0.01)
    models = get_models(advanced_config)

    run_train_pipeline(models=models, df=df, save_path=data_path)
