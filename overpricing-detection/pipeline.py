import os
import joblib

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from consts import ONE_HOT_ENCODE
from features_extraction import process_data
from models import BaseConfig, AdvancedConfig, get_models


def prep_data(train_df: pd.DataFrame):
    train_df = df.drop(columns="listing_id")
    y = train_df["is_overpriced"]
    X = train_df.drop(columns="is_overpriced")
    return X, y


def prep_pipeline(X):
    num_cols = [column for column in X.columns if column not in ONE_HOT_ENCODE]

    numerical_pipeline = Pipeline(
        steps=[
            # for Imputer can be also tested strategy: mean, median
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


def run_train_pipeline(models, df, save_path=""):
    X, y = prep_data(df)
    preprocessor = prep_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model_name in models:
        model = models[model_name]
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )
        pipeline.fit(X_train, y_train)

        y_proba = run_eval_pipeline(pipeline, model_name, X_test)
        res_auc = roc_auc_score(y_test, y_proba)

        print(f"Model: {model_name} | ROC-AUC: {res_auc:.4f}\n")

        joblib.dump(pipeline, os.path.join(save_path, f"{model_name}_v1.pkl"))


def run_eval_pipeline(pipeline, pipeline_name, X):
    if pipeline_name == "baseline_unsupervised_True":
        y_proba = -pipeline.decision_function(X)
    else:
        y_proba = pipeline.predict_proba(X)[:, 1]
    return y_proba


if __name__ == "__main__":
    relative_path = ["..", "data"]
    workdir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(workdir, *relative_path)

    df = process_data(relative_path + ["v3"])
    baseline_config = BaseConfig(contamination=0.1)
    advanced_config = AdvancedConfig(learning_rate=0.01)
    models = get_models(baseline_config, advanced_config)

    run_train_pipeline(models=models, df=df, save_path=data_path)
