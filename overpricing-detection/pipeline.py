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
from models import Baseline, Advanced, get_models


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


def run_pipeline(models, df):
    X, y = prep_data(df)
    preprocessor = prep_pipeline(X)

    # counter = 0
    # for val in y:
    #     if val != 0:
    #         counter += 1
    # print(y)
    # print(counter)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model_name in models:
        model = models[model_name]
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )
        pipeline.fit(X_train, y_train)
        if model_name == "Isolation Forest":
            y_proba = -pipeline.decision_function(X_test)
        else:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        res_auc = roc_auc_score(y_test, y_proba)

        print(f"Model: {model_name} | ROC-AUC: {res_auc:.4f}\n")


if __name__ == "__main__":
    relative_path = ["..", "data", "v3"]
    df = process_data(relative_path)

    baseline_config = Baseline(contamination=0.1)
    advanced_config = Advanced(learning_rate=0.01)
    models = get_models(baseline_config, advanced_config)

    run_pipeline(models, df)
