import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder


from consts import (
    DATA_TAGS,
    SELECTED_FEATURES,
    ONE_HOT_ENCODE,
    MULTI_HOT_ENCODE,
    BOOL_ENCODE,
    REVIEW_SCORES,
    FILLNA_ZERO,
)


def load_data(relative_path: list):
    workdir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(workdir, *relative_path)
    files = os.listdir(data_path)

    dataframes = {}

    for file in files:
        dataframes[file.split(".")[0]] = pd.read_csv(
            os.path.join(data_path, file), sep=","
        )

    return dataframes


def filter_data(dataframes):
    filtered_dataframes = {}
    selected_dataframes = {key: dataframes[key] for key in DATA_TAGS}
    print(selected_dataframes)
    for name in selected_dataframes:
        filtered_dataframes[name] = selected_dataframes[name][SELECTED_FEATURES[name]]
    return filtered_dataframes


def process_data(merged_df):
    # only those listings that have reservation history in calendar can be analysed
    merged_df = merged_df.dropna(subset=["listing_id"])
    merged_df["listing_id"] = merged_df["listing_id"].astype(int)

    merged_df[["price", "reg_price"]] = (
        merged_df[["price", "reg_price"]]
        .str.replace("$", "")
        .str.replace(",", "")
        .astype(float)
    )

    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df["year"] = merged_df["year"].dt.year.astype(int)
    merged_df["month"] = merged_df["month"].dt.month.astype(int)
    merged_df["day"] = merged_df["day"].dt.day.astype(int)
    # season as year quarter
    merged_df["year_quarter"] = merged_df["date"].dt.quarter.astype(int)

    merged_df[BOOL_ENCODE] = merged_df[BOOL_ENCODE].replace({"f": 0, "t": 1}).fillna(0)

    merged_df["review_scores_rating"] = merged_df["review_scores_rating"].fillna(
        merged_df[REVIEW_SCORES].mean(axis=1)  # by columns - for same row
    )
    merged_df["review_scores_rating"] = merged_df["review_scores_rating"].fillna(0)
    merged_df = merged_df.drop(columns=REVIEW_SCORES)

    merged_df[FILLNA_ZERO] = merged_df[FILLNA_ZERO].fillna(0)

    mlb = MultiLabelBinarizer()
    col_encoded = mlb.fit_transform(merged_df[MULTI_HOT_ENCODE])
    encoded_df = pd.DataFrame(col_encoded, columns=mlb.classes_, index=merged_df.index)

    return pd.concat([merged_df.drop(columns=MULTI_HOT_ENCODE), encoded_df], axis=1)


def prep_pipeline(df):
    num_cols = df.columns[~ONE_HOT_ENCODE + MULTI_HOT_ENCODE + ["listing_id"]]

    numerical_pipeline = Pipeline(
        steps=[
            # for Imputer can be also tested strategy: mean, median
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    onehot_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def get_data(relative_path):
    dataframes = filter_data(load_data(relative_path))

    listings_df: pd.DataFrame = dataframes["listings"]
    listings_df.rename(columns={"id": "listing_id", "price": "reg_price"})

    calendar_df: pd.DataFrame = dataframes["calendar"]

    merged_df = calendar_df.merge(
        listings_df, left_index=True, right_index=True, how="outer"
    )

    df = process_data(merged_df)
    # pipeline = prep_pipeline(df)

    return df


if __name__ == "__main__":
    relative_path = ["..", "data", "v3"]
    df = get_data(relative_path)
    print(df.head())
