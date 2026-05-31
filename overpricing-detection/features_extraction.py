import os
import re

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


from consts import (
    DATA_TAGS,
    SELECTED_FEATURES,
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
    for name in selected_dataframes:
        filtered_dataframes[name] = selected_dataframes[name][SELECTED_FEATURES[name]]
    return filtered_dataframes


def process_features(merged_df, encode_multi_hot=False):
    # only those listings that have reservation history in calendar can be analysed
    merged_df = merged_df.dropna(subset=["listing_id"])
    merged_df["listing_id"] = merged_df["listing_id"].astype(int)

    for label in ["price", "reg_price"]:
        merged_df[label] = (
            merged_df[label].str.replace("$", "").str.replace(",", "").astype(float)
        )

    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df = merged_df.dropna(subset=["date"])

    merged_df["year"] = merged_df["date"].dt.year.astype(int)
    merged_df["month"] = merged_df["date"].dt.month.astype(int)
    merged_df["day"] = merged_df["date"].dt.day.astype(int)
    # season as year quarter
    merged_df["year_quarter"] = merged_df["date"].dt.quarter.astype(int)

    merged_df[BOOL_ENCODE] = merged_df[BOOL_ENCODE].replace({"f": 0, "t": 1}).fillna(0)

    merged_df["review_scores_rating"] = merged_df["review_scores_rating"].fillna(
        merged_df[REVIEW_SCORES].mean(axis=1)  # by columns - for same row
    )
    merged_df["review_scores_rating"] = merged_df["review_scores_rating"].fillna(0)

    merged_df[FILLNA_ZERO] = merged_df[FILLNA_ZERO].fillna(0)

    merged_df.drop(columns=REVIEW_SCORES + ["date"], inplace=True)

    if encode_multi_hot:
        merged_df[MULTI_HOT_ENCODE] = (
            merged_df[MULTI_HOT_ENCODE]
            .fillna("")
            .apply(lambda s: re.findall(r'"([^"]+)"', s) if isinstance(s, str) else [])
        )
        mlb = MultiLabelBinarizer()
        col_encoded = mlb.fit_transform(merged_df[MULTI_HOT_ENCODE])
        encoded_df = pd.DataFrame(
            col_encoded, columns=mlb.classes_, index=merged_df.index
        )
        merged_df = pd.concat([merged_df, encoded_df], axis=1)

    merged_df.drop(columns=MULTI_HOT_ENCODE, inplace=True)

    return merged_df


def label_data(df):
    """
    Strategy: if price in given day is higher than mean + std of all listings in same area and of same room type from current month
    it's labeled as overpriced
    """
    group_by_cols = [
        "year",
        "month",
        "neighbourhood_group_cleansed",
        "room_type",
    ]

    mean = df.groupby(group_by_cols)["price"].transform("mean")
    std = df.groupby(group_by_cols)["price"].transform("std")
    std = std.fillna(0)  # if there's only one listing after grouping

    price_threshold = mean + 0.5 * std
    is_overpriced = (df["price"] > price_threshold) & (df["available"] == 1)
    df["is_overpriced"] = is_overpriced.astype(int)

    return df


def process_data(relative_path):
    dataframes = filter_data(load_data(relative_path))

    listings_df: pd.DataFrame = dataframes["listings"]
    listings_df.rename(columns={"price": "reg_price"}, inplace=True)

    calendar_df: pd.DataFrame = dataframes["calendar"]

    merged_df = calendar_df.merge(
        listings_df, left_index=True, right_index=True, how="outer"
    )
    df = process_features(merged_df)
    labeled_df = label_data(df)

    return labeled_df
