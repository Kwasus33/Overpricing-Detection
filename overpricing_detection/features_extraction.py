import os

import numpy as np
import pandas as pd


from .consts import (
    DATA_TAGS,
    SELECTED_FEATURES,
)


def load_data(relative_path: list):
    workdir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(workdir, *relative_path)
    files = os.listdir(data_path)

    dataframes = {}

    for file in files:
        dataframes[file.split(".")[0]] = pd.read_csv(os.path.join(data_path, file), sep=",")

    return dataframes


def filter_data(dataframes):
    filtered_dataframes = {}
    selected_dataframes = {key: dataframes[key] for key in DATA_TAGS}
    for name in selected_dataframes:
        filtered_dataframes[name] = selected_dataframes[name][SELECTED_FEATURES[name]]
    return filtered_dataframes


def process_features(merged_df):
    # only those listings that have reservation history in calendar can be analysed
    merged_df = merged_df.dropna(subset=["listing_id"])
    merged_df["listing_id"] = merged_df["listing_id"].astype(int)

    merged_df["price"] = merged_df["price"].str.replace("$", "").str.replace(",", "").astype(float)

    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df = merged_df.dropna(subset=["date"])

    merged_df["year"] = merged_df["date"].dt.year.astype(int)
    merged_df["month"] = merged_df["date"].dt.month.astype(int)
    merged_df["day"] = merged_df["date"].dt.day.astype(int)
    merged_df["year_quarter"] = merged_df["date"].dt.quarter.astype(int)

    merged_df.drop(columns=["date"], inplace=True)

    merged_df = merged_df.dropna(subset=["price"])
    merged_df = merged_df[merged_df["price"] > 0]
    merged_df["log_price"] = np.log(merged_df["price"])

    return merged_df


def process_data(relative_path):
    dataframes = filter_data(load_data(relative_path))

    listings_df: pd.DataFrame = dataframes["listings"]

    calendar_df: pd.DataFrame = dataframes["calendar"]

    merged_df = calendar_df.merge(listings_df, left_on="listing_id", right_on="id", how="inner")
    merged_df.drop(columns=["id"], inplace=True)
    df = process_features(merged_df)

    return df
