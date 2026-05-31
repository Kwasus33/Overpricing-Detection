import os

import pandas as pd

from consts import DATA_TAGS, SELECTED_FEATURES


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
    return selected_dataframes
    # for name, df in selected_dataframes.items():
    #     filtered_dataframes[name] = df[[SELECTED_FEATURES[name]]]
    # return filtered_dataframes


if __name__ == "__main__":
    relative_path = ["..", "..", "data", "v3"]
    dataframes = filter_data(load_data(relative_path))
    print(dataframes)
