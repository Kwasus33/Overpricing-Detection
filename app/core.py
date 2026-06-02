import json
import os
import random
from datetime import datetime
from typing import Any

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOG_PATH = os.path.join(DATA_DIR, "predictions.log")


def pick_model(model_names: list[str]) -> str:
    return random.choice(model_names)


NUMERIC_OFFER_COLS = ("accommodates", "bathrooms", "bedrooms", "beds")


def offer_to_dataframe(offer_dict: dict[str, Any]) -> pd.DataFrame:
    row = dict(offer_dict)
    date: datetime = row.pop("date")
    row.pop("price", None)  # target leakage: price is what we predict
    row["year"] = date.year
    row["month"] = date.month
    row["day"] = date.day
    row["year_quarter"] = (date.month - 1) // 3 + 1
    df = pd.DataFrame([row])
    # Force float dtype so the imputer sees NaN (not None-as-object) for omitted fields
    for col in NUMERIC_OFFER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def log_prediction(record: dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
