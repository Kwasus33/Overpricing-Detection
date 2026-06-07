import hashlib
import json
import os
import random
from datetime import datetime
from typing import Any

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOG_PATH = os.path.join(DATA_DIR, "predictions.log")


def pick_model(model_names: list[str], user_id: str | None = None) -> str:
    if user_id is None:
        return random.choice(model_names)
    digest = hashlib.md5(user_id.encode()).hexdigest()
    bucket = int(digest, 16) % len(model_names)
    return sorted(model_names)[bucket]


NUMERIC_OFFER_COLS = ("accommodates", "bathrooms", "bedrooms", "beds")


def offer_to_dataframe(offer_dict: dict[str, Any]) -> pd.DataFrame:
    row = dict(offer_dict)
    date: datetime = row.pop("date")
    row.pop("price", None)
    row.pop("user_id", None)
    row["year"] = date.year
    row["month"] = date.month
    row["day"] = date.day
    row["year_quarter"] = (date.month - 1) // 3 + 1
    df = pd.DataFrame([row])
    for col in NUMERIC_OFFER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def log_prediction(record: dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
