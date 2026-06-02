import json
import os
import urllib.error
import urllib.request

import pandas as pd

URL = "http://localhost:8080/predict"
DATA_VERSION = "v4"
MAX_REQUESTS = 100
SEED = 42
VAL_FRAC = 0.2

OFFER_COLS = [
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
]


def post(offer: dict) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps(offer).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def parse_price(value: str) -> float:
    return float(str(value).replace("$", "").replace(",", ""))


def pick_val_listing_ids(all_ids: pd.Series, val_frac: float, seed: int) -> set:
    shuffled = pd.Series(all_ids.unique()).sample(frac=1, random_state=seed)
    n_val = int(len(shuffled) * val_frac)
    return set(shuffled.iloc[:n_val])


def build_offer(cal_row: pd.Series, lst_row: pd.Series) -> dict:
    offer = {
        "date": str(cal_row["date"]),
        "price": parse_price(cal_row["price"]),
    }
    for col in OFFER_COLS:
        value = lst_row[col]
        if pd.isna(value):
            continue
        offer[col] = value.item() if hasattr(value, "item") else value
    return offer


if __name__ == "__main__":
    workdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workdir, "..", "data", DATA_VERSION)

    calendar_df = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    listings_df = pd.read_csv(os.path.join(data_dir, "listings.csv")).drop_duplicates(subset="id")
    listings_idx = listings_df.set_index("id")

    val_ids = pick_val_listing_ids(calendar_df["listing_id"], VAL_FRAC, SEED)
    val_rows = calendar_df[calendar_df["listing_id"].isin(val_ids) & calendar_df["price"].notna()]

    if len(val_rows) > MAX_REQUESTS:
        val_rows = val_rows.sample(MAX_REQUESTS, random_state=SEED)

    print(f"sending {len(val_rows)} val-set offers to {URL}")
    for _, cal_row in val_rows.iterrows():
        if cal_row["listing_id"] not in listings_idx.index:
            continue
        offer = build_offer(cal_row, listings_idx.loc[cal_row["listing_id"]])
        try:
            result = post(offer)
        except urllib.error.HTTPError as e:
            print(f"  skip ({e.code}): {offer}")
            continue
        print(
            f"price={offer['price']:>7.2f} -> "
            f"is_overpriced={result['is_overpriced']}, "
            f"predicted_price={result['predicted_price']:>7.2f}, "
            f"price_ratio={result['price_ratio']:.2f}"
        )
