import os
from enum import Enum
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from overpricing_detection.pipeline import run_eval_pipeline

app = FastAPI()


def load_models():
    data_path = ["..", "data"]
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, *data_path)
    models = {
        file.split(".")[0]: file
        for file in os.listdir(model_path)
        if file.endswith(".pkl")
    }
    return models


MODELS = load_models()


class ModelType(str, Enum):
    baseline_supervised = "baseline_unsupervised_True"
    baseline_unsupervised = "baseline_unsupervised_False"
    advanced = "advanced"


class Offer(BaseModel):
    date: datetime
    available: str  # t/f
    reg_price: str  # needs removing '$'
    price: str  # needs removing '$'
    neighbourhood_group_cleansed: str  # onehot
    property_type: str  # onehot
    room_type: str  # onehot
    # # dunno if below needed
    # "accommodates",
    # "bathrooms",
    # "bedrooms",
    # "beds",
    # "amenities",  # to multi-hot
    # "availability_30",  ### it can be anomaly if listing was booked for most of time during 1 or 2 months and not booked only for few days when price went high
    # "availability_60",
    # "number_of_reviews_ltm",
    # "number_of_reviews_l30d",
    # "review_scores_rating",
    # "instant_bookable",  # to encode t==1, f==0
    # "calculated_host_listings_count",
    # "reviews_per_month",


@app.post("/predict/{model_type}")
async def predict(model_type: ModelType, new_offer: Offer):
    model = MODELS.get(model_type)
    if model:
        y_proba = run_eval_pipeline(model, model_type, new_offer)

    return {"Predictions": y_proba if y_proba else []}
