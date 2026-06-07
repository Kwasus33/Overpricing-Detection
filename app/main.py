import math
import os
import uuid
from datetime import datetime, timezone, date as date_type

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from overpricing_detection.pipeline import run_eval_pipeline
from app.core import DATA_DIR, log_prediction, offer_to_dataframe, pick_model

app = FastAPI(title="Overpricing Detection")

# threshold for flagging: actual price more than MULTIPLIER * predicted fair price
MULTIPLIER = 2


def load_models() -> dict:
    models = {}
    if not os.path.isdir(DATA_DIR):
        return models
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pkl"):
            name = file.removesuffix(".pkl").removesuffix("_v1")
            models[name] = joblib.load(os.path.join(DATA_DIR, file))
    return models


MODELS = load_models()


class Offer(BaseModel):
    user_id: str | None = None  # for A/B group
    date: date_type
    price: float
    neighbourhood_group_cleansed: str
    property_type: str | None = None
    room_type: str | None = None
    accommodates: int | None = None
    bathrooms: float | None = None
    bedrooms: float | None = None
    beds: float | None = None


class PredictionResponse(BaseModel):
    request_id: str
    is_overpriced: int
    predicted_price: float
    price_ratio: float


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


@app.post("/predict", response_model=PredictionResponse)
async def predict(offer: Offer) -> PredictionResponse:
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded")

    model_name = pick_model(list(MODELS.keys()), user_id=offer.user_id)
    pipeline = MODELS[model_name]

    X = offer_to_dataframe(offer.model_dump())
    y_pred_log = run_eval_pipeline(pipeline, X)
    predicted_price = float(math.exp(y_pred_log[0]))
    price_ratio = offer.price / predicted_price
    label = int(price_ratio > MULTIPLIER)

    request_id = str(uuid.uuid4())
    log_prediction(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model": model_name,
            "input": offer.model_dump(mode="json"),
            "is_overpriced": label,
            "predicted_price": predicted_price,
            "price_ratio": price_ratio,
            "user_id": offer.user_id,
        }
    )

    return PredictionResponse(
        request_id=request_id,
        is_overpriced=label,
        predicted_price=predicted_price,
        price_ratio=price_ratio,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
