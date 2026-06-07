# Overpricing-Detection

Project aims at detection of price manipulation / overpricing rooms for rent

- run training pipeline for both models 
```sh
uv run python -m overpricing_detection.pipeline
```

- run microservice serving /predict endpoint allowing to get predicitions for new offers
```sh
uv run --active uvicorn app.main:app --host 0.0.0.0 --port 8080
```

- run script serving sample request to microservice api (as result predictions and log for A/B tests are generated)
```sh
uv run python scripts/sample_request.py
```

- example of curl api call
```sh
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"user_id\":\"xxx\",\"date\":\"2025-07-15\",\"price\":700,\"neighbourhood_group_cleansed\":\"Reinickendorf\",\"property_type\":\"Entire rental unit\",\"room_type\":\"Entire home/apt\",\"accommodates\":2,\"bathrooms\":1.0,\"bedrooms\":1.0,\"beds\":1.0}"

```

