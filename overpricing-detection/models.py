from dataclasses import dataclass, asdict
from typing import Annotated

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


@dataclass
class Config:
    n_estimators: Annotated[int, "Number of trees to build Isolation Forest"] = 100
    random_state = 42


@dataclass
class Baseline(Config):
    contamination: Annotated[float, "Suspected number of anomalies"] = 0.1


@dataclass
class Advanced(Config):
    learning_rate: Annotated[float, "Learning rate"] = None
    booster: Annotated[float, "Type of booster, e.g. gbtree, gblinear, dart"] = "gbtree"


def get_models(baseline_config: Baseline, advanced_config: Advanced):
    baseline_model = IsolationForest(**asdict(baseline_config))
    mid_model = LogisticRegression(random_state=42)
    advanced_model = xgb.XGBClassifier(**asdict(advanced_config))
    return {
        "Isolation Forest": baseline_model,
        "Logistic Regression": mid_model,
        "XGBoost": advanced_model,
    }
