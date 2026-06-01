from dataclasses import dataclass, asdict
from typing import Annotated

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


@dataclass
class Config:
    random_state = 42
    n_estimators: Annotated[int, "Number of trees to build Isolation Forest"] = 100


@dataclass
class BaseConfig(Config):
    contamination: Annotated[float, "Suspected number of anomalies"] = 0.1


@dataclass
class AdvancedConfig(Config):
    learning_rate: Annotated[float, "Learning rate"] = None
    booster: Annotated[float, "Type of booster, e.g. gbtree, gblinear, dart"] = "gbtree"


def get_models(
    baseline_config: BaseConfig,
    advanced_config: AdvancedConfig,
    unsupervised_base=False,
):
    baseline_model = (
        IsolationForest(**asdict(baseline_config))
        if unsupervised_base
        else LogisticRegression(random_state=baseline_config.random_state)
    )
    advanced_model = xgb.XGBClassifier(**asdict(advanced_config))
    return {
        f"baseline_unsupervised_{unsupervised_base}": baseline_model,
        "advanced": advanced_model,
    }
