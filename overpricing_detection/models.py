from dataclasses import dataclass, asdict
from typing import Annotated

from sklearn.linear_model import LinearRegression
import xgboost as xgb


@dataclass
class Config:
    random_state: Annotated[int, "Seed for reproducibility"] = 42
    n_estimators: Annotated[int, "Number of trees"] = 100


@dataclass
class BaseConfig(Config):
    pass


@dataclass
class AdvancedConfig(Config):
    learning_rate: Annotated[float, "Learning rate"] = 0.05
    n_estimators: Annotated[int, "Number of trees"] = 500
    max_depth: Annotated[int, "Max tree depth"] = 5
    min_child_weight: Annotated[float, "Min samples per leaf (weighted)"] = 10
    subsample: Annotated[float, "Row subsample per tree"] = 0.8
    colsample_bytree: Annotated[float, "Feature subsample per tree"] = 0.8
    gamma: Annotated[float, "Min loss reduction to split"] = 0.1
    booster: Annotated[str, "Type of booster, e.g. gbtree, gblinear, dart"] = "gbtree"


def get_models(
    baseline_config: BaseConfig,
    advanced_config: AdvancedConfig,
):
    baseline_model = LinearRegression()
    advanced_model = xgb.XGBRegressor(**asdict(advanced_config))
    return {
        "baseline": baseline_model,
        "advanced": advanced_model,
    }
