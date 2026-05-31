from dataclasses import dataclass, asdict
from typing import Annotated

from sklearn.ensemble import IsolationForest


@dataclass
class BaselineConfig:
    n_estimators: Annotated[int, "Number of trees to build Isolation Forest"]
    contamination: Annotated[float, "Suspected number of anomalies"]


if __name__ == "__main__":
    config = BaselineConfig(n_estimators=100, contamination=0.1)
    baseline_model = IsolationForest(**asdict(config), random_state=42)
    print(baseline_model)
