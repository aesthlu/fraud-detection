import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.features import FeatureBuilder


def build_model(cfg: dict, y_train: pd.Series) -> Pipeline:
    """Pipeline complet : features -> (rééchantillonnage) -> XGBoost.

    On utilise le Pipeline d'imblearn : le rééchantillonnage n'est appliqué
    qu'au fit, jamais au predict.
    """
    steps = [("features", FeatureBuilder())]

    if cfg["resampling"] == "smoteenn":
        steps.append(("resampling", SMOTEENN(random_state=cfg["model"]["random_state"])))
        scale_pos_weight = 1.0  # les classes sont déjà rééquilibrées
    else:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    steps.append(
        (
            "classifier",
            XGBClassifier(
                **cfg["model"],
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
            ),
        )
    )
    return Pipeline(steps)
