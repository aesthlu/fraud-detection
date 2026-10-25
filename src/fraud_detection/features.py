import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SECONDS_PER_DAY = 24 * 3600


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Construit les features à partir des colonnes brutes du dataset.

    C'est un transformer scikit-learn : il fait partie du pipeline enregistré
    dans MLflow, donc l'inférence applique exactement les mêmes transformations
    que l'entraînement (pas de risque de « training/serving skew »).
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Heure de la journée (0-23) : les fraudes sont plus fréquentes la nuit.
        X["hour"] = (X["Time"] % SECONDS_PER_DAY) // 3600
        # Montants très asymétriques -> échelle logarithmique.
        X["Amount_log"] = np.log1p(X["Amount"])
        # "Time" brut (secondes depuis le début du dataset) ne généralise pas
        # au futur : on ne le garde pas comme feature.
        return X.drop(columns=["Time", "Amount"])
