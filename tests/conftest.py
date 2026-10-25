import numpy as np
import pandas as pd
import pytest

N_FEATURES = 28  # V1..V28 dans le vrai dataset


def make_transactions(n_rows: int = 3000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Faux dataset avec exactement les colonnes de creditcard.csv.

    Les fraudes sont décalées sur V1-V3 et ont des montants plus élevés :
    le signal est facile à apprendre, ce qui permet de vérifier que le pipeline
    apprend réellement quelque chose, en quelques secondes.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) < fraud_rate).astype(int)
    V = rng.normal(size=(n_rows, N_FEATURES))
    V[:, :3] += 2.5 * y[:, None]

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, N_FEATURES + 1)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 2 * 24 * 3600, n_rows)))
    df["Amount"] = rng.exponential(50, n_rows) * (1 + 3 * y)
    df["Class"] = y
    return df


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture
def train_config(tmp_path, monkeypatch, transactions) -> dict:
    """Config d'entraînement isolée : CSV et MLflow dans un dossier temporaire."""
    # MLflow écrit ses artefacts dans ./mlruns : on se place dans tmp_path pour ne
    # pas polluer le projet.
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "transactions.csv"
    transactions.to_csv(data_path, index=False)
    return {
        "data": {"path": str(data_path), "target": "Class"},
        "split": {"train_ratio": 0.6, "val_ratio": 0.2},
        "resampling": "none",
        "model": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3, "random_state": 42},
        "costs": {"false_negative": 100, "false_positive": 1},
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "experiment_name": "test",
            "registered_model_name": "fraud-detection-test",
            "promotion_metric": "test_business_cost",
            "promotion_mode": "minimize",
        },
    }
