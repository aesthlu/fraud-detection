"""Tests de l'API : un vrai modèle est entraîné sur les données synthétiques, exporté, puis servi."""

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from fraud_detection import api
from fraud_detection.pipeline import run_training
from fraud_detection.registry import export_champion

pytestmark = pytest.mark.slow


@pytest.fixture
def client(train_config, tmp_path, monkeypatch):
    run_training(train_config)
    model_dir = tmp_path / "model"
    export_champion(MlflowClient(), train_config["mlflow"]["registered_model_name"], model_dir)
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)

    with TestClient(api.app) as c:  # le "with" déclenche le chargement du modèle (lifespan)
        yield c


@pytest.fixture
def fraud(transactions) -> dict:
    return transactions[transactions["Class"] == 1].iloc[-1].drop("Class").to_dict()


@pytest.fixture
def normal(transactions) -> dict:
    return transactions[transactions["Class"] == 0].iloc[-1].drop("Class").to_dict()


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_version"] == 1
    assert 0 < body["threshold"] < 1
    assert "test_business_cost" in body["metrics"]


def test_predict_flags_obvious_fraud(client, fraud, normal):
    fraud_pred = client.post("/predict", json=fraud).json()
    normal_pred = client.post("/predict", json=normal).json()

    assert fraud_pred["is_fraud"] is True
    assert normal_pred["is_fraud"] is False
    assert fraud_pred["fraud_probability"] > normal_pred["fraud_probability"]
    assert fraud_pred["model_version"] == "1"


def test_decision_uses_exported_threshold(client, normal):
    pred = client.post("/predict", json=normal).json()

    assert pred["threshold"] == client.get("/health").json()["threshold"]
    assert pred["is_fraud"] == (pred["fraud_probability"] >= pred["threshold"])


def test_batch_returns_one_prediction_per_transaction(client, fraud, normal):
    response = client.post("/predict/batch", json={"transactions": [fraud, normal, normal]})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda t: t.pop("V3"), id="champ-manquant"),
        pytest.param(lambda t: t.update(Amount=-5), id="montant-negatif"),
        pytest.param(lambda t: t.update(V1="abc"), id="mauvais-type"),
    ],
)
def test_invalid_transaction_is_rejected(client, normal, change):
    change(normal)

    assert client.post("/predict", json=normal).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
