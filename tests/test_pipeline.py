"""Tests de bout en bout : le pipeline complet tourne sur un petit dataset synthétique.

Ce sont eux qui auraient attrapé le bug skops de SMOTEENN : l'option n'avait
jamais été exécutée jusqu'à l'enregistrement du modèle.
"""

import mlflow
import pytest
from mlflow import MlflowClient

from fraud_detection.ingestion import temporal_split
from fraud_detection.pipeline import run_training
from fraud_detection.registry import CHAMPION

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("resampling", ["none", "smoteenn"])
def test_training_logs_run_and_registers_champion(train_config, resampling):
    train_config["resampling"] = resampling

    run_id = run_training(train_config)

    run = MlflowClient().get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["resampling"] == resampling
    for metric in ("threshold", "fit_seconds", "test_business_cost", "test_average_precision"):
        assert metric in run.data.metrics
    # Le signal synthétique est facile : un modèle qui marche le détecte.
    assert run.data.metrics["test_roc_auc"] > 0.9

    champion = MlflowClient().get_model_version_by_alias(
        train_config["mlflow"]["registered_model_name"], CHAMPION
    )
    assert champion.run_id == run_id


def test_champion_can_be_reloaded_and_predicts(train_config, transactions):
    """Ce que fera l'API de serving : charger le champion depuis le registry et prédire."""
    run_training(train_config)

    model = mlflow.sklearn.load_model(f"models:/{train_config['mlflow']['registered_model_name']}@{CHAMPION}")
    _, _, test = temporal_split(transactions, 0.6, 0.2)
    probs = model.predict_proba(test.drop(columns="Class"))[:, 1]

    assert probs.shape == (len(test),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_retraining_same_config_is_reproducible(train_config):
    first_id = run_training(train_config)
    second_id = run_training(train_config)
    # Le client est créé après run_training, qui configure la base MLflow du test.
    client = MlflowClient()
    first, second = client.get_run(first_id).data.metrics, client.get_run(second_id).data.metrics

    assert first["test_business_cost"] == second["test_business_cost"]
    assert first["test_average_precision"] == second["test_average_precision"]
