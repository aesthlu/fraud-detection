import mlflow
import pytest
from mlflow import MlflowClient

from fraud_detection.registry import CHAMPION, promote_if_better

MODEL = "registry-test"


@pytest.fixture
def client(tmp_path, monkeypatch) -> MlflowClient:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # MLflow garde l'expérience active en mémoire entre les tests : on la redéfinit.
    mlflow.set_experiment("registry-test")
    c = MlflowClient()
    c.create_registered_model(MODEL)
    return c


def register(client: MlflowClient, cost: float) -> str:
    """Crée un run avec un coût donné et l'enregistre comme nouvelle version."""
    with mlflow.start_run() as run:
        mlflow.log_metric("test_business_cost", cost)
    version = client.create_model_version(
        MODEL, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id
    )
    return version.version


def champion_version(client: MlflowClient) -> str:
    return client.get_model_version_by_alias(MODEL, CHAMPION).version


def test_first_model_becomes_champion(client):
    v1 = register(client, cost=1500)

    assert promote_if_better(client, MODEL, v1, "test_business_cost", "minimize")
    assert champion_version(client) == v1


def test_lower_cost_replaces_champion(client):
    v1 = register(client, cost=1500)
    promote_if_better(client, MODEL, v1, "test_business_cost", "minimize")
    v2 = register(client, cost=1400)

    assert promote_if_better(client, MODEL, v2, "test_business_cost", "minimize")
    assert champion_version(client) == v2


@pytest.mark.parametrize("cost", [1500, 1600], ids=["tie", "worse"])
def test_champion_kept_when_not_strictly_better(client, cost):
    v1 = register(client, cost=1500)
    promote_if_better(client, MODEL, v1, "test_business_cost", "minimize")
    v2 = register(client, cost=cost)

    assert not promote_if_better(client, MODEL, v2, "test_business_cost", "minimize")
    assert champion_version(client) == v1


def test_maximize_mode_prefers_higher_score(client):
    v1 = register(client, cost=1500)
    promote_if_better(client, MODEL, v1, "test_business_cost", "maximize")
    v2 = register(client, cost=1600)

    assert promote_if_better(client, MODEL, v2, "test_business_cost", "maximize")


def test_invalid_mode_raises(client):
    v1 = register(client, cost=1500)

    with pytest.raises(ValueError, match="promotion_mode"):
        promote_if_better(client, MODEL, v1, "test_business_cost", "minimise")
