from pathlib import Path

from fraud_detection.config import flatten, load_config

PROJECT_CONFIG = Path(__file__).parents[1] / "configs" / "train.yaml"


def test_flatten_nested_dict():
    assert flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_project_config_is_valid():
    """La config livrée contient toutes les clés que le pipeline lit."""
    cfg = load_config(PROJECT_CONFIG)

    assert cfg["resampling"] in ("none", "smoteenn")
    assert cfg["mlflow"]["promotion_mode"] in ("minimize", "maximize")
    assert 0 < cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"] < 1
