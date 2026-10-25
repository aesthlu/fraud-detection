from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten(cfg: dict, prefix: str = "") -> dict:
    """Aplati la config ({"model": {"max_depth": 6}} -> {"model.max_depth": 6}) pour mlflow.log_params."""
    flat = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat
