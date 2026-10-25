# fraud-detection

[![CI](https://github.com/aesthlu/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aesthlu/fraud-detection/actions/workflows/ci.yml)

Détection de fraude sur des transactions par carte bancaire (XGBoost), utilisée comme projet d'apprentissage MLOps.

## Installation

```bash
uv sync
```

## Données

Le dataset n'est pas versionné dans Git (144 Mo). Télécharger `creditcard.csv` depuis
[Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
et le placer dans `data/creditcard.csv`.

## Suivi des expériences

```bash
uv run mlflow ui
```

## Développement

```bash
uv run pre-commit install      # une fois : vérifications automatiques à chaque commit
uv run pytest                  # tous les tests (~30 s)
uv run pytest -m "not slow"    # tests unitaires seulement (~5 s)
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Les tests n'utilisent pas `data/creditcard.csv` : ils génèrent un petit dataset
synthétique avec les mêmes colonnes (`tests/conftest.py`), ce qui permet de les
lancer en CI sans télécharger les données.
