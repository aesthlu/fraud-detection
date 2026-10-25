# fraud-detection

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
