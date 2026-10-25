import hashlib

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from fraud_detection.config import flatten
from fraud_detection.evaluate import business_threshold, compute_metrics, plot_pr_curve
from fraud_detection.ingestion import load_data, temporal_split
from fraud_detection.registry import promote_if_better
from fraud_detection.train import build_model

TRUSTED_TYPES = [
    "fraud_detection.features.FeatureBuilder",
    "imblearn.pipeline.Pipeline",
    "imblearn.combine._smote_enn.SMOTEENN",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_training(cfg: dict) -> str:
    """Entraîne, évalue et enregistre le modèle dans un seul run MLflow. Retourne le run_id."""
    target = cfg["data"]["target"]
    cost_fn, cost_fp = cfg["costs"]["false_negative"], cfg["costs"]["false_positive"]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = load_data(cfg["data"]["path"])
    train, val, test = temporal_split(df, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"])
    X_train, y_train = train.drop(columns=target), train[target]
    X_val, y_val = val.drop(columns=target), val[target]
    X_test, y_test = test.drop(columns=target), test[target]

    with mlflow.start_run() as run:
        # 1. Traçabilité : config, version des données, tailles des splits
        mlflow.log_params(flatten(cfg))
        mlflow.set_tag("data_md5", file_md5(cfg["data"]["path"]))
        mlflow.log_input(mlflow.data.from_pandas(train, targets=target, name="train"), context="training")
        for name, split in [("train", train), ("val", val), ("test", test)]:
            mlflow.log_metric(f"{name}_rows", len(split))
            mlflow.log_metric(f"{name}_fraud_rate", split[target].mean())

        # 2. Entraînement
        model = build_model(cfg, y_train)
        model.fit(X_train, y_train)

        # 3. Seuil choisi sur la validation, performance mesurée sur le test
        threshold = business_threshold(y_val, model.predict_proba(X_val)[:, 1], cost_fn, cost_fp)
        mlflow.log_metric("threshold", threshold)

        for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            probs = model.predict_proba(X)[:, 1]
            metrics = compute_metrics(y, probs, threshold, cost_fn, cost_fp)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})

        mlflow.log_figure(plot_pr_curve(y_test, probs, threshold), "pr_curve.png")

        # 4. Modèle + signature (schéma des entrées) + enregistrement dans le registry
        model_name = cfg["mlflow"]["registered_model_name"]
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(X_test.head(100), model.predict_proba(X_test.head(100))),
            input_example=X_test.head(5),
            registered_model_name=model_name,
            # MLflow sérialise avec skops (format sûr, contrairement à pickle) :
            # on déclare explicitement les classes autorisées au chargement.
            skops_trusted_types=TRUSTED_TYPES,
        )
        client = MlflowClient()
        version = str(info.registered_model_version)
        client.set_model_version_tag(model_name, version, "threshold", f"{threshold:.4f}")

        print(f"Run {run.info.run_id} — test AP = {metrics['average_precision']:.4f}, "
              f"seuil = {threshold:.2f}")

    promote_if_better(client, model_name, version, cfg["mlflow"]["promotion_metric"])
    return run.info.run_id
