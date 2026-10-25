import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"
EXPORTED_METRICS = {
    "test_business_cost",
    "test_average_precision",
    "test_roc_auc",
    "test_precision",
    "test_recall",
}


def _run_id(model_version) -> str:
    """run_id d'une version du registry (une version créée hors run n'en a pas)."""
    if model_version.run_id is None:
        raise ValueError(f"La version {model_version.version} n'est liée à aucun run MLflow")
    return model_version.run_id


def promote_if_better(
    client: MlflowClient, model_name: str, version: str, metric: str, mode: str = "minimize"
) -> bool:
    """Donne l'alias « champion » à la nouvelle version si elle bat l'actuel champion.

    L'API de serving chargera toujours `models:/<nom>@champion` : promouvoir un
    modèle revient donc à déplacer cet alias, sans redéployer de code.
    `mode` vaut "minimize" pour un coût, "maximize" pour un score (AUC, AP...).
    """
    if mode not in ("minimize", "maximize"):
        raise ValueError(f"promotion_mode doit valoir 'minimize' ou 'maximize', pas {mode!r}")

    new_run = client.get_run(_run_id(client.get_model_version(model_name, version)))
    new_score = new_run.data.metrics[metric]

    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        champion = None  # premier modèle enregistré

    if champion is not None:
        champion_score = client.get_run(_run_id(champion)).data.metrics.get(metric)
        if champion_score is not None:
            is_better = new_score < champion_score if mode == "minimize" else new_score > champion_score
            if not is_better:
                print(
                    f"v{version} ({metric}={new_score:.4f}) ne bat pas le champion "
                    f"v{champion.version} ({champion_score:.4f}) : pas de promotion."
                )
                return False

    client.set_registered_model_alias(model_name, CHAMPION, version)
    print(f"v{version} ({metric}={new_score:.4f}) est le nouveau champion.")
    return True


def export_champion(client: MlflowClient, model_name: str, output_dir: str | Path) -> dict:
    """Copie le modèle champion dans un dossier local, avec ses métadonnées.

    L'API charge ce dossier : elle n'a pas besoin d'accéder au serveur MLflow
    pour démarrer, et l'image Docker contient exactement le modèle déployé.
    """
    output_dir = Path(output_dir)
    champion = client.get_model_version_by_alias(model_name, CHAMPION)
    threshold = champion.tags.get("threshold")
    if threshold is None:
        raise ValueError(f"La version {champion.version} n'a pas de tag 'threshold'")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}@{CHAMPION}", dst_path=str(output_dir)
    )

    # Métriques de test du run : elles suivent le modèle jusqu'en production
    # (notes de release, /health) et serviront de référence au monitoring.
    run_metrics = client.get_run(_run_id(champion)).data.metrics
    metadata = {
        "model_name": model_name,
        "model_version": champion.version,
        "run_id": champion.run_id,
        "threshold": float(threshold),
        "metrics": {k: v for k, v in sorted(run_metrics.items()) if k in EXPORTED_METRICS},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
