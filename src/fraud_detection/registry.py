from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"


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
