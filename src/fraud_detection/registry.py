from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"


def promote_if_better(client: MlflowClient, model_name: str, version: str, metric: str) -> bool:
    """Donne l'alias « champion » à la nouvelle version si elle bat l'actuel champion.

    L'API de serving chargera toujours `models:/<nom>@champion` : promouvoir un
    modèle revient donc à déplacer cet alias, sans redéployer de code.
    """
    new_run = client.get_run(client.get_model_version(model_name, version).run_id)
    new_score = new_run.data.metrics[metric]

    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        champion = None  # premier modèle enregistré

    if champion is not None:
        champion_score = client.get_run(champion.run_id).data.metrics.get(metric, float("-inf"))
        if new_score <= champion_score:
            print(f"v{version} ({metric}={new_score:.4f}) ne bat pas le champion "
                  f"v{champion.version} ({champion_score:.4f}) : pas de promotion.")
            return False

    client.set_registered_model_alias(model_name, CHAMPION, version)
    print(f"v{version} ({metric}={new_score:.4f}) est le nouveau champion.")
    return True
