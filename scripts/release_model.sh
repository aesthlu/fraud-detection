#!/usr/bin/env bash
# Publie le modèle champion du registry MLflow local comme release GitHub.
#
# Pourquoi : la CI/CD tourne sur les serveurs de GitHub et n'a pas accès à
# mlflow.db (local). La release sert de « registre public » du modèle :
# le workflow CD télécharge l'archive de la release indiquée dans
# configs/serving.yaml et l'embarque dans l'image Docker.
#
# Usage : ./scripts/release_model.sh
set -euo pipefail

cd "$(dirname "$0")/.."

uv run fraud-detection export --output model

VERSION=$(python3 -c 'import json; print(json.load(open("model/metadata.json"))["model_version"])')
TAG="model-v${VERSION}"

if gh release view "$TAG" >/dev/null 2>&1; then
    echo "La release $TAG existe déjà : rien à publier."
else
    tar -czf model.tar.gz model
    NOTES=$(python3 - <<'PY'
import json
m = json.load(open("model/metadata.json"))
lines = [
    f"Modèle `{m['model_name']}` version **{m['model_version']}** (run MLflow `{m['run_id']}`).",
    "",
    f"Seuil de décision : **{m['threshold']}**",
    "",
    "| Métrique (test) | Valeur |",
    "|---|---|",
]
lines += [f"| {k} | {v:.4f} |" for k, v in m["metrics"].items()]
print("\n".join(lines))
PY
)
    # --latest=false : les releases de modèle ne doivent pas passer pour des versions du code.
    gh release create "$TAG" model.tar.gz --title "Modèle v${VERSION}" --notes "$NOTES" --latest=false
    rm model.tar.gz
fi

# Le modèle déployé est choisi dans Git : changer cette ligne (via une PR) = déployer un autre modèle.
python3 - "$TAG" <<'PY'
import re, sys
path = "configs/serving.yaml"
text = open(path).read()
open(path, "w").write(re.sub(r"^model_release: .*$", f"model_release: {sys.argv[1]}", text, flags=re.M))
PY
echo "configs/serving.yaml pointe maintenant sur $TAG : commite ce changement pour le déployer."
