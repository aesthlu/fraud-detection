# Image de serving : l'API FastAPI + le modèle champion exporté.
# Avant le build : uv run fraud-detection export   (crée le dossier model/)
# Build :          docker build -t fraud-detection-api .
# Lancement :      docker run -p 8000:8000 fraud-detection-api

FROM python:3.12-slim

# XGBoost a besoin d'OpenMP, absent de l'image slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv à la même version qu'en local et en CI.
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1. Dépendances seules : cette couche reste en cache tant que uv.lock ne change pas,
#    même si le code change. Les rebuilds sont alors beaucoup plus rapides.
#    --mount=type=cache : le cache de téléchargement de uv sert pendant le build
#    mais n'est pas copié dans l'image (il pesait 1,1 Go).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2. Le code du projet.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 3. Le modèle (change à chaque nouveau champion, donc en dernier).
COPY model ./model

# Ne jamais faire tourner un service en root dans un conteneur.
RUN useradd --create-home --uid 1000 app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_DIR=/app/model \
    MLFLOW_DISABLE_AGENT_HINT=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "fraud_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]
