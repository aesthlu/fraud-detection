"""API de scoring : charge le modèle exporté (fraud-detection export) et prédit la fraude.

Lancer en local : uv run uvicorn fraud_detection.api:app --reload
Documentation interactive : http://localhost:8000/docs
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))

# Une transaction a les mêmes colonnes que creditcard.csv (hors Class).
# V1..V28 sont générés plutôt qu'écrits à la main.
Transaction = create_model(
    "Transaction",
    Time=(float, Field(ge=0, description="Secondes écoulées depuis la première transaction du dataset")),
    **{f"V{i}": (float, ...) for i in range(1, 29)},  # type: ignore[call-overload]
    Amount=(float, Field(ge=0, description="Montant de la transaction")),
)
FEATURE_ORDER = list(Transaction.model_fields)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class BatchResponse(BaseModel):
    predictions: list[Prediction]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargé une seule fois au démarrage, pas à chaque requête.
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    state["metadata"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    yield
    state.clear()


app = FastAPI(title="Fraud detection API", lifespan=lifespan)


def score(transactions: list[BaseModel]) -> list[Prediction]:
    X = pd.DataFrame([t.model_dump() for t in transactions], columns=FEATURE_ORDER)
    probs = state["model"].predict_proba(X)[:, 1]
    threshold = state["metadata"]["threshold"]
    version = str(state["metadata"]["model_version"])
    return [
        Prediction(
            fraud_probability=float(p),
            is_fraud=bool(p >= threshold),
            threshold=threshold,
            model_version=version,
        )
        for p in probs
    ]


@app.get("/health")
def health() -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", **state["metadata"]}


@app.post("/predict")
def predict(transaction: Transaction) -> Prediction:  # type: ignore[valid-type]
    return score([transaction])[0]


@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=score(request.transactions))
