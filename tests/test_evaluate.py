import numpy as np
import pytest

from fraud_detection.evaluate import business_cost, business_threshold, compute_metrics


def test_business_cost_counts_each_error_type():
    y_true = np.array([1, 1, 0, 0, 0])
    preds = np.array([0, 1, 1, 1, 0])  # 1 fraude manquée, 2 fausses alertes

    assert business_cost(y_true, preds, cost_fn=100, cost_fp=1) == 102


def test_threshold_is_low_when_missed_fraud_is_expensive():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    probs = np.clip(0.3 * y + rng.normal(0.2, 0.15, 2000), 0, 1)

    cheap_fn = business_threshold(y, probs, cost_fn=1, cost_fp=1)
    costly_fn = business_threshold(y, probs, cost_fn=100, cost_fp=1)

    assert costly_fn < cheap_fn


def test_perfect_model_metrics():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    m = compute_metrics(y, probs, threshold=0.5, cost_fn=100, cost_fp=1)

    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["business_cost"] == 0


def test_precision_is_zero_when_nothing_flagged():
    y = np.array([0, 1])
    probs = np.array([0.1, 0.2])

    m = compute_metrics(y, probs, threshold=0.9, cost_fn=100, cost_fp=1)

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["business_cost"] == pytest.approx(100)
