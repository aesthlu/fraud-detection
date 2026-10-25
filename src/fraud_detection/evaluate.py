import matplotlib

matplotlib.use("Agg")  # pas de fenêtre graphique (indispensable en CI)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def business_cost(y_true, preds, cost_fn: float, cost_fp: float) -> float:
    fn = ((y_true == 1) & (preds == 0)).sum()
    fp = ((y_true == 0) & (preds == 1)).sum()
    return float(fn * cost_fn + fp * cost_fp)


def business_threshold(y_true, probs, cost_fn: float, cost_fp: float) -> float:
    """Seuil qui minimise le coût métier. À calculer sur la validation, jamais sur le test."""
    thresholds = np.linspace(0.001, 0.99, 990)
    costs = [business_cost(y_true, (probs >= t).astype(int), cost_fn, cost_fp) for t in thresholds]
    return float(thresholds[np.argmin(costs)])


def compute_metrics(y_true, probs, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((y_true == 1) & (preds == 1)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    return {
        "roc_auc": roc_auc_score(y_true, probs),
        "average_precision": average_precision_score(y_true, probs),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
        "business_cost": business_cost(y_true, preds, cost_fn, cost_fp),
    }


def plot_pr_curve(y_true, probs, threshold: float):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    idx = min(np.searchsorted(thresholds, threshold), len(thresholds) - 1)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.scatter(recall[idx], precision[idx], color="red", zorder=3, label=f"seuil = {threshold:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (test)")
    ax.legend()
    return fig
