"""Linear probing of representations for a binary clinical concept."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
) -> dict:
    """Train a logistic-regression probe on flattened `activations` to predict binary `labels`.

    Returns the fitted probe direction plus held-out AUC/accuracy. The direction itself is what
    later gets compared for stability across seeds/models (cosine similarity, principal angles).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    probe = LogisticRegression(max_iter=1000)
    probe.fit(x_train, y_train)
    scores = probe.predict_proba(x_test)[:, 1]
    return {
        "direction": probe.coef_[0],
        "bias": probe.intercept_[0],
        "auc": roc_auc_score(y_test, scores),
        "accuracy": probe.score(x_test, y_test),
    }
