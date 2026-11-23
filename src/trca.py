# Metody odniesienia TRCA i eTRCA, z kalibracją na blokach danej osoby. Służą tylko do porównania z siecią.
import numpy as np
from scipy.linalg import eigh


N_TARGETS = 12


def make_templates(X_cal, y_cal):
    """Average calibration trials into one channel-by-time template per target."""
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    if X_cal.ndim != 3 or y_cal.ndim != 1 or len(X_cal) != len(y_cal):
        raise ValueError("X_cal and y_cal must have shapes (trials, channels, samples) and (trials,)")
    if not np.array_equal(np.unique(y_cal), np.arange(N_TARGETS)):
        raise ValueError("Calibration data must contain all target labels 0 through 11")
    return np.stack(
        [X_cal[y_cal == target].mean(axis=0) for target in range(N_TARGETS)]
    )


def trca_filters(X_cal, y_cal):
    """Compute each target's leading generalized TRCA eigenvector."""
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    if X_cal.ndim != 3 or y_cal.ndim != 1 or len(X_cal) != len(y_cal):
        raise ValueError("X_cal and y_cal must have shapes (trials, channels, samples) and (trials,)")
    if not np.array_equal(np.unique(y_cal), np.arange(N_TARGETS)):
        raise ValueError("Calibration data must contain all target labels 0 through 11")

    filters = np.empty((X_cal.shape[1], N_TARGETS), dtype=float)
    for target in range(N_TARGETS):
        trials = X_cal[y_cal == target]
        centered = trials - trials.mean(axis=-1, keepdims=True)
        within_trial = np.einsum("ict,idt->cd", centered, centered)
        summed_trials = centered.sum(axis=0)
        between_trial = summed_trials @ summed_trials.T - within_trial
        between_trial = (between_trial + between_trial.T) / 2
        eigenvalues, eigenvectors = eigh(between_trial, within_trial)
        filters[:, target] = eigenvectors[:, np.argmax(eigenvalues)]
    return filters


def _pearson_correlation(first, second):
    first = first.ravel()
    second = second.ravel()
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def trca_predict(trial, templates, filters):
    """Predict a target from target-specific spatially filtered correlations."""
    trial = np.asarray(trial)
    correlations = np.empty(N_TARGETS, dtype=float)
    for target in range(N_TARGETS):
        weight = filters[:, target]
        correlations[target] = _pearson_correlation(
            weight @ trial, weight @ templates[target]
        )
    return int(np.argmax(correlations))


def etrca_predict(trial, templates, filters):
    """Predict using the flattened responses of all TRCA spatial filters."""
    trial = np.asarray(trial)
    correlations = np.empty(N_TARGETS, dtype=float)
    for target in range(N_TARGETS):
        correlations[target] = _pearson_correlation(
            filters.T @ trial, filters.T @ templates[target]
        )
    return int(np.argmax(correlations))
