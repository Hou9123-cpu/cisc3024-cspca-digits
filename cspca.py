"""Exact class-kernel CSPCA, after Papazoglou and Yin (2025), Eq. 15."""

import numpy as np


def fit_cspca(x, labels, n_components, kappa):
    """Return training pixel means and descending orthonormal projection axes."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2 or labels.shape != (x.shape[0],) or not len(labels):
        raise ValueError("Expected a non-empty matrix with one label per row")
    if not 1 <= n_components <= x.shape[1] or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Invalid number of components or kappa")
    mean = x.mean(axis=0)
    centered = x - mean
    classes, indices = np.unique(labels, return_inverse=True)
    class_sums = np.zeros((len(classes), x.shape[1]))
    np.add.at(class_sums, indices, centered)
    # class_sums.T @ class_sums equals X.T @ Delta @ X for the same-class kernel.
    objective = class_sums.T @ class_sums + kappa * (centered.T @ centered)
    _, axes = np.linalg.eigh(objective)
    return mean, axes[:, -n_components:][:, ::-1]
