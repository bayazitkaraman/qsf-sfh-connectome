"""Exact component support for spectral functions of undirected Laplacians."""
from __future__ import annotations

import numpy as np


def component_support(matrix: np.ndarray) -> np.ndarray:
    """Return reachability from nonzero off-diagonal entries, without a cutoff."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A square matrix is required.")
    adjacency = (matrix != 0) | (matrix.T != 0)
    np.fill_diagonal(adjacency, False)
    labels = np.full(len(matrix), -1, dtype=int)
    for start in range(len(matrix)):
        if labels[start] >= 0:
            continue
        labels[start] = start
        pending = [start]
        while pending:
            node = pending.pop()
            neighbors = np.flatnonzero(adjacency[node] & (labels < 0))
            labels[neighbors] = start
            pending.extend(neighbors.tolist())
    return labels[:, None] == labels[None, :]


def supported_spectral_matrix(
    eigenvectors: np.ndarray,
    factors: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Reconstruct a matrix function and remove only cross-component roundoff."""
    matrix = (eigenvectors * factors[None, :]) @ eigenvectors.T
    matrix[~support] = 0
    return matrix
