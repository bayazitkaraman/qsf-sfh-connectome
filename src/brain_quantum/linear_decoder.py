"""Numerically stable ridge coefficients for an already centered design."""
from __future__ import annotations

import numpy as np


def ridge_coefficients(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Use direct least squares at zero penalty, avoiding squared conditioning."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge penalty must be finite and nonnegative.")
    if alpha == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    system = design.T @ design + alpha * np.eye(design.shape[1])
    rhs = design.T @ target
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        augmented = np.vstack([design, np.sqrt(alpha) * np.eye(design.shape[1])])
        augmented_target = np.vstack([target, np.zeros((design.shape[1], target.shape[1]))])
        return np.linalg.lstsq(augmented, augmented_target, rcond=None)[0]
