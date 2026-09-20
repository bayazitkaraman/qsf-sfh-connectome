"""Equivalent positive-ridge fallback for a nonconvergent direct design SVD."""
import numpy as np

from brain_quantum.decoder_robustness import Decoder, transform
from . import model


class AugmentedPath:
    """Compress the centered design with QR, then solve penalized least squares."""

    def __init__(self, x, y, scaling, count):
        if scaling != 'tanh3':
            raise ValueError('The additional controls require tanh3 preprocessing.')
        x, y = np.asarray(x, float), np.asarray(y, float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite fitting arrays.')
        self.mode = 'tanh3'
        self.mean = x.mean(axis=0, keepdims=True)
        self.scale = x.std(axis=0, keepdims=True)
        self.scale[self.scale < 1e-8] = 1.
        z = transform((x-self.mean)/self.scale, self.mode)
        self.transformed_mean = z.mean(axis=0, keepdims=True)
        z -= self.transformed_mean
        self.y_mean = y.mean(axis=0, keepdims=True)
        q, self.r = np.linalg.qr(z, mode='reduced')
        self.target = q.T @ (y-self.y_mean)
        _, _, self.rank, self.singular = np.linalg.lstsq(self.r, self.target, rcond=None)

    def model(self, alpha):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('The fallback requires a strictly positive ridge penalty.')
        columns = self.r.shape[1]
        design = np.vstack([self.r, np.sqrt(alpha)*np.eye(columns)])
        target = np.vstack([self.target, np.zeros((columns, self.target.shape[1]))])
        coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank != columns:
            raise np.linalg.LinAlgError('Augmented ridge design is numerically rank deficient.')
        return Decoder(self.mean, self.scale, self.transformed_mean, self.y_mean, coef, self.mode)


def fit(inner, final, fit_ids, val, train, count, scaling):
    """Retry only SVD nonconvergence, with the unchanged validation candidate set."""
    try:
        return model.fit(inner, final, fit_ids, val, train, count, scaling)
    except np.linalg.LinAlgError as exc:
        if 'SVD did not converge' not in str(exc):
            raise
    result = model.fit(inner, final, fit_ids, val, train, count, scaling,
                       path_factory=AugmentedPath)
    result[-1]['solver_recovery'] = 'QR plus augmented least squares'
    return result
