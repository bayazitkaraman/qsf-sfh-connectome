import unittest
from unittest.mock import patch

import numpy as np

from brain_quantum.linear_decoder import ridge_coefficients
from brain_quantum.methodology_controls import fit_ridge_arrays
from brain_quantum.source_coordinate_v2 import predict_ridge


class StableLinearDecoderTests(unittest.TestCase):
    def test_nearly_dependent_zero_penalty_uses_design_not_gram(self):
        rng = np.random.default_rng(41)
        base = rng.normal(size=(200, 8))
        x = np.column_stack([base, base[:, :4] + 1e-9 * rng.normal(size=(200, 4)), np.ones(200)])
        y = rng.normal(size=(200, 3))
        expected = np.linalg.lstsq(x, y, rcond=None)[0]
        np.testing.assert_array_equal(ridge_coefficients(x, y, 0.), expected)
        model = fit_ridge_arrays(x, y, 0.)
        design = (x - model.mean) / model.std
        coef = np.linalg.lstsq(design, y - model.y_mean, rcond=None)[0]
        np.testing.assert_allclose(predict_ridge(model, x), design @ coef + model.y_mean, atol=1e-12)

    def test_positive_penalty_and_fallback_match_augmented_design(self):
        rng = np.random.default_rng(18)
        x, y = rng.normal(size=(80, 10)), rng.normal(size=(80, 3))
        for alpha in (.001, 25., 1000.):
            design = np.vstack([x, np.sqrt(alpha) * np.eye(10)])
            target = np.vstack([y, np.zeros((10, 3))])
            expected = np.linalg.lstsq(design, target, rcond=None)[0]
            np.testing.assert_allclose(ridge_coefficients(x, y, alpha), expected, atol=1e-12)
            with patch("brain_quantum.linear_decoder.np.linalg.solve", side_effect=np.linalg.LinAlgError):
                np.testing.assert_allclose(ridge_coefficients(x, y, alpha), expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
