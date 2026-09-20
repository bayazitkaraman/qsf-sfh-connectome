import unittest

import numpy as np

from brain_quantum.decoder_robustness import MODES, RidgePath, choose_preprocessing, transform
from brain_quantum.methodology_controls import fit_ridge_arrays
from brain_quantum.source_coordinate_v2 import predict_ridge


class RobustDecoderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1337)
        self.x = rng.normal(size=(80, 9))
        self.y = rng.normal(size=(80, 3)) + 7

    def test_standard_svd_agrees_with_existing_well_conditioned_fit(self):
        path = RidgePath(self.x, self.y, "standard")
        for alpha in (0., .001, 25., 1000.):
            reference = fit_ridge_arrays(self.x, self.y, alpha)
            np.testing.assert_allclose(path.model(alpha).predict(self.x), predict_ridge(reference, self.x), atol=1e-12)

    def test_zero_penalty_uses_minimum_norm_design_solution(self):
        x = np.column_stack([self.x, self.x[:, 0], np.ones(len(self.x))])
        model = RidgePath(x, self.y, "standard").model(0.)
        z = (x - model.mean) / model.scale
        reference = np.linalg.lstsq(z, self.y - model.y_mean, rcond=None)[0]
        np.testing.assert_allclose(model.predict(x), z @ reference + model.y_mean, atol=1e-12)

    def test_transformed_ridge_matches_augmented_least_squares(self):
        for mode in MODES:
            model = RidgePath(self.x, self.y, mode).model(25.)
            z = transform((self.x - model.mean) / model.scale, mode) - model.transformed_mean
            design = np.vstack([z, 5 * np.eye(z.shape[1])])
            target = np.vstack([self.y - model.y_mean, np.zeros((z.shape[1], 3))])
            coef = np.linalg.lstsq(design, target, rcond=None)[0]
            np.testing.assert_allclose(model.coef, coef, atol=1e-12)

    def test_test_values_do_not_change_training_parameters(self):
        for mode in MODES:
            model = RidgePath(self.x, self.y, mode).model(25.)
            before = [v.copy() for v in (model.mean, model.scale, model.transformed_mean, model.coef)]
            prediction = model.predict(np.full((5, 9), 1e6))
            self.assertTrue(np.all(np.isfinite(prediction)))
            for expected, actual in zip(before, (model.mean, model.scale, model.transformed_mean, model.coef)):
                np.testing.assert_array_equal(expected, actual)
        self.assertLessEqual(np.abs(transform(np.array([-1e6, 1e6]), "tanh3")).max(), 3.)

    def test_constant_features_and_invalid_inputs(self):
        for mode in MODES:
            for alpha in (0., 25.):
                model = RidgePath(np.ones((80, 4)), self.y, mode).model(alpha)
                np.testing.assert_allclose(model.predict(np.ones((3, 4))), np.repeat(self.y.mean(axis=0)[None, :], 3, axis=0))
        with self.assertRaises(ValueError):
            RidgePath(np.empty((0, 3)), np.empty((0, 3)), "standard")
        with self.assertRaises(ValueError):
            RidgePath(self.x, self.y, "unknown")
        with self.assertRaises(ValueError):
            RidgePath(self.x, self.y, "standard").model(-1.)

    def test_preprocessing_choice_uses_validation_and_fixed_ties(self):
        grid = [{"preprocessing": mode, "alpha": alpha, "validation_nrmse": value, "validation_se": .01}
                for mode in MODES for alpha, value in ((0., .1), (25., .105))]
        self.assertEqual(choose_preprocessing(grid), ("standard", 25.))
        for row in grid:
            if row["preprocessing"] == "asinh":
                row["validation_nrmse"] -= .02
        self.assertEqual(choose_preprocessing(grid), ("asinh", 25.))


if __name__ == "__main__":
    unittest.main()
