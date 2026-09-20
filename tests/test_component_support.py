from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from brain_quantum.analysis_suite import compute_walk_stack
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.graph_propagation import component_support
from brain_quantum.null_controls import record_from_weights
from brain_quantum.qrc_connectome import Connectome, normalized_laplacian, walk_features
from brain_quantum.source_coordinate_v2 import anchor_features, cache_subject


class ComponentSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(707)
        self.groups = [[0, 2, 5, 8], [1, 4, 7], [3, 6, 9], [10], [11]]
        self.weights = np.zeros((12, 12))
        self.labels = np.empty(12, dtype=int)
        for label, group in enumerate(self.groups):
            self.labels[group] = label
            block = rng.uniform(0.2, 2.0, (len(group), len(group)))
            block = (block + block.T) / 2
            np.fill_diagonal(block, 0)
            self.weights[np.ix_(group, group)] = block
        self.support = self.labels[:, None] == self.labels[None, :]
        self.positions = rng.uniform(50, 150, (12, 3))
        self.times = [0.0, 0.25, 1.0, 8.0]
        self.sources = np.array([0, 1, 10])

    def record(self, weights=None, positions=None):
        graph = Connectome(
            path=Path("synthetic.graphml"), subject_id="synthetic",
            adjacency=self.weights if weights is None else weights,
            positions=self.positions if positions is None else positions,
            node_ids=[str(i) for i in range(12)],
            node_names=[str(i) for i in range(12)],
            hemispheres=["left"] * 6 + ["right"] * 6,
            regions=["synthetic"] * 12,
        )
        with patch("brain_quantum.source_coordinate_v2.parse_graphml", return_value=graph):
            return cache_subject(graph.path, self.times, "log1p")

    def test_all_channels_and_averages_have_exact_support(self):
        laplacian = normalized_laplacian(self.weights)
        np.testing.assert_array_equal(component_support(laplacian), self.support)
        stack = compute_walk_stack(laplacian, self.times)
        for key in ("q_real", "q_imag", "q_prob", "c_heat", "c_heat_raw"):
            for matrix in stack[key]:
                np.testing.assert_array_equal(matrix[~self.support], 0)
        for key in ("q_avg", "c_avg"):
            np.testing.assert_array_equal(stack[key][~self.support], 0)
        for real, imag, prob, heat in zip(stack["q_real"], stack["q_imag"], stack["q_prob"], stack["c_heat_raw"]):
            unitary = real + 1j * imag
            np.testing.assert_allclose(unitary @ unitary.conj().T, np.eye(12), atol=1e-12)
            np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-12)
            for isolated in (10, 11):
                self.assertAlmostEqual(real[isolated, isolated], 1)
                self.assertAlmostEqual(imag[isolated, isolated], 0)
                self.assertAlmostEqual(heat[isolated, isolated], 1)

    def test_float32_cache_and_null_rebuild_do_not_invent_moments(self):
        record = self.record()
        rebuilt = record_from_weights(record, record.weights, self.times)
        unreachable = np.array([3, 6, 9, 11])
        for candidate in (record, rebuilt):
            for method in ("qrc", "classical", "classical_raw"):
                features, response = anchor_features(candidate, self.sources, method)
                np.testing.assert_array_equal(response[unreachable], 0)
                np.testing.assert_array_equal(features[unreachable], 0)

    def test_no_reachable_positioned_source_gives_zero_moments(self):
        positions = self.positions.copy()
        positions[1] = np.nan
        record = self.record(positions=positions)
        for method in ("qrc", "classical", "classical_raw"):
            features, response = anchor_features(record, self.sources, method)
            self.assertTrue(np.any(response[4] > 0))
            np.testing.assert_array_equal(features[[4, 7], -6:], 0)
            self.assertTrue(np.all(np.isfinite(features)))
        positions[self.sources] = np.nan
        record = self.record(positions=positions)
        for method in ("qrc", "classical", "classical_raw"):
            features, _ = anchor_features(record, self.sources, method)
            np.testing.assert_array_equal(features[:, -6:], 0)

    def test_node_permutation_preserves_features(self):
        order = np.random.default_rng(901).permutation(12)
        inverse = np.argsort(order)
        original = self.record()
        permuted = self.record(self.weights[np.ix_(order, order)], self.positions[order])
        for method in ("qrc", "classical", "classical_raw"):
            before, _ = anchor_features(original, self.sources, method)
            after, _ = anchor_features(permuted, inverse[self.sources], method)
            np.testing.assert_allclose(before, after[inverse], atol=2e-5, rtol=1e-6)

    def test_legacy_and_dense_heat_paths_enforce_support(self):
        laplacian = normalized_laplacian(self.weights)
        q, heat, q_avg, c_avg = walk_features(laplacian, self.times)
        for blocks in (q, heat):
            for matrix in np.split(blocks, blocks.shape[1] // 12, axis=1):
                np.testing.assert_array_equal(matrix[~self.support], 0)
        for average in (q_avg, c_avg):
            np.testing.assert_array_equal(average[~self.support], 0)
        for matrix in dense_heat_kernels(self.record(), self.times):
            np.testing.assert_array_equal(matrix[~self.support], 0)

    def test_weak_within_component_responses_are_not_thresholded(self):
        weights = np.array([[0, 1, 0, 0], [1, 0, 1e-10, 0], [0, 1e-10, 0, 1], [0, 0, 1, 0]], dtype=float)
        laplacian = normalized_laplacian(weights)
        self.assertTrue(component_support(laplacian).all())
        stack = compute_walk_stack(laplacian, [0.25])
        self.assertGreater(stack["q_prob"][0][1, 2], 0)
        self.assertGreater(stack["c_heat_raw"][0][1, 2], 0)
        self.assertLess(stack["q_prob"][0][1, 2], 1e-18)
        values, vectors = np.linalg.eigh(laplacian)
        expected = (vectors * np.exp(-1j * values * 0.25)[None, :]) @ vectors.T
        np.testing.assert_array_equal(stack["q_real"][0], expected.real)
        np.testing.assert_array_equal(stack["q_imag"][0], expected.imag)

    def test_two_node_component_retains_exact_analytical_probability(self):
        weights = np.array([[0.0, 2.875, 0.0], [2.875, 0.0, 0.0], [0.0, 0.0, 0.0]])
        stack = compute_walk_stack(normalized_laplacian(weights), self.times)
        for time, probability in zip(self.times, stack["q_prob"]):
            self.assertAlmostEqual(probability[0, 1], np.sin(time) ** 2, places=12)
            self.assertEqual(probability[0, 2], 0)
            self.assertEqual(probability[2, 0], 0)

    def test_randomized_spectral_responses_match_independent_power_series(self):
        # A small scaled Taylor reference avoids sharing the production eigensolver.
        def exponential(matrix):
            scale = max(0, int(np.ceil(np.log2(max(1., np.linalg.norm(matrix, 1))))))
            scaled = matrix / 2 ** scale
            result = np.eye(len(matrix), dtype=matrix.dtype)
            term = result.copy()
            for k in range(1, 41):
                term = term @ scaled / k
                result += term
            for _ in range(scale):
                result = result @ result
            return result

        rng = np.random.default_rng(782)
        for _ in range(30):
            groups = rng.integers(0, 4, size=17)
            weights = rng.uniform(.05, 3., (17, 17))
            weights = (weights + weights.T) / 2
            weights[groups[:, None] != groups[None, :]] = 0
            np.fill_diagonal(weights, 0)
            laplacian = normalized_laplacian(weights)
            times = [.25, .5, 1., 2., 4., 8.]
            stack = compute_walk_stack(laplacian, times)
            for k, time in enumerate(times):
                actual = stack["q_real"][k] + 1j * stack["q_imag"][k]
                np.testing.assert_allclose(actual, exponential(-1j * time * laplacian), atol=3e-13, rtol=3e-13)
                np.testing.assert_allclose(stack["c_heat_raw"][k], exponential(-time * laplacian), atol=3e-13, rtol=3e-13)


if __name__ == "__main__":
    unittest.main()
