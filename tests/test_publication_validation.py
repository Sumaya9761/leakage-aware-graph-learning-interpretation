import unittest

import numpy as np

import publication_validation as pv


class PublicationValidationTests(unittest.TestCase):
    def test_probability_normalization(self):
        probabilities = np.array([[0.2, 0.3, 0.4999999]])
        normalized = pv.normalize_probabilities(probabilities)
        self.assertTrue(np.allclose(normalized.sum(axis=1), 1.0))

    def test_temperature_preserves_probability_rows(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        adjusted = pv.apply_temperature(probabilities, 1.7)
        self.assertTrue(np.allclose(adjusted.sum(axis=1), 1.0))
        self.assertTrue((adjusted >= 0).all())

    def test_mci_offset_only_changes_decision_scores(self):
        probabilities = np.array([[0.45, 0.40, 0.15], [0.10, 0.35, 0.55]])
        adjusted = pv.apply_mci_offset(probabilities, 0.5)
        self.assertTrue(np.allclose(adjusted.sum(axis=1), 1.0))
        self.assertTrue((adjusted[:, 1] > probabilities[:, 1]).all())

    def test_fusion_selection_uses_passed_validation_labels(self):
        gnn = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        )
        logistic = np.array(
            [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]
        )
        labels = np.array([0, 1, 2])
        weight, score, _ = pv.select_fusion_weight(gnn, logistic, labels)
        self.assertEqual(weight, 1.0)
        self.assertEqual(score, 1.0)

    def test_best_seed_is_selected_from_validation_scores(self):
        labels = np.array([0, 1, 2])
        good = np.eye(3)[labels]
        poor = np.roll(good, 1, axis=1)
        seed, score, scores = pv.select_best_validation_seed(
            {42: poor, 43: good}, labels
        )
        self.assertEqual(seed, 43)
        self.assertEqual(score, 1.0)
        self.assertLess(scores[42], scores[43])

    def test_cluster_bootstrap_returns_paired_intervals(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        subjects = np.array(["A", "B", "C", "D", "E", "F"])
        perfect = np.eye(3)[labels] * 0.9 + 0.1 / 3
        imperfect = np.full((6, 3), 1 / 3)
        result = pv.cluster_bootstrap(
            labels,
            subjects,
            {"perfect": perfect, "reference": imperfect},
            reference_name="reference",
            n_bootstrap=100,
            seed=7,
        )
        self.assertGreater(result["accepted_replicates"], 0)
        self.assertIn("perfect", result["paired_difference_vs_reference"])


if __name__ == "__main__":
    unittest.main()
