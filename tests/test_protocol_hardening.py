import unittest

import numpy as np
import torch

import hybrid_gnn


class ValidationLockedUncertaintyTests(unittest.TestCase):
    def test_threshold_uses_only_reference_mask(self):
        uncertainty = {
            "predictive_entropy": np.array([0.1, 0.2, 0.3, 10.0, 20.0])
        }
        validation_mask = np.array([True, True, True, False, False])

        threshold = hybrid_gnn.select_uncertainty_threshold(
            uncertainty, validation_mask, percentile=75.0
        )

        self.assertAlmostEqual(threshold, 0.25)

    def test_empty_reference_partition_is_rejected(self):
        uncertainty = {"predictive_entropy": np.array([0.1, 0.2])}
        with self.assertRaisesRegex(ValueError, "reference partition is empty"):
            hybrid_gnn.select_uncertainty_threshold(
                uncertainty, np.array([False, False])
            )


class CausalTemporalGraphTests(unittest.TestCase):
    def test_temporal_branch_never_aggregates_from_future_visit(self):
        months = np.array([0.0, 6.0, 12.0, 0.0])
        subjects = np.array(["A", "A", "A", "B"])

        raw = hybrid_gnn.build_temporal_adjacency_causal(
            months, subjects, temporal_lambda=0.2
        )

        rows, cols = np.nonzero(raw)
        self.assertTrue(np.all(months[cols] <= months[rows]))
        self.assertEqual(raw[0, 1], 0.0)
        self.assertGreater(raw[2, 1], 0.0)

    def test_population_causal_mask_does_not_change_cross_subject_edges(self):
        raw = np.array(
            [
                [0.0, 0.8, 0.4],
                [0.8, 0.0, 0.5],
                [0.4, 0.5, 0.0],
            ]
        )
        months = np.array([0.0, 12.0, 6.0])
        subjects = np.array(["A", "A", "B"])

        causal, stats = hybrid_gnn.enforce_causal_same_subject_edges(
            raw, months, subjects
        )

        self.assertEqual(causal[0, 1], 0.0)
        self.assertEqual(causal[1, 0], 0.8)
        self.assertEqual(causal[0, 2], 0.4)
        self.assertEqual(causal[2, 0], 0.4)
        self.assertEqual(stats["future_edges_removed"], 1)


class RenormalizedDropEdgeTests(unittest.TestCase):
    def test_undirected_pairs_are_dropped_together_and_renormalized(self):
        raw = torch.tensor(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.0, 0.7],
                [0.5, 0.7, 0.0],
            ],
            dtype=torch.float32,
        )
        torch.manual_seed(7)

        normalized = hybrid_gnn.drop_edge(raw, drop_rate=0.5)

        self.assertTrue(torch.allclose(normalized, normalized.T, atol=1e-7))
        self.assertTrue(torch.all(torch.diag(normalized) > 0))
        self.assertTrue(torch.isfinite(normalized).all())

    def test_directed_causal_edge_is_not_mirrored(self):
        raw = torch.tensor(
            [[0.0, 0.0], [0.8, 0.0]], dtype=torch.float32
        )

        normalized = hybrid_gnn.drop_edge(raw, drop_rate=0.0)

        self.assertEqual(float(normalized[0, 1]), 0.0)
        self.assertGreater(float(normalized[1, 0]), 0.0)

    def test_rate_one_keeps_only_self_loops(self):
        raw = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
        )
        normalized = hybrid_gnn.drop_edge(raw, drop_rate=1.0)
        self.assertTrue(torch.allclose(normalized, torch.eye(2)))


class SplitSafeNormalizationTests(unittest.TestCase):
    def test_masked_batch_statistics_ignore_held_out_rows(self):
        train_rows = torch.tensor(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float32
        )
        held_out_a = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
        held_out_b = torch.tensor([[1000.0, -1000.0]], dtype=torch.float32)
        mask = torch.tensor([True, True, True, False])

        norm_a = hybrid_gnn.SplitSafeBatchNorm1d(2)
        norm_b = hybrid_gnn.SplitSafeBatchNorm1d(2)
        norm_b.load_state_dict(norm_a.state_dict())
        norm_a.train()
        norm_b.train()

        output_a = norm_a(
            torch.cat([train_rows, held_out_a]), stats_mask=mask
        )
        output_b = norm_b(
            torch.cat([train_rows, held_out_b]), stats_mask=mask
        )

        self.assertTrue(torch.allclose(output_a[:3], output_b[:3], atol=1e-6))
        self.assertTrue(
            torch.allclose(norm_a.running_mean, norm_b.running_mean, atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(norm_a.running_var, norm_b.running_var, atol=1e-7)
        )

    def test_layer_normalization_has_no_running_batch_statistics(self):
        model = hybrid_gnn.SpectralGCN(
            in_dim=4,
            hidden_dim=8,
            num_classes=3,
            num_gcn_layers=2,
            normalization="layer",
            use_temporal_branch=False,
        )

        self.assertEqual(model.normalization, "layer")
        self.assertFalse(
            any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
        )
        self.assertTrue(
            any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
        )


class CounterfactualEnsembleTests(unittest.TestCase):
    class _FixedLogitModel(torch.nn.Module):
        def __init__(self, class_zero_logit):
            super().__init__()
            self.class_zero_logit = float(class_zero_logit)

        def forward(self, x, adjacency, **kwargs):
            logits = torch.tensor(
                [self.class_zero_logit, 0.0],
                dtype=x.dtype,
                device=x.device,
            ).repeat(x.shape[0], 1)
            return logits, None

    def test_counterfactual_prediction_uses_probability_ensemble(self):
        models = [
            self._FixedLogitModel(np.log(0.9 / 0.1)),
            self._FixedLogitModel(np.log(0.9 / 0.1)),
            self._FixedLogitModel(np.log(0.001 / 0.999)),
        ]
        x = torch.zeros((2, 1), dtype=torch.float32)
        adjacency = torch.eye(2, dtype=torch.float32)

        predictions, probabilities = hybrid_gnn._cf_ensemble_predict(
            models, x, adjacency, torch.device("cpu")
        )

        self.assertTrue(np.all(predictions == 0))
        self.assertTrue(np.allclose(probabilities[:, 0], (0.9 + 0.9 + 0.001) / 3))


if __name__ == "__main__":
    unittest.main()
