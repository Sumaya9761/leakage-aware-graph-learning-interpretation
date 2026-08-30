import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import hybrid_gnn
import bio_leaflet


class FixedSubjectSplitTests(unittest.TestCase):
    def test_reuses_assignments_and_ignores_extra_subjects(self):
        cohort = pd.DataFrame(
            {
                "subject_id": ["A", "A", "B", "C", "D"],
                "DIAGNOSIS": ["0", "0", "0.5", "1+", "0.5"],
            }
        )
        assignments = pd.DataFrame(
            {
                "subject_id": ["A", "B", "C", "D", "EXTRA"],
                "split": ["train", "val", "test", "train", "test"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            split_file = Path(tmp) / "split.csv"
            assignments.to_csv(split_file, index=False)
            train, val, test = hybrid_gnn.subject_split_from_file(
                cohort, "subject_id", split_file
            )

        self.assertEqual(train.tolist(), [True, True, False, False, True])
        self.assertEqual(val.tolist(), [False, False, True, False, False])
        self.assertEqual(test.tolist(), [False, False, False, True, False])

    def test_missing_subject_assignment_is_rejected(self):
        cohort = pd.DataFrame({"subject_id": ["A", "B", "C"]})
        assignments = pd.DataFrame(
            {"subject_id": ["A", "B"], "split": ["train", "val"]}
        )

        with tempfile.TemporaryDirectory() as tmp:
            split_file = Path(tmp) / "split.csv"
            assignments.to_csv(split_file, index=False)
            with self.assertRaisesRegex(ValueError, "does not assign"):
                hybrid_gnn.subject_split_from_file(cohort, "subject_id", split_file)


class TargetLabelTests(unittest.TestCase):
    def tearDown(self):
        hybrid_gnn.configure_class_labels("diagnosis")

    def test_cdr_exports_do_not_use_diagnosis_names(self):
        hybrid_gnn.configure_class_labels("cdr")
        self.assertEqual(hybrid_gnn.CLASS_DISPLAY["0.5"], "CDR 0.5")
        self.assertEqual(hybrid_gnn.CLASS_EXPORT["0.5"], "CDR0_5")


class DecisionAdjustmentTests(unittest.TestCase):
    def test_offset_is_selected_from_validation_rows(self):
        probabilities = np.array(
            [
                [0.55, 0.40, 0.05],
                [0.05, 0.40, 0.55],
                [0.80, 0.15, 0.05],
                [0.05, 0.15, 0.80],
            ]
        )
        labels = np.array([1, 1, 0, 2])
        offset, score = hybrid_gnn.select_mci_log_probability_offset(
            probabilities, labels, offsets=[0.0, 0.5]
        )
        self.assertEqual(offset, 0.5)
        self.assertEqual(score, 1.0)

    def test_offset_probabilities_remain_normalized(self):
        probabilities = np.array([[0.6, 0.3, 0.1]])
        adjusted = hybrid_gnn.apply_mci_log_probability_offset(probabilities, 0.2)
        self.assertTrue(np.allclose(adjusted.sum(axis=1), 1.0))


class FeatureAttributionRoutingTests(unittest.TestCase):
    def test_patient_specific_evidence_is_preferred(self):
        gnn = {
            "patient_evidence": pd.DataFrame(
                {
                    "node_idx": [7, 7],
                    "pred_class": ["0.5", "0.5"],
                    "rank": [1, 2],
                    "feature": ["local_a", "local_b"],
                    "importance": [0.9, 0.8],
                }
            ),
            "feature_importance": pd.DataFrame(
                {
                    "class": ["0.5"],
                    "rank": [1],
                    "feature": ["global_a"],
                    "importance": [0.7],
                }
            ),
        }
        features, scope, _ = bio_leaflet.get_feature_attributions(
            gnn, 7, "0.5", top_k=5
        )
        self.assertEqual(scope, "patient-specific")
        self.assertEqual([row["feature"] for row in features], ["local_a", "local_b"])

    def test_class_profile_is_an_explicit_fallback(self):
        gnn = {
            "patient_evidence": pd.DataFrame(),
            "feature_importance": pd.DataFrame(
                {
                    "class": ["1+"],
                    "rank": [1],
                    "feature": ["global_a"],
                    "importance": [0.7],
                }
            ),
        }
        features, scope, label = bio_leaflet.get_feature_attributions(
            gnn, 9, "1+", top_k=5
        )
        self.assertEqual(scope, "predicted-class aggregate")
        self.assertIn("local evidence unavailable", label)
        self.assertEqual(features[0]["feature"], "global_a")

    def test_visit_specific_values_match_the_explained_session(self):
        gnn = {
            "uncertainty": pd.DataFrame(
                {
                    "node_idx": [7],
                    "pred_class": ["0.5"],
                    "true_class": ["0.5"],
                    "max_prob": [0.7],
                    "is_high_uncertainty": [False],
                    "predictive_entropy": [0.4],
                    "epistemic_uncertainty": [0.01],
                    "aleatoric_uncertainty": [0.39],
                    "pred_std": [0.02],
                }
            ),
            "patient_evidence": pd.DataFrame(
                {
                    "node_idx": [7],
                    "pred_class": ["0.5"],
                    "rank": [1],
                    "feature": ["clinical_MMSCORE"],
                    "importance": [0.9],
                }
            ),
            "feature_importance": pd.DataFrame(),
            "neighbors": pd.DataFrame(
                {
                    "node_idx": [7],
                    "rank": [1],
                    "graph_agreement_pct": [70.0],
                    "class_influence_pct_0": [10.0],
                    "class_influence_pct_0.5": [70.0],
                    "class_influence_pct_1+": [20.0],
                    "neighbor_idx": [8],
                    "influence_pct": [30.0],
                }
            ),
            "counterfactuals": pd.DataFrame({"node_idx": []}),
            "label_mapping": {"classes": ["0", "0.5", "1+"]},
        }
        patient = pd.DataFrame(
            {
                "clinical_session_id": ["bl", "m06", "m12"],
                "clinical_MMSCORE": [30.0, 20.0, 10.0],
            }
        )
        context = bio_leaflet.get_patient_context(
            "subject", 7, gnn, patient, session_id="m06"
        )
        self.assertEqual(context["top_features"][0]["value"], "20.00")


if __name__ == "__main__":
    unittest.main()
